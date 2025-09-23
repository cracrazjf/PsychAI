from __future__ import annotations

import os
import json
import ast
from tqdm import tqdm
import threading
from pathlib import Path
from functools import partial
from ..model_manager.llm import LLM_ModelManager
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import TextIteratorStreamer

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.use_unsloth = config.USE_UNSLOTH
        self.models_root = config.MODELS_ROOT
        self.model_cache_root = config.MODEL_CACHE_ROOT
        self.data_root = config.DATA_ROOT
        self.model_manager = LLM_ModelManager()

    def list_available_models(self) -> Dict[str, str]:
        """Return {name: path} for subdirectories under models_root."""
        out: Dict[str, str] = {}
        root = Path(self.models_root)
        cache_root = Path(self.model_cache_root)
        if not root.exists() and not cache_root.exists():
            return out
        if root.exists():
            for item in root.iterdir():
                if item.is_dir():
                    for dirpath, _, filenames in os.walk(item):
                        has_config = "config.json" in filenames
                        has_adapter_config = "adapter_config.json" in filenames
                        has_safetensor = any(f.endswith(".safetensors") for f in filenames)
                        if has_config or has_adapter_config and has_safetensor:
                            out[item.name] = str(item)
        if cache_root.exists():
            for item in cache_root.iterdir():
                if item.is_dir():
                    for dirpath, _, filenames in os.walk(item):
                        has_config = "config.json" in filenames
                        has_safetensor = any(f.endswith(".safetensors") for f in filenames)
                        if has_config and has_safetensor:
                            model_name = "/".join(item.name.split("--")[1:])
                            out[model_name] = model_name
        return dict(sorted(out.items()))

    def load_model_and_tokenizer(self, model_name: str, model_path: str, reasoning: bool, max_seq_length: int, load_in_4bit: bool, dtype: str) -> Tuple[Any, Any]:
        self.model_manager.load_model(model_name, 
                                      model_path, 
                                      reasoning=reasoning, 
                                      use_unsloth=self.use_unsloth, 
                                      for_training=False, 
                                      max_seq_length=max_seq_length, 
                                      load_in_4bit=load_in_4bit, 
                                      full_finetuning=False, 
                                      dtype=dtype)
        self.device = next(self.model_manager.model.parameters()).device

    def list_available_datasets(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        base_path = Path(self.data_root)
        if not base_path.exists():
            return out
        for item in base_path.iterdir():
            if item.is_dir():
                for root, dirs, files in os.walk(item):
                    for dir in dirs:
                        if "processed" in dir:
                            processed_path = Path(root) / dir
                            test_json: Optional[str] = None
                            for j in processed_path.glob("test*.json*"):
                                test_json = str(j)
                                break
                            out[item.name] = test_json
        return dict(sorted(out.items()))

    def format_chat(self, examples: Any, reasoning_effort: Optional[str] = None) -> dict:
        convos = examples["messages"]
        labels = []
        texts = []
        for conv in convos:
            last_ass = next((m for m in reversed(conv) if m["role"] == "assistant"), None)
            label = last_ass["content"] if last_ass else ""

            if conv and conv[-1]["role"] == "assistant":
                conv = conv[:-1]
            
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": True
            }
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            texts.append(self.model_manager.tokenizer.apply_chat_template(conv, **kwargs))
            labels.append(label)

        return {"text": texts, "label": labels}

    def format_instruction(self, examples: Any, prompt_template: Optional[str] = None) -> dict:
        texts = []
        labels = []
        if prompt_template is None:
            prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            {}
            
            ### Input:
            {}
            
            ### Response:
            {}"""
        for example in examples:
            text = prompt_template.format(example["instruction"], example.get("input", ""), "")
            texts.append(text)
            labels.append(example["output"])

        return {"text": texts, "label": labels}

    def get_analysis_and_final_re(self) -> Tuple[str, str]:
        if self.model_manager.model_company == "gpt":
            analysis_re = r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>"
            final_re = r"<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)<\|return\|>"
        return analysis_re, final_re
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             generate_args: Dict[str, Any]):
        
        reasoning_effort = generate_args.get("reasoning_effort", None)
        
        kwargs = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        
        formatted_inputs = self.model_manager.tokenizer.apply_chat_template(messages, **kwargs)
        formatted_inputs = formatted_inputs.to(self.device)
        streamer = TextIteratorStreamer(self.model_manager.tokenizer, skip_prompt=True, skip_special_tokens=not self.model_manager.reasoning)

        def generate_response():
            with torch.no_grad():
                self.model_manager.model.generate(
                    **formatted_inputs,
                    streamer=streamer,
                    max_new_tokens=generate_args["max_new_tokens"],
                    temperature=generate_args["temperature"],
                    do_sample=generate_args["do_sample"],
                    top_p=generate_args["top_p"],
                    top_k=generate_args["top_k"],
                )
        
        def to_user(s):
            print(s, end="", flush=True)

        def stream_with_labels(chunks, analysis_re, final_re, to_user):
            analysis_open = analysis_re.split("(.*?)")[0].replace("\\", "")
            analysis_close = analysis_re.split("(.*?)")[1].replace("\\", "")
            final_open = final_re.split("(.*?)")[0].replace("\\", "")
            final_close = final_re.split("(.*?)")[1].replace("\\", "")
            buf = ""
            mode = None
            printed_thinking_hdr = False
            printed_model_hdr = False
            tail_keep = max(len(analysis_open), len(analysis_close),
                            len(final_open),   len(final_close)) - 1

            def write_thinking(s):
                nonlocal printed_thinking_hdr
                if not s: return
                if not printed_thinking_hdr:
                    to_user("\nThinking: ")
                    printed_thinking_hdr = True
                to_user(s)

            def write_model(s):
                nonlocal printed_model_hdr
                if not s: return
                if not printed_model_hdr:
                    to_user("\nModel: ")
                    printed_model_hdr = True
                to_user(s)

            for chunk in chunks:
                buf += chunk
                while True:
                    if mode is None:
                        ai = buf.find(analysis_open)
                        fi = buf.find(final_open)
                        if ai == fi == -1:
                            if len(buf) > tail_keep:
                                buf = buf[-tail_keep:]
                            break
                        if ai != -1 and (fi == -1 or ai < fi):
                            buf = buf[ai + len(analysis_open):]
                            mode = "analysis"
                        else:
                            buf = buf[fi + len(final_open):]
                            mode = "final"
                    elif mode == "analysis":
                        ci = buf.find(analysis_close)
                        if ci == -1:
                            if len(buf) > tail_keep:
                                write_thinking(buf[:-tail_keep])
                                buf = buf[-tail_keep:]
                            break
                        else:
                            write_thinking(buf[:ci])
                            buf = buf[ci + len(analysis_close):]
                            mode = None
                    else:
                        ci = buf.find(final_close)
                        if ci == -1:
                            if len(buf) > tail_keep:
                                write_model(buf[:-tail_keep])
                                buf = buf[-tail_keep:]
                            break
                        else:
                            write_model(buf[:ci])
                            return

            if mode == "analysis" and buf:
                write_thinking(buf)
            elif mode == "final" and buf:
                write_model(buf)

        thread = threading.Thread(target=generate_response)
        thread.start()
        if self.model_manager.reasoning:
            analysis_re, final_re = self.get_analysis_and_final_re()
            stream_with_labels(streamer, analysis_re, final_re, to_user)
        else:
            print("\nModel: ", end="", flush=True)
            for new_text in streamer:
                print(new_text, end="", flush=True)
        thread.join()
        print()

    def evaluate_formatted_text(self,
                                data: Any,  
                                data_type: str,
                                batch_size: int,
                                generate_args: Dict[str, Any],
                                result_dir: str,
                                prompt_template: Optional[str] = None
                                ):

        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{data_type}_formatted_results.jsonl"   
        open(result_path, "w").close()

        reasoning_effort = generate_args.get("reasoning_effort", None)
        if data_type == "chat":
            data = data.map(partial(self.format_chat, reasoning_effort=reasoning_effort), batched=True)
        elif data_type == "instruction":
            data = data.map(partial(self.format_instruction, prompt_template=prompt_template), batched=True)
        else:
            raise ValueError(f"Invalid data type: {data_type}")

        def collate_for_generate(batch):
            prompts = [ex["text"] for ex in batch]
            enc = self.model_manager.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            enc["labels"] = [ex["label"] for ex in batch]
            return enc

        loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_for_generate)

        sample_id = 0
        tqdm_loader = tqdm(loader, desc="Evaluating Formatted Text")
        for batch in tqdm_loader:
            # store the labels
            labels = batch.pop("labels")  

            # generate the outputs
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model_manager.model.generate(**batch,
                                                        max_new_tokens = generate_args["max_new_tokens"], 
                                                        temperature = generate_args["temperature"],
                                                        do_sample = generate_args["do_sample"],
                                                        top_p = generate_args["top_p"],
                                                        top_k = generate_args["top_k"],
                                                        use_cache = True,
                                                        return_dict_in_generate=True
                                                        )
            
            # get corresponding scores and tokens
            sequences = outputs.sequences
            # decide the input length
            input_len = batch["input_ids"].size(1)

            if data_type == "instruction":
                pass
                # decoded_outputs = self.model_manager.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                # predictions = []
                # for decoded_output in decoded_outputs:
                #     head, sep, tail = decoded_output.rpartition("### Response:")
                #     pred = (tail if sep else decoded_output).strip()
                #     predictions.append(pred)
                # pred_texts.extend(predictions)
            elif data_type == "chat":
                new_tokens = sequences[:, input_len:]
                eos_ids = torch.tensor(self.model_manager.model.generation_config.eos_token_id, device=self.device)
                valid_new_tokens_mask = ~torch.isin(new_tokens, eos_ids)
                valid_new_tokens_length = valid_new_tokens_mask.sum(dim=1)

                for mini_batch_idx, valid_length in enumerate(valid_new_tokens_length):
                    valid_new_tokens = new_tokens[mini_batch_idx, :valid_length]
                    if self.model_manager.reasoning:
                        decoded_valid_new_tokens = self.model_manager.tokenizer.decode(valid_new_tokens, skip_special_tokens=False)
                    else:
                        decoded_valid_new_tokens = self.model_manager.tokenizer.decode(valid_new_tokens, skip_special_tokens=True)

                    result = {
                        "sample_id": sample_id,
                        "prompt": self.model_manager.tokenizer.decode(batch["input_ids"][mini_batch_idx], skip_special_tokens=True),
                        "decoded_tokens": decoded_valid_new_tokens,
                        "label": labels[mini_batch_idx],
                        "token_ids": valid_new_tokens.tolist()
                        }

                    if self.model_manager.reasoning:
                        analysis_re, final_re = self.get_analysis_and_final_re()
                        ANALYSIS_RE = re.compile(analysis_re, re.DOTALL | re.IGNORECASE)
                        FINAL_RE = re.compile(final_re, re.DOTALL | re.IGNORECASE)
                        analysis_match = ANALYSIS_RE.search(decoded_valid_new_tokens)
                        final_match = FINAL_RE.search(decoded_valid_new_tokens)
                        if analysis_match:
                            result["analysis"] = analysis_match.group(1).strip()
                        if final_match:
                            result["final"] = final_match.group(1).strip()

                    with open(result_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

                    sample_id += 1

    def evaluate_plain_text(self,
                            data: Any,
                            batch_size: int,
                            result_dir: str,
                            layer: list[int]= [-1],
                            top_k: int = 50,
                            output_hidden_states: bool = False) -> Dict[str, Dict[str, Optional[float]]]:

        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"plain_text_results.jsonl"
        open(result_path, "w").close()

        def collate_for_activation(batch):
            prompts = [ex["text"] for ex in batch]
            enc = self.model_manager.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            return enc
        
        loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_for_activation, pin_memory=True)

        sample_id = 0
        shard_id = 0
        results = []
        heavy_results = []
        tqdm_loader = tqdm(loader, desc="Evaluating Plain Text")
        for batch in tqdm_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model_manager.model(**batch,
                                                return_dict=True,
                                                output_logits=True,
                                                output_hidden_states=output_hidden_states
                                                )

            mask = batch["attention_mask"].bool()
            input_ids = batch["input_ids"]
            logits = outputs.logits
            topk_logits, topk_logits_ids = torch.topk(logits, k=top_k, dim=-1)

            for i, m in enumerate(mask):
                shift_label_ids = input_ids[i][m][1:]
                shift_logits = logits[i][m][:-1, :]
                chosen_logits = shift_logits.gather(1, shift_label_ids.unsqueeze(1)).squeeze(1)
                last_valid_index = int(m.sum().item() - 1)

                result = {
                    "sample_id": sample_id,
                    "input_ids": input_ids[i][m].tolist(),
                    "decoded_input": [self.model_manager.tokenizer.decode(id, skip_special_tokens=False) for id in input_ids[i][m]],
                    "label": [self.model_manager.tokenizer.decode(id, skip_special_tokens=False) for id in shift_label_ids],
                    "chosen_logits": chosen_logits.tolist(),
                    "topk_logits": topk_logits[i][m].tolist(),
                    "topk_logits_ids": topk_logits_ids[i][m].tolist(),
                }

                heavy_result = {
                    "sample_id": sample_id,
                    "last_logits": logits[i][last_valid_index].detach().cpu().to(torch.float16),
                    "last_hiddens": torch.stack([outputs.hidden_states[l][i][last_valid_index] for l in layer], dim=0).detach().cpu().to(torch.float16)
                }
                
                results.append(result)
                heavy_results.append(heavy_result)

                with open(result_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

                def _flush_buffer(buffer):
                    V = buffer[0]["last_logits"].shape[0]
                    L, H = buffer[0]["last_hiddens"].shape
                    N = len(buffer)

                    logits_shard = np.empty((N, V), dtype=np.float16)
                    hiddens_shard = np.empty((N, L, H), dtype=np.float16)
                    sids = []

                    for i, row in enumerate(buffer):
                        logits_shard[i] = row["last_logits"].numpy()
                        hiddens_shard[i] = row["last_hiddens"].numpy()
                        sids.append(row["sample_id"])

                    sample_ids = np.array(sids, dtype=np.int32)

                    shard_path = result_dir / "heavy_results_shard_{:05d}.fp16.npz".format(shard_id)
                    np.savez_compressed(shard_path,
                                        logits=logits_shard,
                                        hiddens=hiddens_shard,
                                        sample_ids=sample_ids,
                                        vocab_size=np.array([V], dtype=np.int32),
                                        num_layers=np.array([L], dtype=np.int32),
                                        hidden_size=np.array([H], dtype=np.int32))
                
                if len(heavy_results) >= 667:
                    _flush_buffer(heavy_results)
                    shard_id += 1
                    heavy_results = []
                sample_id += 1

        _flush_buffer(heavy_results)

    def evaluate_text(
        self,
        model_name: str,
        model_path: str,
        reasoning: bool,
        data_map: Dict[str, Any],
        data_type: str,
        batch_size: int,
        result_dir: str,
        layer: list[int] = [-1],
        output_hidden_states: bool = False,
        max_samples: Optional[int] = None,
        model_args: Optional[Dict[str, Any]] = None,
        generate_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Optional[float]]]:

        self.load_model_and_tokenizer(model_name, 
                                      model_path, 
                                      reasoning, 
                                      model_args["max_seq_length"], 
                                      model_args["load_in_4bit"], 
                                      model_args["dtype"])

        for data_name, data_path in data_map.items():
            data = load_dataset("json", data_files=data_path, split="train")
            if max_samples:
                data = data.select(range(max_samples))
            result_dir = f"{result_dir}/{model_name}_{data_name}"

            if data_type == "plain":
                self.evaluate_plain_text(data,
                                        batch_size=batch_size,
                                        layer=layer,
                                        result_dir=result_dir,
                                        top_k=generate_args["top_k"],
                                        output_hidden_states=output_hidden_states)
            else:
                self.evaluate_formatted_text(data,
                                            data_type,
                                            batch_size=batch_size,
                                            generate_args=generate_args,
                                            result_dir=result_dir,
                                            prompt_template=prompt_template
                                            )

    def interactive_text(self) -> None:
        models = self.list_available_models()
        datasets = self.list_available_datasets()

        def _get_generate_args(is_reasoning: bool):
            default_generate_args = self.config.GENERATE_ARGS
            
            prompts = {
                "max_new_tokens": (int, "Please enter the maximum number of new tokens: "),
                "temperature": (float, "Please enter the temperature: "),
                "do_sample": (lambda x: x.lower() in ["true", "1", "yes"], "Please enter whether to sample tokens [True/False]: "),
                "top_p": (float, "Please enter top_p: "),
                "top_k": (int, "Please enter top_k: "),
            }
            if is_reasoning:
                prompts["reasoning_effort"] = (str, "Please enter the reasoning effort [low/medium/high]: ")

            generate_args = {}
            print("\nLeave blank if you want to use the default value")
            for key, (cast, message) in prompts.items():
                raw = input(message).strip()
                if raw == "":
                    generate_args[key] = default_generate_args[key]
                else:
                    try:
                        generate_args[key] = cast(raw)
                    except ValueError:
                        print(f"⚠️ Invalid input for {key}, using default.")
                        generate_args[key] = default_generate_args[key]

            print("\nThis model will use the following generate args:")
            for k, v in generate_args.items():
                print(f"{k}: {v}")

            return generate_args

        def _get_model_args():
            while True:
                reasoning_in = input("Is this a reasoning model? (y/n): ").strip().lower()
                if reasoning_in in ("y", "yes"):
                    reasoning = True
                    break
                elif reasoning_in in ("n", "no"):
                    reasoning = False
                    break
                else:
                    print("⚠️ Please enter 'y' or 'n'.")
            prompts = {
                    "max_seq_length": (int, "Please enter maximum context window size: ", 2048),
                    "load_in_4bit": (lambda x: x.lower() in ["true", "1", "yes"], "Load the model in 4bit [True/False]: ", True),
                    "dtype": (str, "Please enter dtype: ", None),
                }
            model_args = {}
            print("\nLeave blank if you want to use the default value")
            for key, (cast, message, default) in prompts.items():
                raw = input(message).strip()
                if raw == "":
                    model_args[key] = default
                else:
                    try:
                        model_args[key] = cast(raw)
                    except ValueError:
                        print(f"⚠️ Invalid input for {key}, using default.")
                        model_args[key] = default
            return reasoning, model_args

        print("\n Welcome to PSYCHAI Interactive LLM Evaluator")
        print("Commands: chat | switch <model_name> | models | datasets | evaluate | help | quit")

        while True:
            try:
                user = input("\nPSYCHAI: What would you like to do? ").strip()
            except KeyboardInterrupt:
                print("\n👋 Bye! See you next time!")
                break

            if user in ("quit", "exit", "q"):
                print("👋 Bye! See you next time!")
                break

            if user == "help":
                print("Commands: chat | switch <model_name> | models | datasets | evaluate | help | quit")
                continue

            if user == "models":
                print(f"Here are the available local models :")
                print("\n".join(models.keys()) or "(none)")
                continue

            if user == "datasets":
                print(f"Here are the available datasets :")
                print("\n".join(datasets.keys()) or "(none)")
                continue

            if user.startswith("switch "):
                reasoning, model_args = _get_model_args()
                model_name = user.split(" ", 1)[1].strip()
                model_path = models.get(model_name, model_name)
                self.load_model_and_tokenizer(model_name,
                                              model_path,
                                              reasoning,
                                              max_seq_length=model_args["max_seq_length"],
                                              load_in_4bit=model_args["load_in_4bit"],
                                              dtype=model_args["dtype"])
                
                print(f"Switched to: {model_name}")
                print(f"This model is reasoning: {reasoning}")
                print(f"This model's max_seq_length is: {model_args['max_seq_length']}")
                print(f"This model is loaded in 4bit: {model_args['load_in_4bit']}")
                print(f"This model's dtype is: {model_args['dtype']}")
                continue

            if user == "chat":
                if self.model_manager.model is None:
                    print("⚠️ Load a model first (use 'switch').")
                    continue
                print(f"Chatting with {self.model_manager.model_name}...")
                print("Type 'exit' to leave chat anytime.")

                generate_args = _get_generate_args(self.model_manager.reasoning)

                system_prompt = input("Please enter the system prompt (optional): ").strip()
                messages = [{"role": "system", "content": system_prompt}] if system_prompt else []

                while True:
                    try:
                        msg = input("\nYou: ").strip()
                        messages.append({"role": "user", "content": msg})
                    except (KeyboardInterrupt, EOFError):
                        print("\n👋 Bye, See you next time!")
                        break
                    if msg.lower() in ("exit", "quit", "q"):
                        break
                    if not msg:
                        continue
                    self.chat(messages, generate_args)
                continue

            if user == "evaluate":
                print("You've entered the evaluate mode.")

                model_name = input("Please enter the model name: ").strip()
                model_path = models.get(model_name, model_name)
                reasoning, model_args = _get_model_args()

                print(f"Here are the available datasets:")
                print("\n".join(datasets.keys()) or "(none)")
                print("if you don't see the data you want, you can enter the data path directly")
                data_name = input("Please enter the data name: ").strip()
                data_path = input("Please enter the data path: ").strip()
                data_type = input("Please enter the data format(chat/instruction/plain): ").strip()
                max_samples = input("Please enter the number of samples you want to evaluate: ").strip()
                max_samples = int(max_samples) if max_samples else None
                batch_size = input("Please enter the batch size of the evaluation: ").strip()
                batch_size = int(batch_size) if batch_size else 1
                result_dir = input("Where do you want to save the results: ").strip()

                if data_type == "plain":
                    while True:
                        output_hidden_states = input("Please enter whether to output hidden states(True/False): ").strip()
                        if output_hidden_states in ("true", "1", "yes"):
                            output_hidden_states = True
                            break
                        elif output_hidden_states in ("false", "0", "no"):
                            output_hidden_states = False
                            break
                        else:
                            print("⚠️ Please enter 'True' or 'False'.")
                    if output_hidden_states:
                        layer = input("Please enter the layers you want to output(separated by comma): ").strip()
                        layer = [int(l) for l in layer.split(",")]
                    else:
                        layer = [-1]
                else:
                    generate_args = _get_generate_args(reasoning)
                    while True:
                        output_scores = input("Please enter whether to output scores(True/False): ").strip()
                        if output_scores in ("true", "1", "yes"):
                            output_scores = True
                            break
                        elif output_scores in ("false", "0", "no"):
                            output_scores = False
                            break
                        else:
                            print("⚠️ Please enter 'True' or 'False'.")

                    while True:
                        output_logits = input("Please enter whether to output logits(True/False): ").strip()
                        if output_logits in ("true", "1", "yes"):
                            output_logits = True
                            break
                        elif output_logits in ("false", "0", "no"):
                            output_logits = False
                            break
                        else:
                            print("⚠️ Please enter 'True' or 'False'.")

                print(f"Will start evaluating {model_name} on {data_name}...")

                self.evaluate_text(model_name, 
                                    model_path,
                                    reasoning, 
                                    data_name, 
                                    data_path, 
                                    data_type,
                                    result_dir=result_dir,
                                    max_samples=max_samples,
                                    batch_size=batch_size,
                                    model_args=model_args,
                                    generate_args=generate_args,
                                    output_scores=output_scores,
                                    output_logits=output_logits,
                                    layer=layer,
                                    output_hidden_states=output_hidden_states)
                continue

            print("Unknown command. Try: chat | switch <model> | models | datasets | evaluate | help | quit")
