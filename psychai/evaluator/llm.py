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

    def evaluate_outputs(self,
                         data: Any,  
                         data_type: str,
                         batch_size: int,
                         generate_args: Dict[str, Any],
                         result_dir: str,
                         prompt_template: Optional[str] = None,
                         output_scores: bool = False,
                         output_logits: bool = False
                         ):

        reasoning_effort = generate_args.get("reasoning_effort", None)

        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"results.jsonl"
        open(result_path, "w").close()
        
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
        shard_id = 0
        max_valid_length = 0
        buffer = []

        tqdm_loader = tqdm(loader, desc="Evaluating")
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
                                                        return_dict_in_generate=True,
                                                        output_scores=output_scores,
                                                        output_logits=output_logits,
                                                        output_hidden_states=output_hidden_states,
                                                        # output_attentions=output_attentions,
                                                        )
            print(f"Outputs: {outputs.keys()}")
            
            # get corresponding scores and tokens
            sequences = outputs.sequences
            # print(f"Sequences: {sequences}")
            if output_scores:
                scores = torch.stack(outputs.scores, dim=1)
                topk_scores, topk_scores_ids = torch.topk(scores, k=generate_args["top_k"], dim=-1)
            if output_logits:
                logits = torch.stack(outputs.logits, dim=1)
                topk_logits, topk_logits_ids = torch.topk(logits, k=generate_args["top_k"], dim=-1)

            topk_logits, topk_logits_ids = torch.topk(logits, k=generate_args["top_k"], dim=-1)

        #     # decide the input length
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
                    max_valid_length = max(max_valid_length, valid_length)

                    result = {
                        "sample_id": sample_id,
                        "prompt": self.model_manager.tokenizer.decode(batch["input_ids"][mini_batch_idx], skip_special_tokens=True),
                        "prediction": self.model_manager.tokenizer.decode(valid_new_tokens, skip_special_tokens=False),
                        "label": labels[mini_batch_idx],
                        "token_ids": valid_new_tokens.tolist()
                        }

                    temp_buffer = {
                        "sample_id": sample_id,
                        "valid_length": int(valid_length),
                    }

                    if output_scores:
                        valid_scores = scores[mini_batch_idx, :valid_length, :]
                        result["chosen_scores"] = valid_scores.gather(1, valid_new_tokens.view(-1, 1)).squeeze(1).tolist()
                        result["topk_scores_ids"] = topk_scores_ids[mini_batch_idx, :valid_length, :].tolist()
                        result["topk_scores"] = topk_scores[mini_batch_idx, :valid_length, :].tolist()

                    if output_logits:
                        valid_logits = logits[mini_batch_idx, :valid_length, :]
                        result["topk_logits_ids"] = topk_logits_ids[mini_batch_idx, :valid_length, :].tolist()
                        result["topk_logits"] = topk_logits[mini_batch_idx, :valid_length, :].tolist()

                        temp_buffer["logits"] = valid_logits.detach().cpu().to(torch.float16)

                    buffer.append(temp_buffer)

                    with open(result_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

                    def _flush_buffer(buffer, content_key):
                        _,vocab_size = buffer[0][content_key].shape
                        shard = torch.full((len(buffer), max_valid_length, vocab_size), -float("inf"), dtype=torch.float16)
                        for i, row in enumerate(buffer):
                            shard[i, :row["valid_length"], :] = row[content_key]
                        shard_path = result_dir / "{}_shard_{:05d}.fp16.npz".format(content_key, shard_id)
                        np.savez_compressed(shard_path, shard.numpy())

                        manifest_path = result_dir / f"{content_key}_manifest.jsonl"
                        with open(manifest_path, "w", encoding="utf-8") as f:
                            for i, row in enumerate(buffer):
                                f.write(json.dumps({
                                    "sample_id": row["sample_id"],
                                    "shard_path": str(shard_path),
                                    "valid_length": row["valid_length"],
                                    "row_idx": i,
                                }) + "\n")

                    if len(buffer) >= 100:
                        if output_logits:
                            _flush_buffer(buffer, "logits")
                        shard_id += 1
                        max_valid_length = 0
                        buffer = []

                    sample_id += 1


        #         # if self.model_manager.reasoning:
        #         #     pred_text = self.model_manager.tokenizer.batch_decode(
        #         #         # pad_sequence(sliced_outputs, batch_first=True, padding_value=self.model_manager.tokenizer.pad_token_id),
        #         #         sliced_output_seqs,
        #         #         skip_special_tokens=False
        #         #     )
        #         #     analysis_re, final_re = self.get_analysis_and_final_re()
        #         #     pred_text_batch = []
        #         #     think_text_batch = []
        #         #     for text in pred_text:
        #         #         ANALYSIS_RE = re.compile(analysis_re, re.DOTALL | re.IGNORECASE)
        #         #         FINAL_RE = re.compile(final_re, re.DOTALL | re.IGNORECASE)
        #         #         analysis_match = ANALYSIS_RE.search(text)
        #         #         final_match = FINAL_RE.search(text)
        #         #         if analysis_match:
        #         #             think_text_batch.append(analysis_match.group(1).strip())
        #         #             if final_match:
        #         #                 pred_text_batch.append(final_match.group(1).strip())
        #         #             else:
        #         #                 pred_text_batch.append(text)
        #         #         else:
        #         #             think_text_batch.append("")
        #         #             pred_text_batch.append(text)

        #         #     think_texts.extend(think_text_batch)
        #         #     pred_texts.extend(pred_text_batch)
        #         # else:
        #         #     pred_text = self.model_manager.tokenizer.batch_decode(
        #         #         new_tokens,
        #         #         skip_special_tokens=True
        #         #     )
        #         #     pred_texts.extend(pred_text)

        _flush_buffer(buffer, "logits")

    def evaluate_hidden_states(self,
                              data: Any,
                              batch_size: int,
                              layer: list[int],
                              result_dir: str,
                              output_hidden_states: bool = False,
                              output_attentions: bool = False) -> Dict[str, Dict[str, Optional[float]]]:

        def collate_for_activation(batch):
            prompts = [ex["text"] for ex in batch]
            enc = self.model_manager.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            return enc
        
        loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_for_activation, pin_memory=True)

        tqdm_loader = tqdm(loader, desc="Evaluating Activations")
        for batch in tqdm_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model_manager.model(**batch,
                                                return_dict=True,
                                                output_hidden_states=output_hidden_states,
                                                output_attentions=output_attentions,
                                                )
            print(f"Outputs: {outputs.keys()}")
            print(f"Input: {batch['input_ids'].shape}")
            print(f"Mask: {batch['attention_mask'].shape}")
            print(f"Hidden States: {outputs.hidden_states.shape}")
            print(f"Last Hidden States: {outputs.hidden_states[-1].shape}")
            
        

    def benchmark_text(
        self,
        model_name: str,
        model_path: str,
        reasoning: bool,
        data_map: Dict[str, str],
        data_type: str,
        batch_size: int,
        result_dir: str,
        output_scores: bool = False,
        output_logits: bool = False,
        layer: list[int] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
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
            result_dir = f"{result_dir}/{model_name}_{data_name}"
            if max_samples:
                data = data.select(range(max_samples))
            if output_hidden_states or output_attentions:
                self.evaluate_hidden_states(data,
                                            batch_size=batch_size,
                                            layer=layer,
                                            result_dir=result_dir,
                                            output_hidden_states=output_hidden_states,
                                            output_attentions=output_attentions)
            else:
                self.evaluate_outputs(data,
                                    data_type,
                                    batch_size=batch_size,
                                    generate_args=generate_args,
                                    result_dir=result_dir,
                                    output_scores=output_scores,
                                    output_logits=output_logits,
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
        print("Commands: chat | switch <model_name> | models | datasets | benchmark | compare | help | quit")

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
                print("Commands: chat | switch <model_name> | models | datasets | benchmark | compare | help | quit")
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

            if user == "benchmark":
                print("You've entered the benchmark mode, you can run a single model on multiple datasets.")

                model_name = input("Please enter the model name: ").strip()
                model_path = models.get(model_name, model_name)
                reasoning, model_args = _get_model_args()

                print(f"Here are the available datasets:")
                print("\n".join(datasets.keys()) or "(none)")
                data_names = input("Please enter the data names (separated by comma or 'all'): ").strip()
                if data_names == "all":
                    data_names = list(datasets.keys())
                else:
                    data_names = [d.strip() for d in data_names.split(",")]
                selected_data = {data_name: datasets[data_name] for data_name in data_names}
                data_type = input("Please enter the data format(chat or instruction): ").strip()
                result_dir = input("Where do you want to save the results: ").strip()
                # labels_map = {}
                # for data_name in data_names:
                #     while True:
                #         labels = input(f"Please enter the true labels for {data_name} e.g.[label1,label2]: ").strip()
                #         if labels:
                #             labels = ast.literal_eval(labels)
                #             labels_map[data_name] = labels
                #             break
                #         else:
                #             print("You must enter labels for each data")

                max_samples = input("Please enter the number of samples you want to evaluate: ").strip()
                max_samples = int(max_samples) if max_samples else None
                batch_size = input("Please enter the batch size of the evaluation: ").strip()
                batch_size = int(batch_size) if batch_size else 1

                generate_args = _get_generate_args(reasoning)
                print(f"Will start evaluating {model_name} on {data_names}...")

                self.benchmark_text(model_name, 
                                    model_path,
                                    reasoning, 
                                    selected_data, 
                                    data_type,
                                    result_dir=result_dir,
                                    # labels_map, 
                                    max_samples=max_samples,
                                    batch_size=batch_size,
                                    model_args=model_args,
                                    generate_args=generate_args)
                continue

            print("Unknown command. Try: chat | switch <model> | models | datasets | benchmark | compare | help | quit")
