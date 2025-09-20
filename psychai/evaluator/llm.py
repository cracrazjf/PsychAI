from __future__ import annotations

import os
import json
import ast
import re
import queue
from tqdm import tqdm
import pandas as pd
import threading
from pathlib import Path
from functools import partial
from ..model_manager.llm import LLM_ModelManager
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import LogitsProcessor, LogitsProcessorList
import torch
from torch.nn.utils.rnn import pad_sequence
try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
except Exception:  # pragma: no cover - allow usage without sklearn
    accuracy_score = None
    classification_report = None
    confusion_matrix = None
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
    
    class CaptureLogits(LogitsProcessor):
            def __init__(self, out_queue: queue.Queue, top_k: int = 50):
                self.out_queue = out_queue
                self.top_k = top_k
                self.step = 0

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
                topk_vals, topk_ids = torch.topk(scores, k=min(self.top_k, scores.size(-1)), dim=-1)
                payload = {
                    "step": self.step,
                    "topk_ids": topk_ids[0].detach().cpu().tolist(),
                    "topk_vals": topk_vals[0].detach().cpu().tolist(),
                }
                self.out_queue.put(payload)
                self.step += 1
                return scores

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
                         prompt_template: Optional[str] = None,
                         labels_list: List[str] = None) -> str:

        reasoning_effort = generate_args.get("reasoning_effort", None)
        
        if data_type == "chat":
            data = data.map(partial(self.format_chat, reasoning_effort=reasoning_effort), batched=True)
        elif data_type == "instruction":
            data = data.map(partial(self.format_instruction, prompt_template=prompt_template), batched=True)
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        # data.set_format(type=None)

        def collate_for_generate(batch):
            prompts = [ex["text"] for ex in batch]
            enc = self.model_manager.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            enc["labels"] = [ex["label"] for ex in batch]
            return enc

        loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_for_generate)

        pred_texts = []
        gold_texts = []
        think_texts = []
        tqdm_loader = tqdm(loader, desc="Evaluating")
        for batch in tqdm_loader:
            labels = batch.pop("labels")    
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model_manager.model.generate(**batch, 
                                                        max_new_tokens = generate_args["max_new_tokens"], 
                                                        temperature = generate_args["temperature"],
                                                        do_sample = generate_args["do_sample"],
                                                        top_p = generate_args["top_p"],
                                                        top_k = generate_args["top_k"],
                                                        use_cache = True)
            gold_texts.extend(labels)
            input_len = batch["input_ids"].size(1)
            if data_type == "instruction":
                decoded_outputs = self.model_manager.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                predictions = []
                for decoded_output in decoded_outputs:
                    head, sep, tail = decoded_output.rpartition("### Response:")
                    pred = (tail if sep else decoded_output).strip()
                    predictions.append(pred)
                pred_texts.extend(predictions)
            elif data_type == "chat":
                sliced_outputs = []
                for i in range(outputs.size(0)):
                    sliced_output = outputs[i, input_len:]
                    sliced_outputs.append(sliced_output)
                    print(f"full outputs: {self.model_manager.tokenizer.decode(outputs[i], skip_special_tokens=False)}") 
                    # print(f"sliced_output: {self.model_manager.tokenizer.decode(sliced_output, skip_special_tokens=False)}")
                
                if self.model_manager.reasoning:
                    pred_text = self.model_manager.tokenizer.batch_decode(
                        # pad_sequence(sliced_outputs, batch_first=True, padding_value=self.model_manager.tokenizer.pad_token_id),
                        sliced_outputs,
                        skip_special_tokens=False
                    )
                    analysis_re, final_re = self.get_analysis_and_final_re()
                    pred_text_batch = []
                    think_text_batch = []
                    for text in pred_text:
                        ANALYSIS_RE = re.compile(analysis_re, re.DOTALL | re.IGNORECASE)
                        FINAL_RE = re.compile(final_re, re.DOTALL | re.IGNORECASE)
                        analysis_match = ANALYSIS_RE.search(text)
                        final_match = FINAL_RE.search(text)
                        if analysis_match:
                            think_text_batch.append(analysis_match.group(1).strip())
                            if final_match:
                                pred_text_batch.append(final_match.group(1).strip())
                            else:
                                pred_text_batch.append(text)
                        else:
                            think_text_batch.append("")
                            pred_text_batch.append(text)

                    think_texts.extend(think_text_batch)
                    pred_texts.extend(pred_text_batch)
                else:
                    pred_text = self.model_manager.tokenizer.batch_decode(
                        # pad_sequence(sliced_outputs, batch_first=True, padding_value=self.model_manager.tokenizer.pad_token_id),
                        sliced_outputs,
                        skip_special_tokens=True
                    )
                    pred_texts.extend(pred_text)

        for i, text in enumerate(pred_texts[:10]):
                print("="*40, f"Example {i}", "="*40)
                print("PRED:", text)
                if len(think_texts) > 0:
                    print("THINK:", think_texts[i])
                print("GOLD:", gold_texts[i])
                print()

        def extract_label(text: str, labels_list: Optional[List[str]]) -> str:
                txt = text.lower().strip()
                if labels_list:
                    for lbl in labels_list:
                        if lbl.lower() in txt:
                            return lbl
                return txt.split()[0] if txt else "unknown"

        pred_labels = [extract_label(p, labels_list) for p in pred_texts]
        true_labels = [extract_label(t, labels_list) for t in gold_texts]
        
        if accuracy_score is not None and classification_report is not None and confusion_matrix is not None:
            acc = float(accuracy_score(true_labels, pred_labels))
            uniq = sorted(list(set(true_labels + pred_labels)))
            report = classification_report(true_labels, pred_labels, labels=uniq, output_dict=True, zero_division=0)
            cm = confusion_matrix(true_labels, pred_labels, labels=uniq)
        
            print("="*50)
            print(f"Accuracy: {acc:.2%}")
            print("="*50)

            print("\nClassification report:")
            print(pd.DataFrame(report).T.round(3))   # precision, recall, f1-score

            print("\nConfusion Matrix:")
            print(pd.DataFrame(cm, index=uniq, columns=uniq))
        
        return {
            "predictions": pred_texts,
            "ground_truths": gold_texts,
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

    def benchmark_text(
        self,
        model_name: str,
        model_path: str,
        reasoning: bool,
        data_map: Dict[str, str],
        data_type: str,
        labels_map: Dict[str, Any],
        batch_size: int,
        max_samples: Optional[int] = None,
        model_args: Optional[Dict[str, Any]] = None,
        generate_args: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        
        results= {}

        self.load_model_and_tokenizer(model_name, model_path, reasoning, model_args["max_seq_length"], model_args["load_in_4bit"], model_args["dtype"])

        for data_name, data_path in data_map.items():
            data = load_dataset("json", data_files=data_path, split="train")
            results[data_name] = {}
            if max_samples:
                data = data.select(range(max_samples))
            res = self.evaluate_outputs(data,
                                        data_type,
                                        batch_size=batch_size,
                                        generate_args=generate_args,
                                        labels_list=labels_map[data_name])
            results[data_name] = res

        
        print(f"\n{'='*20} BENCHMARK SUMMARY {'='*20}")
        print(f"Model: {model_name}")
        print("-" * 40)
        for data_name in data_map.keys():
            print(f"\n📊 Dataset: {data_name}")
            print("-" * 40)
            
            accuracy = results[data_name].get("accuracy", None)
            if accuracy is not None:
                print(f"  {data_name:20s}: {accuracy:.2%}")
            else:
                print(f"  {data_name:20s}: Failed")

        if output_path:
            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"benchmark_{model_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model": model_name,
                    "datasets": list(data_map.keys()),
                    "results": results,
                }, f, indent=2)
            print(f"💾 Benchmark summary saved to {out_path}")

        return results

    def compare_text(
        self,
        dataset_name: str,
        data_type: str,
        model_names: Union[List[str], str],
        reasoning_map: Dict[str, bool],
        labels: List[str],
        max_samples: Optional[int] = None,
        model_args_map: Optional[Dict] = None,
        generate_args: Optional[Dict[str, Any]] = None,
        save_summary: Optional[bool] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Optional[float]]:

        selected_models = self.select_models(model_names)
        selected_dataset = self.select_datasets(dataset_name)

        test_data = load_dataset("json", data_files=selected_dataset[dataset_name], split="train")
        generate_args = generate_args or self.config.GENERATE_ARGS

        results = {}
        if max_samples:
            test_data = test_data.select(range(max_samples))
        for model_name, model_path in selected_models.items():
            model_args = model_args_map.get(model_name, {})
            max_seq_length = model_args.get("max_seq_length", self.config.MAX_SEQ_LENGTH)
            load_in_4bit = model_args.get("load_in_4bit", self.config.LOAD_IN_4BIT)
            dtype = model_args.get("dtype", self.config.DTYPE)
            self.load_model_and_tokenizer(model_name, model_path, reasoning_map[model_name], max_seq_length, load_in_4bit, dtype)
            res = self.evaluate_outputs(data_type, test_data, 
                                        labels_list=labels, 
                                        batch_size=batch_size,
                                        generate_args=generate_args)
            results[model_name] = res

        print(f"\n{'='*20} COMPARISON SUMMARY {'='*20}")
        print(f"Dataset: {dataset_name}")
        print("-" * 40)

        sorted_results = sorted(results.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True)
        for model_name, result in sorted_results:
            print(f"\n📊 Model: {model_name}")
            print("-" * 40)
            accuracy = result.get("accuracy", None)
            if accuracy is not None:
                print(f"  {model_name:20s}: {accuracy:.2%}")
            else:
                print(f"  {model_name:20s}: Failed")

        if save_summary:
            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"comparison_{dataset_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "dataset": dataset_name,
                    "models": selected_models.keys(),
                    "results": results,
                }, f, indent=2)
            print(f"💾 Comparison saved to {out_path}")

        return results

    def interactive_text(self) -> None:
        """Simple REPL for chatting, switching models, benchmark and compare."""
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
                labels_map = {}
                for data_name in data_names:
                    while True:
                        labels = input(f"Please enter the true labels for {data_name} e.g.[label1,label2]: ").strip()
                        if labels:
                            labels = ast.literal_eval(labels)
                            labels_map[data_name] = labels
                            break
                        else:
                            print("You must enter labels for each data")

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
                                    labels_map, 
                                    max_samples=max_samples,
                                    batch_size=batch_size,
                                    model_args=model_args,
                                    generate_args=generate_args)
                continue

            if user == "compare":
                print("🎮 The compare command is used to compare the performance of multiple models on a single dataset.")
                model_names = input("Please enter the models (separated by comma or 'all'): ").strip()
                if model_names == "all":
                    model_names = list(models.keys())
                else:
                    model_names = [m.strip() for m in model_names.split(",")]
                reasoning_map = {}
                model_args_map = {}
                for model_name in model_names:
                    reasoning_map[model_name] = input(f"Is the model {model_name} reasoning? (y/n): ").strip().lower() == "y"
                    max_seq_length = input(f"Enter max_seq_length for {model_name}(empty for default): ").strip() or self.config.MAX_SEQ_LENGTH
                    load_in_4bit = input(f"Enter load_in_4bit for {model_name}(empty for default): ").strip() or self.config.LOAD_IN_4BIT
                    dtype = input(f"Enter dtype for {model_name}(empty for default): ").strip() or self.config.DTYPE
                    model_args_map[model_name] = {  
                        "max_seq_length": max_seq_length,
                        "load_in_4bit": load_in_4bit,
                        "dtype": dtype,
                    }
                dataset_name = input("Please enter the dataset name: ").strip()
                data_type = input("Please enter the data type(chat or instruction): ").strip()
                max_samples = input("Please enter the number of samples you want to evaluate: ").strip()
                max_samples = int(max_samples) if max_samples else None

                while True:
                    labels = input(f"Please enter the labels for the dataset {dataset_name} e.g. [label1,label2]: ").strip()
                    if labels:
                        labels = ast.literal_eval(labels)
                        break
                    else:
                        print("You must enter labels for the dataset")

                generate_args = _get_generate_args()
                print(f"😎 Comparing the models {model_names} on the dataset {dataset_name}...")
                self.compare_text(dataset_name, data_type, model_names, reasoning_map, labels, max_samples=max_samples, model_args_map=model_args_map, generate_args=generate_args)
                continue

            print("Unknown command. Try: chat | switch <model> | models | datasets | benchmark | compare | help | quit")
