"""
Text-focused evaluation utilities

Simple, minimal evaluator for text models with:
- Interactive chat
- Benchmark across multiple models and datasets
- Model comparison on a dataset

Audio/vision evaluators can follow the same interface later.
"""

from __future__ import annotations

import os
import json
import ast
import gc
import tqdm
import traceback
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
except Exception:  # pragma: no cover - allow usage without sklearn
    accuracy_score = None
    classification_report = None
    confusion_matrix = None

from ..models import ModelManager
from transformers import TextIteratorStreamer
from ..config import EvaluationConfig
from ..data import validate_format, load_jsonl, find_file, load_json

class Evaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.models_root = config.MODELS_ROOT
        self.model_cache_root = config.MODEL_CACHE_ROOT
        self.data_root = config.DATA_ROOT
        self.use_unsloth = config.USE_UNSLOTH
        self.model_manager = ModelManager()

    # list available models in models_root and model_cache_root
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

    def select_models(self, model_names: Union[List[str], str]) -> List[str]:
        models_dict = self.list_available_models()
        selected_models = {}

        if model_names == "all":
            model_list = list(models_dict.keys())
            selected_models = models_dict
        else:
            model_list = model_names if isinstance(model_names, list) else [model_names]
            selected_models = {model_name: models_dict[model_name] for model_name in model_list}

        return selected_models

    def load_model_and_tokenizer(self, model_name: str, model_path: str, max_seq_length: int, load_in_4bit: bool, dtype: str) -> Tuple[Any, Any]:
        self.model_manager.load_model(model_name, model_path, self.use_unsloth, for_training=False, max_seq_length=max_seq_length, load_in_4bit=load_in_4bit, full_finetuning=False, dtype=dtype)
        return self.model_manager.model, self.model_manager.tokenizer

    # list available datasets in data_root
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
                            meta = find_file(root, "meta.json")
                            if meta:
                                meta = load_json(meta)
                                data_type = meta["type"]
                            else:
                                data_type = "chat"
                            processed_path = Path(root) / dir
                            test_json: Optional[str] = None
                            for j in processed_path.glob("test*.json*"):
                                test_json = str(j)
                                break
                            out[item.name+"_"+data_type] = test_json
        return dict(sorted(out.items()))

    def select_datasets(self, dataset_names: Union[List[str], str]) -> List[str]:
        datasets_dict = self.list_available_datasets()
        selected_datasets = {}

        if dataset_names == "all":
            dataset_list = list(datasets_dict.keys())
            selected_datasets = datasets_dict
        else:
            dataset_list = dataset_names if isinstance(dataset_names, list) else [dataset_names]
            selected_datasets = {dataset_name: datasets_dict[dataset_name] for dataset_name in dataset_list}

        return selected_datasets

    def load_test_data(self, dataset_name: str, data_path: str) -> List[Any]:
        data_type = dataset_name.split("_")[-1]
        data = list(load_jsonl(data_path))
        if data_type == "chat":
            if validate_format(data, "chat"):
                return data, data_type
        elif data_type == "instruction":
            if validate_format(data, "instruction"):
                return data, data_type
        else:
            raise ValueError(f"Invalid data type: {data_type}")

    def format_chat(self, messages: Any, reasoning_effort: Optional[str] = None) -> str:
        device = next(self.model_manager.model.parameters()).device
        if reasoning_effort is None:
            reasoning_effort = 'low'
        return self.model_manager.tokenizer.apply_chat_template(messages, 
                                                            add_generation_prompt=True, 
                                                            return_tensors = "pt",
                                                            return_dict = True,
                                                            reasoning_effort=reasoning_effort).to(device)

    def format_instruction(self, messages: Any, prompt_template: str) -> str:
        device = next(self.model_manager.model.parameters()).device
        formatted_messages = []
        for message in messages:
            formatted_messages.append(prompt_template.format(message["instruction"], message["input"], message["output"]))
        return self.model_manager.tokenizer(formatted_messages,
                                    return_tensors = "pt",
                                    return_dict = True).to(device)

    def chat(self, 
                messages: List[Dict[str, str]], 
                generate_args: Optional[Dict[str, Any]] = None) -> str:
        
        max_new_tokens = generate_args.get("max_new_tokens", 128)
        temperature = generate_args.get("temperature", 0.7)
        do_sample = generate_args.get("do_sample", True)
        top_p = generate_args.get("top_p", 0.95)
        top_k = generate_args.get("top_k", 50)
        reasoning_effort = generate_args.get("reasoning_effort", None)
        
        formatted_inputs = self.format_chat(messages, reasoning_effort)
        
        streamer = TextIteratorStreamer(self.model_manager.tokenizer, skip_prompt=True, skip_special_tokens=True)

        def generate_response():
            with torch.no_grad():
                self.model_manager.model.generate(
                    **formatted_inputs,
                    streamer=streamer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                )
        
        thread = threading.Thread(target=generate_response)
        thread.start()
        print("Model: ", end="", flush=True)
        for new_text in streamer:
            if new_text.strip().startswith("analysis"):
                print("Thinking: ", end="", flush=True)
            print(new_text, end="", flush=True)
        thread.join()
        print()

    def evaluate_outputs(self, 
                    data_type: str,
                    messages: Any, 
                    labels: List[str],
                    generate_args: Optional[Dict[str, Any]] = None) -> str:
        
        max_new_tokens = generate_args.get("max_new_tokens", 128)
        temperature = generate_args.get("temperature", 0.7)
        do_sample = generate_args.get("do_sample", True)
        top_p = generate_args.get("top_p", 0.95)
        top_k = generate_args.get("top_k", 50)
        reasoning_effort = generate_args.get("reasoning_effort", None)
        prompt_template = self.config.PROMPT_TEMPLATE
        
        if data_type == "chat":
            formatted_inputs = self.format_chat(messages, reasoning_effort)
            answers = []
            for message in messages:
                for dict in message:
                    if dict["role"] == "assistant":
                        answers.append(dict["content"])
        elif data_type == "instruction":
            formatted_inputs = self.format_instruction(messages, prompt_template)
            answers = []
            for message in messages:
                answers.append(message["output"])
        else:
            raise ValueError(f"Invalid data type: {data_type}")

        outputs = self.model_manager.model.generate(**formatted_inputs, 
                                                max_new_tokens = max_new_tokens, 
                                                use_cache = True,
                                                temperature = temperature,
                                                do_sample = do_sample,
                                                top_p = top_p,
                                                top_k = top_k)
        
        
        predictions = []
        if data_type == "instruction":
            decoded_outputs = self.model_manager.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for decoded_output in decoded_outputs:
                head, sep, tail = decoded_output.rpartition("### Response:")
                pred = (tail if sep else decoded_output).strip()
                predictions.append(pred)
        elif data_type == "chat":
            sliced_outputs = []
            for i in range(outputs.size(0)):
                input_len = (formatted_inputs["input_ids"][i] != self.model_manager.tokenizer.pad_token_id).sum().item()
                sliced_output = outputs[i, input_len:]
                sliced_outputs.append(sliced_output)
            decoded_outputs = self.model_manager.tokenizer.batch_decode(sliced_outputs, skip_special_tokens=True)
            predictions = decoded_outputs

        def extract_label(text: str, labels_list: Optional[List[str]]) -> str:
                txt = text.lower().strip()
                if labels_list:
                    for lbl in labels_list:
                        if lbl.lower() in txt:
                            return lbl
                return txt.split()[0] if txt else "unknown"

        pred_labels = [extract_label(p, labels) for p in predictions]
        true_labels = [extract_label(t, labels) for t in answers]
        
        if accuracy_score is not None and classification_report is not None and confusion_matrix is not None:
            acc = float(accuracy_score(true_labels, pred_labels))
            uniq = sorted(list(set(true_labels + pred_labels)))
            report = classification_report(true_labels, pred_labels, labels=uniq, output_dict=True, zero_division=0)
            cm = confusion_matrix(true_labels, pred_labels, labels=uniq)
        
        return {
            "predictions": predictions,
            "ground_truths": answers,
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,
        }

    def print_results(results: Dict[str, Any]) -> None:
        print("\n📊 Evaluation Results")
        print("=" * 30)
        for key in ["total_samples", "processed_samples", "accuracy", "exact_match_accuracy", "avg_prediction_length", "avg_truth_length"]:
            if key in results:
                val = results[key]
                if isinstance(val, float):
                    print(f"{key}: {val:.4f}")
                else:
                    print(f"{key}: {val}")

        # Show a few examples
        preds = results.get("predictions", [])
        truths = results.get("ground_truths", [])
        if preds and truths:
            print("\n📝 Examples:")
            for i in range(min(3, len(preds))):
                print(f"- Pred: {preds[i][:512]}{'...' if len(preds[i]) > 120 else ''}")
                print(f"  True: {truths[i][:512]}{'...' if len(truths[i]) > 120 else ''}")

    def benchmark_text(
        self,
        model_name: str,
        dataset_names: Union[List[str], str],
        labels_map: Dict[str, Any],
        max_samples: Optional[int] = None,
        max_seq_length: Optional[int] = None,
        load_in_4bit: Optional[bool] = None,
        dtype: Optional[str] = None,
        generate_args: Optional[Dict[str, Any]] = None,
        save_summary: Optional[bool] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        
        selected_model = self.select_models(model_name)
        selected_datasets = self.select_datasets(dataset_names)
        model_path = selected_model.get(model_name, model_name)

        results= {}

        max_seq_length = max_seq_length or self.config.MAX_SEQ_LENGTH
        load_in_4bit = load_in_4bit or self.config.LOAD_IN_4BIT
        dtype = dtype or self.config.DTYPE
        generate_args = generate_args or self.config.GENERATE_ARGS

        self.load_model_and_tokenizer(model_name, model_path, max_seq_length, load_in_4bit, dtype)

        for dataset_name, data_path in selected_datasets.items():
            test_data, data_type = self.load_test_data(dataset_name, data_path)
            results[dataset_name] = {}
            if max_samples:
                test_data = test_data[:max_samples]
            res = self.evaluate_outputs(data_type, test_data, 
                                        labels=labels_map.get(dataset_name), 
                                        generate_args=generate_args)
            results[dataset_name] = res

        if save_summary:
            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"benchmark_{model_name}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model": model_name,
                    "datasets": selected_datasets.keys(),
                    "results": results,
                }, f, indent=2)
            print(f"💾 Benchmark summary saved to {out_path}")

        return results

    def compare_text(
        self,
        dataset_name: str,
        model_names: Union[List[str], str],
        labels: List[str],
        max_samples: Optional[int] = None,
        max_seq_length: Optional[int] = None,
        load_in_4bit: Optional[bool] = None,
        dtype: Optional[str] = None,
        generate_args: Optional[Dict[str, Any]] = None,
        save_summary: Optional[bool] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Optional[float]]:

        selected_models = self.select_models(model_names)
        selected_dataset = self.select_datasets(dataset_name)

        test_data, data_type = self.load_test_data(dataset_name, selected_dataset[dataset_name])
        max_seq_length = max_seq_length or self.config.MAX_SEQ_LENGTH
        load_in_4bit = load_in_4bit or self.config.LOAD_IN_4BIT
        dtype = dtype or self.config.DTYPE
        generate_args = generate_args or self.config.GENERATE_ARGS

        results = {}
        if max_samples:
            test_data = test_data[:max_samples]
            
        for model_name, model_path in selected_models.items():
            self.load_model_and_tokenizer(model_name, model_path, max_seq_length, load_in_4bit, dtype)
            res = self.evaluate_outputs(data_type, test_data, 
                                        labels=labels, 
                                        generate_args=generate_args)
            results[model_name] = res

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

        def _get_generate_args():
            default_generate_args = self.config.GENERATE_ARGS
            generate_args = {}
            generate_args["max_new_tokens"] = input("Enter max_new_tokens(empty for default): ").strip() or default_generate_args["max_new_tokens"]
            generate_args["temperature"] = input("Enter temperature(empty for default): ").strip() or default_generate_args["temperature"]
            generate_args["do_sample"] = input("Enter do_sample(empty for default): ").strip() or default_generate_args["do_sample"]
            generate_args["top_p"] = input("Enter top_p(empty for default): ").strip() or default_generate_args["top_p"]
            generate_args["top_k"] = input("Enter top_k(empty for default): ").strip() or default_generate_args["top_k"]
            generate_args["reasoning_effort"] = input("Enter reasoning_effort(empty for default): ").strip() or default_generate_args["reasoning_effort"]
            return generate_args

        if not models:
            print(f"⚠️ No local models found under {self.models_root} or {self.model_cache_root}. You can still load HF refs by name via 'switch <ref>'.")

        print("\n🎮 Welcome to PSYCHAI Interactive Text Evaluation")
        print("You can load/switch between models by using 'switch <model_name>'.")
        print("Commands: chat | switch <model_name> | models | datasets | benchmark | compare | quit")

        while True:
            try:
                user = input("\n> ").strip()
            except KeyboardInterrupt:
                print("\n👋 Bye")
                break

            if user in ("quit", "exit", "q"):
                print("👋 Bye")
                break

            if user == "models":
                print("Models:", ", ".join(models.keys()) or "(none)")
                continue

            if user == "datasets":
                print("Datasets:", ", ".join(datasets.keys()) or "(none)")
                continue

            if user.startswith("switch "):
                max_seq_length = input("Enter max_seq_length(empty for default): ").strip() or self.config.MAX_SEQ_LENGTH
                load_in_4bit = input("Enter load_in_4bit(empty for default): ").strip() or self.config.LOAD_IN_4BIT
                dtype = input("Enter dtype(empty for default): ").strip() or self.config.DTYPE
                model_name = user.split(" ", 1)[1].strip()
                model_path = models.get(model_name, model_name)
                self.load_model_and_tokenizer(model_name, model_path, max_seq_length, load_in_4bit, dtype)
                print(f"🔄 Switched to: {model_name}")
                continue

            if user == "chat":
                if self.model_manager.model is None:
                    print("⚠️ Load a model first (use 'switch').")
                    continue
                print(f"Chatting with {self.model_manager.model_name}")
                print("Type 'exit' to leave chat.")
                generate_args = _get_generate_args()
                
                system_prompt = input("System prompt(optional): ").strip()
                if system_prompt:
                    messages = [{"role": "system", "content": system_prompt}]
                else:
                    messages = []
                while True:
                    try:
                        msg = input("You: ").strip()
                        messages.append({"role": "user", "content": msg})
                    except KeyboardInterrupt:
                        print("\n👋 Bye")
                        break
                    if msg.lower() in ("exit", "quit", "q"):
                        break
                    if not msg:
                        continue
                    self.chat(messages, generate_args)
                continue

            if user == "benchmark":
                model_name = input("Model: ").strip()
                dataset_names = input("Datasets (comma or 'all'): ").strip()
                if dataset_names == "all":
                    dataset_names = datasets.keys()
                else:
                    dataset_names = [d.strip() for d in dataset_names.split(",")]
                labels_map = {}
                for dataset_name in dataset_names:
                    while True:
                        labels = input(f"Enter labels for the dataset {dataset_name} e.g.[label1,label2]: ").strip()
                        if labels:
                            labels = ast.literal_eval(labels)
                            labels_map[dataset_name] = labels
                            break
                        else:
                            print("You must enter labels for each dataset")

                max_samples = input("Num samples: ").strip()
                max_samples = int(max_samples) if max_samples else None

                generate_args = _get_generate_args()

                self.benchmark_text(model_name, dataset_names, labels_map, max_samples=max_samples, generate_args=generate_args)
                continue

            if user == "compare":
                model_names = input("Models (comma or 'all'): ").strip()
                if model_names == "all":
                    model_names = models.keys()
                else:
                    model_names = [m.strip() for m in model_names.split(",")]
                dataset_name = input("Dataset: ").strip()
                max_samples = input("Num samples: ").strip()
                max_samples = int(max_samples) if max_samples else None

                while True:
                    labels = input(f"Enter labels for the dataset {dataset_name} e.g.[label1,label2]: ").strip()
                    if labels:
                        labels = ast.literal_eval(labels)
                        break
                    else:
                        print("You must enter labels for the dataset")

                generate_args = _get_generate_args()
                self.compare_text(dataset_name, model_names, labels, max_samples=max_samples, generate_args=generate_args)
                continue

            print("Unknown command. Try: chat | switch <model> | models | datasets | benchmark | compare | quit")
