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

from transformers import TextStreamer
from ..models import ModelManager
from ..data import validate_format

def format_chat(model_manager: ModelManager, messages: Any, reasoning_effort: Optional[str] = None) -> str:
    device = next(model_manager.model.parameters()).device
    return model_manager.tokenizer.apply_chat_template(messages, 
                                                        add_generation_prompt=True, 
                                                        return_tensors = "pt",
                                                        return_dict = True,
                                                        reasoning_effort=reasoning_effort).to(device)

def format_instruction(model_manager: ModelManager, messages: Any, prompt_template: str) -> str:
    device = next(model_manager.model.parameters()).device
    formatted_messages = []
    for message in messages:
        formatted_messages.append(prompt_template.format(message["instruction"], message["input"], message["output"]))
    return model_manager.tokenizer(formatted_messages,
                                   return_tensors = "pt",
                                   return_dict = True).to(device)

def chat(model_manager: ModelManager, 
             messages: List[Dict[str, str]], 
             *,
             max_new_tokens: int = 128, 
             temperature: float = 0.7, 
             do_sample: bool = True, 
             top_p: float = 0.95, 
             top_k: int = 50,
             reasoning_effort: Optional[str] = None) -> str:
    
    formatted_inputs = format_chat(model_manager, messages, reasoning_effort)
    
    streamer = TextStreamer(model_manager.tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        outputs = model_manager.model.generate(
            **formatted_inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
        )
    
    return outputs

def evaluate_outputs(model_manager: ModelManager, 
                  data_type: str,
                  messages: Any, 
                  labels: List[str],
                  max_new_tokens: int = 128, 
                  temperature: float = 0.7, 
                  do_sample: bool = True, 
                  top_p: float = 0.95, 
                  top_k: int = 50,
                  prompt_template: Optional[str] = None,
                  reasoning_effort: Optional[str] = None) -> str:
    
    if data_type == "chat":
        formatted_inputs = format_chat(model_manager, messages, reasoning_effort)
        answers = []
        for message in messages:
            for dict in message:
                if dict["role"] == "assistant":
                    answers.append(dict["content"])
    elif data_type == "instruction":
        formatted_inputs = format_instruction(model_manager, messages, prompt_template)
        answers = []
        for message in messages:
            answers.append(message["output"])
    else:
        raise ValueError(f"Invalid data type: {data_type}")

    outputs = model_manager.model.generate(**formatted_inputs, 
                                            max_new_tokens = max_new_tokens, 
                                            use_cache = True,
                                            temperature = temperature,
                                            do_sample = do_sample,
                                            top_p = top_p,
                                            top_k = top_k)
    
    
    predictions = []
    if data_type == "instruction":
        decoded_outputs = model_manager.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for decoded_output in decoded_outputs:
            head, sep, tail = decoded_output.rpartition("### Response:")
            pred = (tail if sep else decoded_output).strip()
            predictions.append(pred)
    elif data_type == "chat":
        sliced_outputs = []
        for i in range(outputs.size(0)):
            input_len = (formatted_inputs["input_ids"][i] != model_manager.tokenizer.pad_token_id).sum().item()
            sliced_output = outputs[i, input_len:]
            sliced_outputs.append(sliced_output)
        decoded_outputs = model_manager.tokenizer.batch_decode(sliced_outputs, skip_special_tokens=True)
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

def list_available_models(models_root: str) -> Dict[str, str]:
    """Return {name: path} for subdirectories under models_root."""
    out: Dict[str, str] = {}
    root = Path(models_root)
    if not root.exists():
        return out
    for item in root.iterdir():
        if item.is_dir():
            config_path = item / "config.json"
            if config_path.exists():
                out[item.name] = str(item)
    return dict(sorted(out.items()))

def list_available_datasets(data_root: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    base_path = Path(data_root)
    if not base_path.exists():
        return out
    for item in base_path.iterdir():
        if item.is_dir():
            for root, dirs, files in os.walk(item):
                for dir in dirs:
                    if "processed" in dir:
                        data_type = dir.split("/")[0]
                        processed_path = Path(root) / dir
                        test_json: Optional[str] = None
                        for j in processed_path.glob("test*.json"):
                            test_json = str(j)
                            break
                        out[item.name+"_"+data_type] = test_json
    return dict(sorted(out.items()))

def load_test_data(dataset_name: str, available_datasets: Dict) -> List[List[Dict[str, str]]]:
    path = available_datasets.get(dataset_name)
    assert path is not None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def select_models(model_names: Union[List[str], str], models_path: str) -> List[str]:
    models_dict = list_available_models(models_path)
    selected_models = {}

    if model_names == "all":
        model_list = list(models_dict.keys())
        selected_models = models_dict
    else:
        model_list = model_names if isinstance(model_names, list) else [model_names]
        selected_models = {model_name: models_dict[model_name] for model_name in model_list}

    return selected_models

def select_datasets(dataset_names: Union[List[str], str], data_path: str) -> List[str]:
    datasets_dict = list_available_datasets(data_path)
    selected_datasets = {}

    if dataset_names == "all":
        dataset_list = list(datasets_dict.keys())
        selected_datasets = datasets_dict
    else:
        dataset_list = dataset_names if isinstance(dataset_names, list) else [dataset_names]
        selected_datasets = {dataset_name: datasets_dict[dataset_name] for dataset_name in dataset_list}

    return selected_datasets

def benchmark_text(
    mm: ModelManager,
    dataset_names: Union[List[str], str],
    model_name: str,
    models_root: str,
    data_root: str,
    labels_map: Dict[str, Any],
    prompt_template: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_new_tokens: int = 128, temperature: float = 0.7, do_sample: bool = True, top_p: float = 0.95, top_k: int = 50,
    max_samples: Optional[int] = None,
    save_summary: bool = True,
    results_dir: str = "results",
) -> Dict[str, Dict[str, Optional[float]]]:
    
    selected_model = select_models(model_name, models_root)
    selected_datasets = select_datasets(dataset_names, data_root)
    model_path = selected_model.get(model_name, model_name)
    test_datas = [load_test_data(dataset_name, selected_datasets) for dataset_name in selected_datasets]

    results= {}

    if mm is None:
        mm = ModelManager()

    mm.load(model_name, model_path)
    for i, dataset_name in enumerate(selected_datasets.keys()):
        results[dataset_name] = {}
        test_data = test_datas[i]
        if max_samples:
            test_data = test_data[:max_samples]
        data_type = dataset_name.split("_")[-1]
        res = evaluate_outputs(mm, data_type, test_data, 
                                labels=labels_map.get(dataset_name), 
                                max_new_tokens=max_new_tokens, 
                                temperature=temperature, 
                                do_sample=do_sample, 
                                top_p=top_p, top_k=top_k,
                                prompt_template=prompt_template,
                                reasoning_effort=reasoning_effort)
        results[dataset_name] = res

    if save_summary:
        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "benchmark_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "models": model_name,
                "datasets": dataset_names,
                "results": results,
            }, f, indent=2)
        print(f"💾 Benchmark summary saved to {out_path}")

    return results


def compare_text(
    mm: ModelManager,
    dataset_name: str,
    models_name: List[str],
    models_root: str,
    data_root: str,
    labels: List[str],
    prompt_template: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_new_tokens: int = 128, temperature: float = 0.7, do_sample: bool = True, top_p: float = 0.95, top_k: int = 50,
    max_samples: Optional[int] = None,
    save_summary: bool = True,
    results_dir: str = "results",
) -> Dict[str, Optional[float]]:
    selected_models = select_models(models_name, models_root)
    selected_datasets = select_datasets(dataset_name, data_root)
    test_data = load_test_data(dataset_name, selected_datasets)
    data_type = dataset_name.split("_")[-1]
    results = {}
    if max_samples:
        test_data = test_data[:max_samples]
    for i, model_name in enumerate(selected_models.keys()):
        model_path = selected_models.get(model_name, model_name)
        mm.load(model_name, model_path)
        res = evaluate_outputs(mm, data_type, test_data, 
                                labels=labels, 
                                max_new_tokens=max_new_tokens, 
                                temperature=temperature, 
                                do_sample=do_sample, top_p=top_p, top_k=top_k,  
                                prompt_template=prompt_template,
                                reasoning_effort=reasoning_effort)
        results[model_name] = res

    if save_summary:
        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"comparison_{dataset_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": dataset_name,
                "models": models_name,
                "results": results,
            }, f, indent=2)
        print(f"💾 Comparison saved to {out_path}")

    return results


def interactive_text(models_root: str, data_root: str) -> None:
    """Simple REPL for chatting, switching models, benchmark and compare."""
    models = list_available_models(models_root)
    datasets = list_available_datasets(data_root)

    if not models:
        print(f"⚠️ No local models found under {models_root}. You can still load HF refs by name via 'switch <ref>'.")

    mm = ModelManager()

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
            current_model_name = user.split(" ", 1)[1].strip()
            ref_path = models.get(current_model_name, current_model_name)
            mm.load(current_model_name, ref_path) 
            print(f"🔄 Switched to: {current_model_name}")
            continue
        if user == "chat":
            if mm.model is None:
                print("⚠️ Load a model first (use 'switch').")
                continue
            print(f"Chatting with {current_model_name}")
            print("Type 'exit' to leave chat.")
            gen_args = input("Generation args separated by commas (max_new_tokens, temperature, do_sample, top_p, top_k, reasoning_effort): ").strip()
            while True:
                try:
                    msg = input("You: ").strip()
                except KeyboardInterrupt:
                    print("\n👋 Bye")
                    break
                if msg.lower() in ("exit", "quit", "q"):
                    break
                if not msg:
                    continue
                if gen_args:
                    max_new_tokens, temperature, do_sample, top_p, top_k, reasoning_effort = map(float, gen_args.split(","))
                    print("Model:", chat(mm, msg, max_new_tokens=max_new_tokens, temperature=temperature, 
                        do_sample=do_sample, top_p=top_p, top_k=top_k, reasoning_effort=reasoning_effort))
                else:
                    print("Model:", chat(mm, msg))
            continue
        if user == "benchmark":
            model_name = input("Model: ").strip()
            dss = input("Datasets (comma or 'all'): ").strip()
            datasets_names = "all" if dss == "all" else [d.strip() for d in dss.split(",")]
            max_samples = input("Num samples: ").strip()
            if max_samples:
                max_samples = int(max_samples)
            else:
                max_samples = None
            labels = input(f"Enter labels for each dataset {datasets_names} separated by commas e.g.[[label1,label2], [label1,label2]]: ").strip()
            if labels:
                labels = ast.literal_eval(labels)
                labels_map = {datasets_names[i]: labels[i] for i in range(len(datasets_names))}
                print(labels_map)
            else:
                labels_map = None
            gen_args = input("Generation args separated by commas (max_new_tokens, temperature, do_sample, top_p, top_k, reasoning_effort): ").strip()
            if gen_args:
                    max_new_tokens, temperature, do_sample, top_p, top_k, reasoning_effort = map(float, gen_args.split(","))
                    benchmark_text(mm, datasets_names, model_name, models_root, data_root, labels_map, max_new_tokens=max_new_tokens, 
                               temperature=temperature, do_sample=do_sample, top_p=top_p, top_k=top_k, max_samples=max_samples, reasoning_effort=reasoning_effort)
            else:
                benchmark_text(mm, datasets_names, model_name, models_root, data_root, labels_map)
            continue
        if user == "compare":
            mdls = input("Models (comma or 'all'): ").strip()
            model_names = 'all' if mdls == 'all' else [m.strip() for m in mdls.split(",")]
            dataset_name = input("Dataset: ").strip()
            max_samples = input("Num samples: ").strip()
            if max_samples:
                max_samples = int(max_samples)
            else:
                max_samples = None
            labels = input(f"Enter labels for the dataset {dataset_name} e.g.[label1,label2]: ").strip()
            if labels:
                labels = ast.literal_eval(labels)
                print(labels)
            else:
                labels = None
            gen_args = input("Generation args separated by commas (max_new_tokens, temperature, do_sample, top_p, top_k, reasoning_effort): ").strip()
            if gen_args:
                max_new_tokens, temperature, do_sample, top_p, top_k, reasoning_effort = map(float, gen_args.split(","))
                compare_text(mm, dataset_name, model_names, models_root, data_root, labels, max_new_tokens=max_new_tokens, 
                               temperature=temperature, do_sample=do_sample, top_p=top_p, top_k=top_k, max_samples=max_samples, reasoning_effort=reasoning_effort)
            else:
                compare_text(mm, dataset_name, model_names, models_root, data_root, labels)
            continue

        print("Unknown command. Try: chat | switch <model> | models | datasets | benchmark | compare | quit")


