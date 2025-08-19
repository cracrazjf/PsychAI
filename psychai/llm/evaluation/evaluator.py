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

from ..models import load_model, load_model_unsloth
from ..data import validate_format


class ModelManager:
    """Minimal model manager for loading/switching models."""

    def __init__(self, verbose: bool = True) -> None:
        self.model = None
        self.tokenizer = None
        self.model_ref: Optional[str] = None
        self.model_type: Optional[str] = None
        self.verbose = verbose

    def load(self, model_name: str, model_ref: str, model_type: str = "llama", max_seq_length: int = 512,
             load_in_4bit: bool = False) -> Tuple[Any, Any]:

        self.free_memory()

        self.model_name = model_name
        self.model_ref = model_ref
        self.model_type = model_type

        # Prefer Unsloth if available
        try:
            model, tokenizer = load_model_unsloth(
                model_name=model_name,
                model_path=model_ref,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                for_training=False,
            )
        except Exception:
            if self.verbose:
                print("⚠️ Unsloth not available or failed. Falling back to standard loading.")
            model, tokenizer = load_model(model_name=model_name, model_path=model_ref, for_training=False)

        self.model = model
        self.tokenizer = tokenizer
        print(f"✅ Model loaded: {model_name} from {model_ref}")
        return model, tokenizer

    def free_memory(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            try:
                del self.model
                print("✅ Current model deleted")
            except Exception:
                pass
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            try:
                del self.tokenizer
                print("✅ Current tokenizer deleted")
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        self.model = None
        self.tokenizer = None
        print("✅ Cache cleared")

    def info(self) -> Dict[str, Any]:
        if self.model is None:
            return {"status": "no model loaded"}
        device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else "unknown"
        num_parameters = sum(p.numel() for p in self.model.parameters()) if hasattr(self.model, "parameters") else 0
        return {
            "model_ref": self.model_ref,
            "model_type": self.model_type,
            "device": str(device),
            "num_parameters": int(num_parameters),
        }


class PromptFormatter:
    """Tiny helper for building prompts from text or chat messages."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def from_chat(self, messages: List[Dict[str, str]], reasoning_effort: Optional[str] = None) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                if reasoning_effort is not None:
                    return self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, reasoning_effort=reasoning_effort
                    )
                else:
                    return self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        # Fallback: join user/system text
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"[{role}] {content}")
        parts.append("[assistant]")
        return "\n".join(parts)

    def from_text(self, text: str, system: Optional[str] = None, instruction: Optional[str] = None, reasoning_effort: Optional[str] = None) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                content = f"{instruction}\n\n{text}" if instruction else text
                messages.append({"role": "user", "content": content})
                if reasoning_effort is not None:
                    return self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, reasoning_effort=reasoning_effort
                    )
                else:
                    return self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
            except Exception:
                pass
        parts: List[str] = []
        if system:
            parts.append(system)
        if instruction:
            parts.append(instruction)
        parts.append(text)
        return "\n\n".join(parts)


class Evaluator:
    """Simple text evaluator for generation and classification."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def _ensure_model(self, mm: ModelManager) -> None:
        if mm.model is None or mm.tokenizer is None:
            raise ValueError("Model and tokenizer must be set before evaluation.")

    def chat(self, mm: ModelManager, input_data: Union[str, List[Dict[str, str]]],
             max_new_tokens: int = 128, temperature: float = 0.7,
             do_sample: bool = True, top_p: float = 0.95, top_k: int = 50, reasoning_effort: Optional[str] = None) -> str:
        self._ensure_model(mm)
        formatter = PromptFormatter(mm.tokenizer)

        if isinstance(input_data, str):
            prompt = formatter.from_text(input_data, reasoning_effort=reasoning_effort)
        else:
            prompt = formatter.from_chat(input_data, reasoning_effort=reasoning_effort)

        inputs = mm.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        device = next(mm.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = mm.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=mm.tokenizer.eos_token_id,
                eos_token_id=mm.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        return mm.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def evaluate_generation(self, mm: ModelManager, test_data: List[List[Dict[str, str]]], max_samples: Optional[int] = None ,
                            max_new_tokens: int = 128, temperature: float = 0.7, do_sample: bool = True, top_p: float = 0.95, 
                            top_k: int = 50, reasoning_effort: Optional[str] = None) -> Dict[str, Any]:
        if not validate_format(test_data, "chat"):
            raise ValueError("test_data must be in chat format")

        if max_samples is not None:
            test_data = test_data[:max_samples]

        predictions: List[str] = []
        truths: List[str] = []
        formatter = PromptFormatter(mm.tokenizer)

        for conversation in tqdm.tqdm(test_data, desc="Evaluating generation"):
            user_messages = [m for m in conversation if m.get("role") != "assistant"]
            assistant_messages = [m for m in conversation if m.get("role") == "assistant"]
            if not user_messages or not assistant_messages:
                continue
            prompt = formatter.from_chat(user_messages)
            if reasoning_effort is not None:
                pred = self.chat(mm, prompt, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample, top_p=top_p, top_k=top_k, reasoning_effort=reasoning_effort)
            else:
                pred = self.chat(mm, prompt, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample, top_p=top_p, top_k=top_k)
            predictions.append(pred)
            truths.append(assistant_messages[-1].get("content", ""))

        results: Dict[str, Any] = {
            "total_samples": len(test_data),
            "processed_samples": len(predictions),
            "predictions": predictions,
            "ground_truths": truths,
        }

        # Simple text metrics
        if predictions and truths:
            exact = sum(1 for p, t in zip(predictions, truths) if p.strip() == t.strip())
            results["exact_match_accuracy"] = exact / len(predictions)

            pred_lengths = [len(p.split()) for p in predictions]
            truth_lengths = [len(t.split()) for t in truths]
            results["avg_prediction_length"] = float(np.mean(pred_lengths)) if pred_lengths else 0.0
            results["avg_truth_length"] = float(np.mean(truth_lengths)) if truth_lengths else 0.0

        return results

    def evaluate_classification(
        self,
        mm: ModelManager,
        test_data: List[List[Dict[str, str]]],
        labels: Optional[List[str]] = None,
        label_extractor: Optional[Callable[[str, Optional[List[str]]], str]] = None,
        max_samples: Optional[int] = None,
        max_new_tokens: int = 128, temperature: float = 0.7, do_sample: bool = True, top_p: float = 0.95, top_k: int = 50,
        reasoning_effort: Optional[str] = None
    ) -> Dict[str, Any]:
        if reasoning_effort is None:
            base = self.evaluate_generation(mm, test_data, max_samples=max_samples, max_new_tokens=max_new_tokens,
                                        temperature=temperature, do_sample=do_sample, top_p=top_p, top_k=top_k)
        else:
            base = self.evaluate_generation(mm, test_data, max_samples=max_samples, max_new_tokens=max_new_tokens,
                                        temperature=temperature, do_sample=do_sample, top_p=top_p, top_k=top_k, reasoning_effort=reasoning_effort)
        preds = base.get("predictions", [])
        truths = base.get("ground_truths", [])

        def default_extract(text: str, labels_list: Optional[List[str]]) -> str:
            txt = text.lower().strip()
            if labels_list:
                for lbl in labels_list:
                    if lbl.lower() in txt:
                        return lbl
            # fallback: first token
            return txt.split()[0] if txt else "unknown"

        extractor = label_extractor or default_extract
        pred_labels = [extractor(p, labels) for p in preds]
        true_labels = [extractor(t, labels) for t in truths]

        if accuracy_score is not None and classification_report is not None and confusion_matrix is not None:
            try:
                acc = float(accuracy_score(true_labels, pred_labels))
                uniq = sorted(list(set(true_labels + pred_labels)))
                report = classification_report(true_labels, pred_labels, labels=uniq, output_dict=True, zero_division=0)
                cm = confusion_matrix(true_labels, pred_labels, labels=uniq)
                base.update({
                    "accuracy": acc,
                    "classification_report": report,
                    "confusion_matrix": cm.tolist(),
                    "predicted_labels": pred_labels,
                    "true_labels": true_labels,
                    "unique_labels": uniq,
                })
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Metrics calculation failed: {e}")
                base.update({
                    "predicted_labels": pred_labels,
                    "true_labels": true_labels,
                })
        else:
            base.update({
                "predicted_labels": pred_labels,
                "true_labels": true_labels,
            })
        if self.verbose:
            self.print_results(base)
        return base

    @staticmethod
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
                print(f"- Pred: {preds[i][:120]}{'...' if len(preds[i]) > 120 else ''}")
                print(f"  True: {truths[i][:120]}{'...' if len(truths[i]) > 120 else ''}")


# -------------------------------
# Dataset/model discovery helpers
# -------------------------------

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


def list_available_datasets(data_root: str) -> Dict[str, Dict[str, Optional[str]]]:
    """Return mapping of dataset -> { processed_test: path or None }."""
    out: Dict[str, Dict[str, Optional[str]]] = {}
    root = Path(data_root)
    if not root.exists():
        return out
    for item in root.iterdir():
        if item.is_dir():
            processed = item / "processed"
            test_json: Optional[str] = None
            if processed.exists():
                # pick first test*.json
                for j in processed.glob("test*.json"):
                    test_json = str(j)
                    break
            out[item.name] = {"processed_test": test_json}
    return dict(sorted(out.items()))


def load_test_data(dataset_name: str, data_root: str) -> List[List[Dict[str, str]]]:
    info = list_available_datasets(data_root).get(dataset_name)
    if not info or not info.get("processed_test"):
        raise FileNotFoundError(f"No processed test data found for dataset '{dataset_name}' under {data_root}")
    path = info["processed_test"]
    assert path is not None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# -------------------------------
# Runner functions
# -------------------------------

def benchmark_text(
    model_names: Union[List[str], str],
    model_types: Union[List[str], str],
    dataset_names: Union[List[str], str],
    models_root: str,
    data_root: str,
    mm: Optional[ModelManager] = None,
    evaluator: Optional[Evaluator] = None,
    max_new_tokens: int = 128, temperature: float = 0.7, do_sample: bool = True, top_p: float = 0.95, top_k: int = 50,
    reasoning_effort: Optional[str] = None,
    labels_map: Optional[Dict[str, List[str]]] = None,
    num_samples: Optional[int] = None,
    save_summary: bool = True,
    results_dir: str = "results",
) -> Dict[str, Dict[str, Optional[float]]]:
    models_dict = list_available_models(models_root)

    if model_names == "all":
        model_list = list(models_dict.keys())
    else:
        model_list = model_names if isinstance(model_names, list) else [model_names]

    if dataset_names == "all":
        datasets_dict = list_available_datasets(data_root)
        dataset_list = list(datasets_dict.keys())
    else:
        dataset_list = dataset_names if isinstance(dataset_names, list) else [dataset_names]

    results: Dict[str, Dict[str, Optional[float]]] = {}

    if mm is None:
        mm = ModelManager()
    if evaluator is None:
        evaluator = Evaluator(verbose=True)

    for model_name, model_type in zip(model_list, model_types):
        model_path = models_dict.get(model_name, model_name)  # allow HF ref
        mm.load(model_name, model_path, model_type=model_type)

        results[model_name] = {}
        for dataset_name in dataset_list:
            try:
                test_data = load_test_data(dataset_name, data_root)
                labels = labels_map.get(dataset_name) if labels_map else None
                if reasoning_effort is None:
                    res = evaluator.evaluate_classification(mm, test_data, labels=labels, max_samples=num_samples, 
                                                            max_new_tokens=max_new_tokens, temperature=temperature, 
                                                            do_sample=do_sample, top_p=top_p, top_k=top_k)
                else:
                    res = evaluator.evaluate_classification(mm, test_data, labels=labels, max_samples=num_samples, 
                                                            max_new_tokens=max_new_tokens, temperature=temperature, 
                                                            do_sample=do_sample, top_p=top_p, top_k=top_k, reasoning_effort=reasoning_effort)
                results[model_name][dataset_name] = float(res.get("accuracy", res.get("exact_match_accuracy", 0.0)))
            except Exception as e:
                results[model_name][dataset_name] = None
                print(f"❌ {model_name} on {dataset_name} failed: {e}")
                traceback.print_exc()

    if save_summary:
        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "benchmark_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "models": model_list,
                "datasets": dataset_list,
                "results": results,
            }, f, indent=2)
        print(f"💾 Benchmark summary saved to {out_path}")

    return results


def compare_text(
    model_names: List[str],
    model_types: List[str],
    dataset_name: str,
    models_root: str,
    data_root: str,
    mm: Optional[ModelManager] = None,
    evaluator: Optional[Evaluator] = None,
    max_new_tokens: int = 128, temperature: float = 0.7, do_sample: bool = True, top_p: float = 0.95, top_k: int = 50,
    reasoning_effort: Optional[str] = None,
    labels: Optional[List[str]] = None,
    num_samples: Optional[int] = None,
    save_summary: bool = True,
    results_dir: str = "results",
) -> Dict[str, Optional[float]]:
    models_dict = list_available_models(models_root)
    test_data = load_test_data(dataset_name, data_root)

    if mm is None:
        mm = ModelManager()
    if evaluator is None:
        evaluator = Evaluator(verbose=True)

    results: Dict[str, Optional[float]] = {}
    for model_name, model_type in zip(model_names, model_types):
        model_path = models_dict.get(model_name, model_name)
        mm.load(model_name, model_path, model_type=model_type)
        try:
            if reasoning_effort is None:
                res = evaluator.evaluate_classification(mm, test_data, labels=labels, max_samples=num_samples,
                                                        max_new_tokens=max_new_tokens, temperature=temperature, 
                                                        do_sample=do_sample, top_p=top_p, top_k=top_k)
            else:
                res = evaluator.evaluate_classification(mm, test_data, labels=labels, max_samples=num_samples,
                                                        max_new_tokens=max_new_tokens, temperature=temperature, 
                                                        do_sample=do_sample, top_p=top_p, top_k=top_k, reasoning_effort=reasoning_effort)
            results[model_name] = float(res.get("accuracy", res.get("exact_match_accuracy", 0.0)))
        except Exception as e:
            results[model_name] = None
            print(f"❌ {model_name} failed on {dataset_name}: {e}")

    if save_summary:
        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"comparison_{dataset_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": dataset_name,
                "models": model_names,
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
    evaluator = Evaluator()

    # Load first model if exists
    # load_default = input("Load default model? (y/n)").strip()
    # if load_default == "y":
    #     current_model_name: Optional[str] = next(iter(models.keys()), None)
    #     if current_model_name:
    #         model_path = models[current_model_name]
    #         mm.load(current_model_name, model_path, model_type="local")
    #         print(f"✅ Loaded default model: {current_model_name}")
    #     else:
    #         print("⚠️ No default model found.")
    # else:
    #     print("You can always load a model later by using 'switch <model_name>'.")

    print("\n🎮 Welcome to PSYCHAI Interactive Text Evaluation")
    print("You can load/switch between models by using 'switch <model_name>'.")
    print("Commands: chat | switch <model_name>/<model_type> | models | datasets | benchmark | compare | quit")

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
            current_model_name, model_type = user.split(" ", 1)[1].strip().split("/")
            ref_path = models.get(current_model_name, current_model_name)
            mm.load(current_model_name, ref_path, model_type=model_type) 
            print(f"🔄 Switched to: {current_model_name}")
            continue
        if user == "chat":
            if mm.model is None:
                print("⚠️ Load a model first (use 'switch').")
                continue
            print(f"Chatting with {current_model_name}")
            print("Type 'exit' to leave chat.")
            if mm.model_type == "gpt-oss":
                gen_args = input("Generation args separated by commas (max_new_tokens, temperature, do_sample, top_p, top_k, reasoning_effort): ").strip()
            else:
                gen_args = input("Generation args separated by commas (max_new_tokens, temperature, do_sample, top_p, top_k): ").strip()
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
                    if mm.model_type == "gpt-oss":
                        max_new_tokens, temperature, do_sample, top_p, top_k, reasoning_effort = map(float, gen_args.split(","))
                        print("Model:", evaluator.chat(mm, msg, max_new_tokens=max_new_tokens, temperature=temperature, 
                            do_sample=do_sample, top_p=top_p, top_k=top_k, reasoning_effort=reasoning_effort))
                    else:
                        max_new_tokens, temperature, do_sample, top_p, top_k = map(float, gen_args.split(","))
                        print("Model:", evaluator.chat(mm, msg, max_new_tokens=max_new_tokens, temperature=temperature, 
                            do_sample=do_sample, top_p=top_p, top_k=top_k))
                else:
                    print("Model:", evaluator.chat(mm, msg))
            continue
        if user == "benchmark":
            mdl = input("Models (comma or 'all'): ").strip()
            mdl_list = "all" if mdl == "all" else [m.strip() for m in mdl.split(",")]
            model_names = list(models.keys()) if mdl_list == 'all' else mdl_list

            mdl_types = input(f"Enter model types for each model {model_names} (separated by comma): ").strip()
            if mdl_types:
                mdl_types = ast.literal_eval(mdl_types)
                mdl_types_map = {model_names[i]: mdl_types[i] for i in range(len(model_names))}
                print(mdl_types_map)
            else:
                mdl_types_map = None
        

            dss = input("Datasets (comma or 'all'): ").strip()
            dss_list = "all" if dss == "all" else [d.strip() for d in dss.split(",")]
            num_samples = input("Num samples: ").strip()
            if num_samples:
                num_samples = int(num_samples)
            else:
                num_samples = None
            datasets_names = list(datasets.keys()) if dss_list == 'all' else dss_list
            labels = input(f"Enter labels for each dataset {datasets_names} separated by commas e.g.[[label1,label2], [label1,label2]]: ").strip()
            if labels:
                labels = ast.literal_eval(labels)
                labels_map = {datasets_names[i]: labels[i] for i in range(len(datasets_names))}
                print(labels_map)
            else:
                labels_map = None
            
            if mm.model_type == "gpt-oss":
                gen_args = input("Generation args separated by commas (max_new_tokens, temperature, do_sample, top_p, top_k, reasoning_effort): ").strip()
            else:
                gen_args = input("Generation args separated by commas (max_new_tokens, temperature, do_sample, top_p, top_k): ").strip()
            if gen_args:
                if mm.model_type == "gpt-oss":
                    max_new_tokens, temperature, do_sample, top_p, top_k, reasoning_effort = map(float, gen_args.split(","))
                    benchmark_text(mdl_list, mdl_types_map, dss_list, models_root, data_root, mm=mm, evaluator=evaluator, max_new_tokens=max_new_tokens, 
                               temperature=temperature, do_sample=do_sample, top_p=top_p, top_k=top_k, num_samples=num_samples, labels_map=labels_map, reasoning_effort=reasoning_effort)
                else:
                    max_new_tokens, temperature, do_sample, top_p, top_k = map(float, gen_args.split(","))
                    benchmark_text(mdl_list, mdl_types_map, dss_list, models_root, data_root, mm=mm, evaluator=evaluator, max_new_tokens=max_new_tokens, 
                               temperature=temperature, do_sample=do_sample, top_p=top_p, top_k=top_k, num_samples=num_samples, labels_map=labels_map)
            else:
                benchmark_text(mdl_list, mdl_types_map, dss_list, models_root, data_root, mm=mm, evaluator=evaluator, num_samples=num_samples, labels_map=labels_map)
            continue
        if user == "compare":
            mdl = input("Models (comma): ").strip()
            mdl_names = [m.strip() for m in mdl.split(",")]
            mdl_types = input(f"Enter model types for each model {mdl_names} (separated by comma): ").strip()
            if mdl_types:
                mdl_types = ast.literal_eval(mdl_types)
                mdl_types_map = {mdl_names[i]: mdl_types[i] for i in range(len(mdl_names))}
                print(mdl_types_map)
            else:
                mdl_types_map = None
            dset = input("Dataset: ").strip()
            num_samples = input("Num samples: ").strip()
            if num_samples:
                num_samples = int(num_samples)
            else:
                num_samples = None
            labels = input(f"Enter labels for the dataset {dset} e.g.[label1,label2]: ").strip()
            if labels:
                labels = ast.literal_eval(labels)
                print(labels)
            else:
                labels = None
            if mm.model_type == "gpt-oss":
                gen_args = input("Generation args separated by commas (max_new_tokens, temperature, do_sample, top_p, top_k, reasoning_effort): ").strip()
            else:
                gen_args = input("Generation args separated by commas (max_new_tokens, temperature, do_sample, top_p, top_k): ").strip()
            if gen_args:
                if mm.model_type == "gpt-oss":
                    max_new_tokens, temperature, do_sample, top_p, top_k, reasoning_effort = map(float, gen_args.split(","))
                    compare_text(mdl_names, mdl_types_map, dset, models_root, data_root, mm=mm, evaluator=evaluator, max_new_tokens=max_new_tokens, 
                               temperature=temperature, do_sample=do_sample, top_p=top_p, top_k=top_k, num_samples=num_samples, labels=labels, reasoning_effort=reasoning_effort)
                else:
                    max_new_tokens, temperature, do_sample, top_p, top_k = map(float, gen_args.split(","))
                    compare_text(mdl_names, mdl_types_map, dset, models_root, data_root, mm=mm, evaluator=evaluator, max_new_tokens=max_new_tokens, 
                                 temperature=temperature, do_sample=do_sample, top_p=top_p, top_k=top_k, num_samples=num_samples, labels=labels)
            else:
                compare_text(mdl_names, mdl_types_map, dset, models_root, data_root, mm=mm, evaluator=evaluator, num_samples=num_samples, labels=labels)
            continue

        print("Unknown command. Try: chat | switch <model> | models | datasets | benchmark | compare | quit")


