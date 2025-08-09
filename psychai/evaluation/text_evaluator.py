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
import gc
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

    def load(self, model_ref: str, model_type: str = "local", max_seq_length: int = 512,
             load_in_4bit: bool = False) -> Tuple[Any, Any]:
        """Load a model by reference.

        model_type: "local" | "huggingface" | "finetuned"
        """

        self.free_memory()

        self.model_ref = model_ref
        self.model_type = model_type

        # Prefer Unsloth if available
        try:
            model, tokenizer = load_model_unsloth(
                model_name=model_ref,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                for_training=False,
            )
        except Exception:
            if self.verbose:
                print("⚠️ Unsloth not available or failed. Falling back to standard loading.")
            model, tokenizer = load_model(model_name=model_ref, for_training=False)

        self.model = model
        self.tokenizer = tokenizer
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

    def from_chat(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
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

    def from_text(self, text: str, system: Optional[str] = None, instruction: Optional[str] = None) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                content = f"{instruction}\n\n{text}" if instruction else text
                messages.append({"role": "user", "content": content})
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


class TextEvaluator:
    """Simple text evaluator for generation and classification."""

    def __init__(self, model: Any = None, tokenizer: Any = None, verbose: bool = True) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose

    def set_model(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def _ensure_model(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be set before evaluation.")

    def chat(self, input_data: Union[str, List[Dict[str, str]]],
             max_new_tokens: int = 128, temperature: float = 0.7,
             do_sample: bool = True) -> str:
        self._ensure_model()
        formatter = PromptFormatter(self.tokenizer)

        if isinstance(input_data, str):
            prompt = formatter.from_text(input_data)
        else:
            prompt = formatter.from_chat(input_data)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            self.model = self.model.cuda()

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def evaluate_generation(self, test_data: List[List[Dict[str, str]]], max_samples: Optional[int] = None) -> Dict[str, Any]:
        self._ensure_model()
        if not validate_format(test_data, "chat"):
            raise ValueError("test_data must be in chat format")

        if max_samples is not None:
            test_data = test_data[:max_samples]

        predictions: List[str] = []
        truths: List[str] = []
        formatter = PromptFormatter(self.tokenizer)

        for conversation in test_data:
            user_messages = [m for m in conversation if m.get("role") != "assistant"]
            assistant_messages = [m for m in conversation if m.get("role") == "assistant"]
            if not user_messages or not assistant_messages:
                continue
            prompt = formatter.from_chat(user_messages)
            pred = self.chat(prompt)
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
        test_data: List[List[Dict[str, str]]],
        labels: Optional[List[str]] = None,
        label_extractor: Optional[Callable[[str, Optional[List[str]]], str]] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        base = self.evaluate_generation(test_data, max_samples=max_samples)
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
    dataset_names: Union[List[str], str],
    models_root: str,
    data_root: str,
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
    mm = ModelManager()
    evaluator = TextEvaluator(verbose=False)

    for model_name in model_list:
        model_path = models_dict.get(model_name, model_name)  # allow HF ref
        # Decide type: local if exists, else huggingface
        model_type = "local" if Path(model_path).exists() else "huggingface"
        mm.load(model_path, model_type=model_type)
        evaluator.set_model(mm.model, mm.tokenizer)

        results[model_name] = {}
        for dataset_name in dataset_list:
            try:
                test_data = load_test_data(dataset_name, data_root)
                labels = labels_map.get(dataset_name) if labels_map else None
                res = evaluator.evaluate_classification(test_data, labels=labels, max_samples=num_samples)
                results[model_name][dataset_name] = float(res.get("accuracy", res.get("exact_match_accuracy", 0.0)))
            except Exception as e:
                results[model_name][dataset_name] = None
                print(f"❌ {model_name} on {dataset_name} failed: {e}")

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
    dataset_name: str,
    models_root: str,
    data_root: str,
    labels: Optional[List[str]] = None,
    num_samples: Optional[int] = None,
    save_summary: bool = True,
    results_dir: str = "results",
) -> Dict[str, Optional[float]]:
    models_dict = list_available_models(models_root)
    test_data = load_test_data(dataset_name, data_root)

    mm = ModelManager()
    evaluator = TextEvaluator(verbose=False)

    results: Dict[str, Optional[float]] = {}
    for model_name in model_names:
        model_path = models_dict.get(model_name, model_name)
        model_type = "local" if Path(model_path).exists() else "huggingface"
        mm.load(model_path, model_type=model_type)
        evaluator.set_model(mm.model, mm.tokenizer)
        try:
            res = evaluator.evaluate_classification(test_data, labels=labels, max_samples=num_samples)
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
    evaluator = TextEvaluator()

    # Load first model if exists
    current_model_name: Optional[str] = next(iter(models.keys()), None)
    if current_model_name:
        model_path = models[current_model_name]
        mm.load(model_path, model_type="local")
        evaluator.set_model(mm.model, mm.tokenizer)
        print(f"✅ Loaded default model: {current_model_name}")

    print("\n🎮 Interactive Text Evaluation")
    print("Commands: chat | switch <model_or_hf_ref> | models | datasets | benchmark | compare | quit")

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
            ref = user.split(" ", 1)[1].strip()
            ref_path = models.get(ref, ref)
            mtype = "local" if Path(ref_path).exists() else "huggingface"
            mm.load(ref_path, model_type=mtype)
            evaluator.set_model(mm.model, mm.tokenizer)
            current_model_name = ref
            print(f"🔄 Switched to: {ref}")
            continue
        if user == "chat":
            if evaluator.model is None:
                print("⚠️ Load a model first (use 'switch').")
                continue
            print(f"Chatting with {current_model_name}")
            print("Type 'exit' to leave chat.")
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
                print("Model:", evaluator.chat(msg))
            continue
        if user == "benchmark":
            mdl = input("Models (comma or 'all'): ").strip()
            mdl_list = "all" if mdl == "all" else [m.strip() for m in mdl.split(",")]
            dss = input("Datasets (comma or 'all'): ").strip()
            dss_list = "all" if dss == "all" else [d.strip() for d in dss.split(",")]
            benchmark_text(mdl_list, dss_list, models_root, data_root)
            continue
        if user == "compare":
            mdl = input("Models (comma): ").strip()
            mdl_names = [m.strip() for m in mdl.split(",")]
            dset = input("Dataset: ").strip()
            compare_text(mdl_names, dset, models_root, data_root)
            continue

        print("Unknown command. Try: chat | switch <model> | models | datasets | benchmark | compare | quit")


