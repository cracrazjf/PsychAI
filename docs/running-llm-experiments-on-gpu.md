# Running LLM experiments on GPU with PsychAI

This guide covers how to use this repo to run language-model experiments on a CUDA GPU. There are two workflows:

1. **Training a language model** (from scratch or from a Hugging Face checkpoint) with `psychai.language.lm` — full training loop, evaluation hooks, checkpointing.
2. **Loading and evaluating a pretrained LLM** (optionally 4-bit quantized, with LoRA adapters) with `psychai.language.llm` — Unsloth-accelerated model loading and a batched evaluation loop.

Both are driven by the same config system in `psychai.config`.

---

## 1. Setup on a GPU machine

```bash
git clone https://github.com/cracrazjf/PsychAI.git
cd PsychAI
pip install -e .
```

Core dependencies (`torch`, `transformers`, `datasets`, `safetensors`, …) are installed automatically. Make sure the installed `torch` matches your CUDA version — on a cluster you may want to install torch first from the matching CUDA index, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For the **pretrained-LLM workflow** (`psychai.language.llm`) also install the LLM extras:

```bash
pip install -e ".[llm]"
```

This adds Unsloth, TRL, and PEFT. If Unsloth imports successfully, `llm.ModelManager` uses `FastLanguageModel` (faster, memory-efficient); otherwise it falls back to plain Hugging Face + bitsandbytes 4-bit loading. Verify your GPU is visible before running anything:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2. The config system

Experiments are configured with the `TrainingConfig` dataclass (`psychai/config/config.py`), which nests:

| Section | Class | What it controls |
|---|---|---|
| `model` | `ModelConfig` | model name/path, wrapper (`causal_lm`, `masked_lm`, `seq_cls`, `tok_cls`), tokenizer path, `max_seq_length`, `load_in_4bit`, custom-architecture sizes |
| `data` | `DataConfig` | train/val/test paths, `window_size`, `stride`, `batch_size`, padding, shuffling, num_workers |
| `optim` | `OptimConfig` | optimizer (`adam`/`adamw`/`sgd`), `lr`, `weight_decay`, `momentum` |
| `logging` | `LoggingConfig` | eval strategy, `dataset_eval`, `save_total_limit`, `return_embeddings` / `return_weights`, `save_model`, safetensors preference |
| top level | `TrainingConfig` | `exp_name`, `exp_dir`, `num_runs`, `num_epochs`, `seed`, `device`, `task`, `bp_method` |

You can build a config in Python, or load a YAML file with dot-path overrides:

```python
from psychai.config import load_config

cfg = load_config(
    "configs/my_experiment.yaml",
    overrides={"device": "cuda", "data.batch_size": 64, "optim.lr": 1e-4},
)
```

Example YAML (keys mirror the dataclass fields):

```yaml
exp_name: gpt2_childes
exp_dir: ./experiments/gpt2_childes
num_runs: 3          # each run uses seed + run_index
num_epochs: 100      # must be >= 2 (checkpoint schedule is log-spaced)
seed: 42
device: cuda
task: causal_lm
bp_method: bptt      # "continuous" carries recurrent state across batches

model:
  name: gpt2
  wrapper: causal_lm
  model_type: gpt2

data:
  train_path: ./data/train      # HF dataset dir/name with a "text" column
  val_path: ./data/val
  window_size: 128
  stride: 128
  batch_size: 32
  shuffle_dataloader: true

optim:
  optimizer: adamw
  lr: 3.0e-4
  weight_decay: 0.01

logging:
  eval_strategy: epoch
  dataset_eval: false
  save_total_limit: 10          # also sets how many log-spaced checkpoints
  return_embeddings: false
  save_model: true
```

**GPU-relevant settings:** set `device: cuda` (the trainer moves the model there and all batches follow), raise `data.batch_size` / `data.window_size` until you approach VRAM limits, and set `data.num_workers > 0` so the dataloader doesn't starve the GPU.

---

## 3. Workflow A — training a language model (`psychai.language.lm`)

```python
from psychai.config import load_config
from psychai.language.lm import TrainingManager

cfg = load_config("configs/my_experiment.yaml", overrides={"device": "cuda"})
tm = TrainingManager(cfg)
tm.train()                       # optionally: tm.train(weight_init_fn=..., eval_fn=...)
```

`train()` does, per run (`num_runs` total, seed = `cfg.seed + run`):

1. Seeds torch / CUDA / numpy / random and creates `exp_dir/run_{n}/`. **`exp_dir` must be set** — the log and eval paths are created from it.
2. Loads model + tokenizer. If `model.name` or `model.model_type` contains `"custom"`, it builds a model from a `nn_builder` config/spec at `model.path` (with `model.tokenizer_path` for the tokenizer); otherwise it pulls a Hugging Face checkpoint via the wrapper class chosen by `model.wrapper`.
3. Tokenizes `data.train_path` / `data.val_path` (datasets need a `"text"` column), concatenates all tokens, and slices them into fixed windows of `window_size` with `stride`.
4. Trains for `num_epochs`, and at **log-spaced epoch checkpoints** (roughly `save_total_limit` epochs spread on a log scale, always including the first and last epoch) it: evaluates on the val set, appends the train loss to `log.jsonl`, and saves an optimizer-resumable checkpoint (`save_total_limit` most recent kept). This log spacing is designed for learning-dynamics experiments where early epochs matter most. Because the schedule uses `log10(num_epochs - 1)`, use `num_epochs >= 2`.
5. If `logging.save_model` is true, exports the final model to `run_{n}/export/` (safetensors by default).

### Outputs per run

```
exp_dir/run_1/
├── log.jsonl            # {"epoch": ..., "train_loss": ...} at each checkpoint epoch
├── eval_results.json    # one JSON line per eval record (see eval_fn below)
├── checkpoint-*/        # resumable checkpoints (model + optimizer state)
└── export/              # final model weights
```

### Custom evaluation (`eval_fn`)

Pass `eval_fn` to `train()` to compute your own metrics. It is called with:

```python
eval_fn(mm, cfg, idxs, input_ids, labels, logits, preds, embedding_list, weights)
```

and must return a `dict` (merged into one JSON line with epoch/step/loss) or a `list` of dicts (each written as its own JSON line) to `eval_results.json`.

- **Per-batch mode** (`logging.dataset_eval: false`, default): called once per eval batch; the tensor arguments are that batch's tensors.
- **Dataset mode** (`logging.dataset_eval: true`): the whole val set's inputs/labels/logits/preds/embeddings are accumulated and `eval_fn` is called **once** with lists of per-batch tensors — use this for metrics that need the full dataset (e.g. similarity structure across all items). Note the accumulated logits/embeddings stay on the GPU, so budget VRAM for `val_set_size × window_size × vocab_size` logits, or keep the val set small.

Set `logging.return_embeddings: true` (with `layer_of_interest` / `embed_type`) to receive per-token embeddings, and `logging.return_weights: true` for model weights (custom models only).

---

## 4. Workflow B — pretrained LLMs, 4-bit + LoRA (`psychai.language.llm`)

This path is for evaluating (or preparing to finetune) pretrained instruction/chat LLMs on a GPU. It requires the `[llm]` extras and effectively requires CUDA — 4-bit quantization and Unsloth are GPU-only (there is a slow fp32 CPU fallback).

```python
from psychai.config import load_config
from psychai.language.llm import TrainingManager

cfg = load_config("configs/llama_eval.yaml")
tm = TrainingManager(cfg)

# Load in 4-bit on the GPU (device_map="auto" places it for you)
tm.mm.load_model(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    model_path=None,
    model_type="llama",
    max_seq_length=cfg.model.max_seq_length,
    load_in_4bit=True,
    dtype="bfloat16",
)

# Optional: attach LoRA adapters for finetuning
tm.mm.apply_lora(rank=16, alpha=32, dropout=0.05,
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
tm.mm.choose_chat_template()   # sets the llama-3.1 chat template (Unsloth only)

# Evaluate with a custom metric function
tm.evaluate(dataloader=my_dataloader, eval_fn=my_eval_fn, epoch=0,
            eval_path="results/llama_eval.jsonl")
```

`evaluate()` runs the model under `torch.inference_mode()`, batch by batch, and calls `eval_fn` with the same signature as in Workflow A (`hidden_states` from `logging.layer_of_interest` are used for embeddings when `logging.return_embeddings` is on). Results are written as JSON lines to `eval_path`.

Data helpers in `psychai.language` (`load_any`, `load_any_as_chat`, `load_any_as_instruction`) convert raw files into chat/instruction format for building eval or finetuning dataloaders. For actually running LoRA finetuning, pair the loaded PEFT model with TRL's `SFTTrainer` (installed with the `[llm]` extras).

---

## 5. GPU practicalities

- **Memory:** call `tm.mm.free_memory()` before loading a second model in the same process — it deletes the model/tokenizer and empties the CUDA cache. For 8B-class models use `load_in_4bit=True` (~6 GB VRAM); bf16 needs ~16 GB plus activations.
- **Reproducibility:** `lm.TrainingManager.train()` seeds CUDA per run; runs `run_1..run_N` use consecutive seeds so you can average across them.
- **Long jobs:** run under `nohup`/`tmux` or your scheduler; progress and losses land in `log.jsonl` / `eval_results.json` as the run goes, so you can monitor with `tail -f`.

```bash
nohup python run_experiment.py > train.out 2>&1 &
```

- **Slurm example:**

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
module load cuda
python run_experiment.py --config configs/my_experiment.yaml
```

- **Known sharp edges:** `exp_dir` must be set or `train()` fails when writing logs; `num_epochs` must be at least 2 (log-spaced checkpoint schedule); local `.jsonl` training files currently hit a `load_dataset("jsonl", ...)` call that should use the `"json"` builder — prefer a Hugging Face dataset directory/name for `data.train_path` until that's fixed.
