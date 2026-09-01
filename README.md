# PsychAI

**A modular research framework for training and evaluating language models in psychology research.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PsychAI is built for learning-dynamics experiments: train small language models from scratch (or from Hugging Face checkpoints), evaluate pretrained LLMs with 4-bit quantization and LoRA, and hook in custom per-batch or whole-dataset evaluation functions — with log-spaced checkpointing so the early epochs where learning happens fastest are sampled densely.

---

## Table of contents

- [Installation](#installation)
- [Repository layout](#repository-layout)
- [Configuration](#configuration)
- [Workflow A: train a language model](#workflow-a-train-a-language-model)
- [Workflow B: pretrained LLMs (4-bit + LoRA)](#workflow-b-pretrained-llms-4-bit--lora)
- [Custom evaluation functions](#custom-evaluation-functions)
- [Running on a GPU](#running-on-a-gpu)
- [License](#license)

---

## Installation

```bash
git clone https://github.com/cracrazjf/PsychAI.git
cd PsychAI
pip install -e .
```

Or install straight from GitHub:

```bash
pip install "psychai @ git+https://github.com/cracrazjf/PsychAI.git"
```

Core dependencies (`torch`, `transformers`, `datasets`, `safetensors`, …) install automatically. Make sure `torch` matches your CUDA version — on a cluster you may want to install it first from the matching CUDA index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Optional extras** — install only what you need:

| Extra | Command | Adds |
|---|---|---|
| LLM finetuning | `pip install -e ".[llm]"` | Unsloth, TRL, PEFT |
| Vision | `pip install -e ".[vision]"` | TorchVision, TIMM |

Verify your GPU is visible before running anything:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Repository layout

```
psychai/
├── config/           # TrainingConfig dataclasses + YAML loader with dot-path overrides
├── language/
│   ├── lm.py         # Workflow A: full training loop for LMs (scratch or HF checkpoints)
│   ├── llm.py        # Workflow B: pretrained LLM loading (Unsloth/4-bit/LoRA) + evaluation
│   ├── tokenizer.py  # build custom tokenizers (normalizers, pre-tokenizers, training)
│   └── load_data.py  # load raw data as plain text, chat, or instruction format
├── nn_builder/       # build custom architectures from config specs
├── artificial_data/  # synthetic language generators (e.g. xAyBz)
├── vision/           # vision model training and data loading
└── visualization/    # figure drawing helpers
```

## Configuration

Experiments are driven by the `TrainingConfig` dataclass (`psychai/config/config.py`), which nests:

| Section | What it controls |
|---|---|
| `model` | model name/path, wrapper (`causal_lm`, `masked_lm`, `seq_cls`, `tok_cls`), tokenizer path, `max_seq_length`, `load_in_4bit`, custom-architecture sizes |
| `data` | train/val/test paths, `window_size`, `stride`, `batch_size`, padding, shuffling, `num_workers` |
| `optim` | optimizer (`adam`/`adamw`/`sgd`), `lr`, `weight_decay`, `momentum` |
| `logging` | eval strategy, `dataset_eval`, `save_total_limit`, `return_embeddings`, `return_weights`, `save_model` |
| top level | `exp_name`, `exp_dir`, `num_runs`, `num_epochs`, `seed`, `device`, `task`, `bp_method` |

Build a config in Python, or load a YAML file with dot-path overrides:

```python
from psychai.config import load_config

cfg = load_config(
    "configs/my_experiment.yaml",
    overrides={"device": "cuda", "data.batch_size": 64, "optim.lr": 1e-4},
)
```

<details>
<summary><b>Example YAML config</b> (keys mirror the dataclass fields)</summary>

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

</details>

## Workflow A: train a language model

Full training loop for small LMs — from scratch via `nn_builder` or from any Hugging Face checkpoint.

```python
from psychai.config import load_config
from psychai.language.lm import TrainingManager

cfg = load_config("configs/my_experiment.yaml", overrides={"device": "cuda"})
tm = TrainingManager(cfg)
tm.train()      # optionally: tm.train(weight_init_fn=..., eval_fn=...)
```

For each of `num_runs` runs (seed = `cfg.seed + run`), `train()`:

1. Seeds torch / CUDA / numpy / random and creates `exp_dir/run_{n}/` — **`exp_dir` must be set**.
2. Loads model + tokenizer. If `model.name` or `model.model_type` contains `"custom"`, the model is built from a `nn_builder` config at `model.path` (tokenizer from `model.tokenizer_path`); otherwise a Hugging Face checkpoint is loaded via the wrapper chosen by `model.wrapper`.
3. Tokenizes the datasets (they need a `"text"` column), concatenates all tokens, and slices them into fixed windows of `window_size` with `stride`.
4. Trains for `num_epochs`. At **log-spaced epoch checkpoints** (about `save_total_limit` epochs spread on a log scale, always including the first and last) it evaluates on the val set, appends the train loss to `log.jsonl`, and saves a resumable checkpoint. Log spacing samples early epochs densely — where learning dynamics change fastest. Because the schedule uses `log10(num_epochs − 1)`, use `num_epochs >= 2`.
5. If `logging.save_model` is true, exports the final model to `run_{n}/export/` (safetensors by default).

Each run produces:

```
exp_dir/run_1/
├── log.jsonl            # train loss at each checkpoint epoch
├── eval_results.json    # one JSON line per eval record
├── checkpoint-*/        # resumable checkpoints (model + optimizer state)
└── export/              # final model weights
```

## Workflow B: pretrained LLMs (4-bit + LoRA)

Load and evaluate pretrained instruction/chat LLMs on a GPU. Requires the `[llm]` extras; if Unsloth imports successfully it is used automatically (faster, memory-efficient), otherwise loading falls back to plain Hugging Face + bitsandbytes. Effectively CUDA-only — 4-bit quantization and Unsloth need a GPU (there is a slow fp32 CPU fallback).

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

`evaluate()` runs under `torch.inference_mode()`, batch by batch, and writes JSON lines to `eval_path`. Data helpers in `psychai.language` (`load_any`, `load_any_as_chat`, `load_any_as_instruction`) convert raw files into chat/instruction format. For running LoRA finetuning, pair the loaded PEFT model with TRL's `SFTTrainer` (installed with the `[llm]` extras).

## Custom evaluation functions

Pass an `eval_fn` to `train()` or `evaluate()` to compute your own metrics. It is called as:

```python
eval_fn(mm, cfg, idxs, input_ids, labels, logits, preds, embedding_list, weights)
```

and must return a `dict` (merged into one JSON line with epoch/step/loss) or a `list` of dicts (each written as its own line) to `eval_results.json`.

- **Per-batch mode** (`logging.dataset_eval: false`, default): called once per eval batch with that batch's tensors.
- **Dataset mode** (`logging.dataset_eval: true`): the whole val set's inputs/labels/logits/preds/embeddings are accumulated and `eval_fn` is called **once** with lists of per-batch tensors — for metrics that need the full dataset, e.g. similarity structure across all items. Accumulated logits stay on the GPU, so budget VRAM for `val_set_size × window_size × vocab_size`, or keep the val set small.

Set `logging.return_embeddings: true` (with `layer_of_interest` / `embed_type`) to receive per-token embeddings, and `logging.return_weights: true` for model weights (custom models only).

## Running on a GPU

- **Device:** set `device: cuda` in the config — the trainer moves the model and all batches follow. Workflow B places the model itself with `device_map="auto"`.
- **Throughput:** raise `data.batch_size` / `data.window_size` toward VRAM limits and set `data.num_workers > 0` so the dataloader doesn't starve the GPU.
- **Memory:** call `tm.mm.free_memory()` before loading a second model in one process — it deletes the model/tokenizer and empties the CUDA cache. An 8B model in 4-bit needs ~6 GB VRAM; bf16 needs ~16 GB plus activations.
- **Reproducibility:** CUDA is seeded per run, and runs use consecutive seeds so you can average across them.
- **Long jobs:** run under `nohup`/`tmux` or your scheduler; losses land in `log.jsonl` / `eval_results.json` as the run goes, so you can watch with `tail -f`.

  ```bash
  nohup python run_experiment.py > train.out 2>&1 &
  ```

  <details>
  <summary><b>Slurm example</b></summary>

  ```bash
  #!/bin/bash
  #SBATCH --gres=gpu:1
  #SBATCH --mem=32G
  #SBATCH --time=24:00:00
  module load cuda
  python run_experiment.py --config configs/my_experiment.yaml
  ```

  </details>

- **Known sharp edges:** `exp_dir` must be set or `train()` fails when writing logs; `num_epochs` must be at least 2; local `.jsonl` training files currently hit a `load_dataset("jsonl", ...)` call that should use the `"json"` builder — prefer a Hugging Face dataset directory/name for `data.train_path` until that's fixed.

## License

MIT — see [LICENSE](LICENSE).
