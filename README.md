## PsychAI

Framework for training and evaluating language models/optional audio/vision utilities for psychology research.

### Install

- Stable:
  - `pip install psychai`
- From source:
  - `pip install "psychai @ git+https://github.com/cracrazjf/PsychAI.git"`

Optional extras:
  - `pip install "psychai[audio,vision,unsloth] @ git+https://github.com/cracrazjf/PsychAI.git"`

### Quickstart (Python API)

```python
from PsychAI.training import Trainer
from PsychAI.config import TextTrainingConfig
from PsychAI.data import load_csv_as_chat

# Prepare data (chat format)
train_conversations = load_csv_as_chat("train.csv", input_column="input", output_column="output")
eval_conversations = load_csv_as_chat("eval.csv", input_column="input", output_column="output")

config = TextTrainingConfig(
    MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct",
    OUTPUT_DIR="./outputs/run1",
    MAX_LENGTH=512,
    NUM_EPOCHS=1,
)

trainer = Trainer(config)
trainer.train(train_conversations, eval_conversations)
```

### CLI usage

- Train:
  - `psychai-train --data-name dataname --train-data data/train.json --eval-data data/eval.json --data-format chat --model-name modelname --model-ref meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir outputs/run1 --unsloth`
  - Common flags: `--epochs`, `--batch-size`, `--grad-accum-steps`, `--lr`, `--lora-rank`, `--lora-alpha`, `--lora-dropout`, `--no-lora`

- Interactive evaluation REPL:
  - `psychai-evaluate` (lists local models/datasets if present, lets you chat/switch/benchmark/compare)
  - Optional overrides: `--models-root /path/to/models` `--data-root /path/to/data`


### Environment and caching

Optionally set cache locations (default to `./data/...`):
- `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, `HF_HOME`, `TORCH_HOME`

Hugging Face auth (if needed for gated models):
- `export HF_TOKEN=...`


### Contributing

PRs welcome. Run linters/tests before submitting:
- `pip install -e .[dev]`
- `pytest -q`

### License

MIT


