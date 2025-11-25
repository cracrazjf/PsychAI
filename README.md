## PsychAI

PsychAI is a modular research framework for training and evaluating models (language-first, with optional extensions for LLM finetuning and vision pipelines).

### Installation

**From source**

```bash
pip install --upgrade --force-reinstall --no-deps --no-cache-dir "psychai @ git+https://github.com/cracrazjf/PsychAI.git"
```

### Optional extras

Install only what you need:

- `pip install "psychai[llm]"` → adds Unsloth/TRL/PEFT stack for LLM finetuning.
- `pip install "psychai[vision]"` → adds Pillow/TorchVision/TIMM/Accelerate for multimodal or vision setups.

You can combine extras, e.g. `pip install "psychai[llm,vision]"`.

### Local development

```bash
git clone https://github.com/cracrazjf/PsychAI.git
cd PsychAI
pip install -e .[dev]
pytest
```

`[dev]` can include your preferred tooling (formatters, linters, docs) when defined in `pyproject.toml`.

### Basic usage

```python
from psychai.language import make_pretokenizer, load_any
from psychai.config import TrainingConfig
```

See the examples directory and docstrings for feature-specific workflows.
