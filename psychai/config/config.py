from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

@dataclass
class ModelConfig:
    name: str = None
    wrapper: str = None
    model_type: str = None
    path: str = None
    tokenizer_path: str = None
    trust_remote_code: bool = False

    # model architecture parameters
    num_layers: int = 0
    embed_size: int = 0
    block_size: int = 0
    num_heads: int = 0
    vocab_size: int = 0

@dataclass
class DataConfig:
    train_path: str = None
    val_path: str = None
    test_path: str = None

    # data processing parameters
    window_size: int = 2
    stride: int = 1
    batch_size : int = 1
    pad_left: bool = False
    drop_last: bool = False
    
    shuffle_dataset: bool = False
    shuffle_dataloader: bool = False

    data_process_batch_size: int = 1000
    data_process_num_proc: int = 4
    num_workers: int = 0
    
@dataclass
class OptimConfig:
    lr: float = 3e-4
    optimizer: str = "adam"
    lr_scheduler: Optional[str] = None
    lr_steps: Optional[list] = None
    gamma: float = 0.9
    weight_decay: float = 0.0
    max_epochs: int = 10
    grad_clip: Optional[float] = None

@dataclass
class LoggingConfig:
    log_dir: str = None
    interval_strategy: str = "epoch"
    log_interval: int = 10
    eval_interval: int = 10
    save_interval: int = 10
    
    return_logits: bool = False
    return_weights: bool = False
    return_embeddings: bool = False
    
    metric_for_best_model: str = None
    save_total_limit: int = 5
    load_best_model_at_end: bool = False
    save_model: bool = True

@dataclass
class EvaluationConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # generic metadata fields
    exp_name: str = "default_experiment"
    exp_dir: str = None
    device: str = "cpu"
    task: str = "causal_lm"
    bp_method: str = "continuous"
    layer_type: str = "all"
    embed_type: str = "embeddings"

    # helpers
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
@dataclass
class TrainingConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # generic metadata fields
    exp_name: str = "default_experiment"
    exp_dir: str = None
    num_runs: int = 1
    num_epochs: int = 10
    task: str = "causal_lm"
    bp_method: str = "bptt"
    seed: int = 42
    device: str = "cpu"

    # helpers
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)