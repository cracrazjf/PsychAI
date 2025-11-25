from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

@dataclass
class ModelConfig:
    name: str = None
    path: str = None
    random_seed: Optional[int] = None
    task: str = "causal_lm"
    customized_model: bool = False
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = False
    weight_init: Optional[dict] = None

@dataclass
class DataConfig:
    train_path: str = None
    val_path: str = None
    test_path: str = None
    data_process_batch_size: int = 1000
    data_process_num_proc: int = 4
    num_workers: int = 0
    shuffle_dataset: bool = False
    shuffle_dataloader: bool = False
    stride: int = None
    pad_left: bool = False
    drop_last: bool = False
    batch_size : int = 1
    sequence_length: int = 2

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
    log_interval: int = 100
    eval_interval: int = 1
    save_every_epochs: int = 1
    metric_for_best_model: str = None
    save_total_limit: int = 5
    load_best_model_at_end: bool = False
    save_model: bool = True

@dataclass
class EvaluationConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # generic metadata fields
    experiment_name: str = "default_experiment"
    experiment_directory: str = None
    training_method: str = "continuous"


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
    experiment_name: str = "default_experiment"
    experiment_directory: str = None
    num_runs: int = 1
    num_epochs: int = 10
    seed: int = 42
    training_method: str = "continuous"


    # helpers
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)