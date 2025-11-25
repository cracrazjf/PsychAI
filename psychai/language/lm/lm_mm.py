import torch
import gc
from typing import Dict, Optional, List, Any
from ...nn_builder import from_pretrained, load_config, build_spec_from_config, Model, CausalLMWrapper
from ..tokenizer import print_tokenizer

try:
    from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
except Exception as e:
    raise ImportError("transformers is required. Install with extras: psychai[simple-nn]") from e


class LM_ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self, 
                   model_name: str, 
                   model_path: str = None, 
                   task: str = "causal_lm", 
                   custom: bool = False, 
                   *,
                   random_seed = None,
                   weight_init: Optional[Dict[str, Any]] = None,
                   tokenizer_path: Optional[str] = None, 
                   trust_remote_code: Optional[bool] = True, 
                   new_tokens: Optional[List[str]] = None, 
                   new_tokens_specials: Optional[Dict[str, str]] = None):

        self.free_memory()

        if custom:
            self.tokenizer, self.model = load_custom_model(model_name, 
                                                           model_path, 
                                                           tokenizer_path=tokenizer_path, 
                                                           task=task, 
                                                           random_seed=random_seed, 
                                                           weight_init=weight_init)
        else:
            self.tokenizer, self.model = load_hf_model(model_name, 
                                                       model_path, 
                                                       trust_remote_code, 
                                                       task, new_tokens, 
                                                       new_tokens_specials)

    def free_memory(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            try:
                del self.model
                print("Current model deleted")
            except Exception:
                pass
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            try:
                del self.tokenizer
                print("Current tokenizer deleted")
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()
        self.model = None
        self.model_path = None
        self.tokenizer = None
        self.model_name = None
        print("Cache cleared")

def load_custom_model(model_name: str,
                      model_path: str, 
                      task: str = "causal_lm", 
                      tokenizer_path: str = None,
                      weight_init: Optional[Dict[str, Any]] = None,
                      random_seed: int = None):
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # print_tokenizer(tokenizer)

    task_map = {
        "causal_lm": CausalLMWrapper
    }
    ctor = task_map.get(task)
    try:
        model = from_pretrained(model_path)
    except Exception as e:
        print(e)
        print(f"Model not found, rebuilding model from config")
        if random_seed is not None:
            torch.manual_seed(random_seed)
        config = load_config(model_path)
        model = build_spec_from_config(config)  
        model = Model(model)
    
        if weight_init is not None:
            models_params = dict(model.named_parameters())
            for param_name, init_info in weight_init.items():
                if param_name not in models_params:
                    raise ValueError(f"Parameter {param_name} not found in model parameters.")
                param = models_params[param_name]
                init_type = init_info[0]
                if init_type == "uniform":
                    limit = init_info[1]
                    torch.nn.init.uniform_(param, -limit, limit)
                    print(f"Initialized {param_name} with uniform distribution in [-{limit}, {limit}]")
                elif init_type == "normal":
                    mean = init_info[1]
                    std = init_info[2]
                    torch.nn.init.normal_(param, mean, std)
                    print(f"Initialized {param_name} with normal distribution (mean={mean}, std={std})")
                elif init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')
                    print(f"Initialized {param_name} with Kaiming uniform distribution")
                elif init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(param)
                    print(f"Initialized {param_name} with Xavier uniform distribution")
                else:
                    raise ValueError(f"Unsupported initialization type: {init_type} for parameter {param_name}")
            
    print(model.summary())
    model = ctor(model)
    print(f"Model wrapped with {ctor}")
    return tokenizer, model

def load_hf_model(model_name: str, 
                  model_path: str = None, 
                  task: str = "causal_lm",
                  trust_remote_code: bool = True, 
                  new_tokens: Optional[List[str]] = None, 
                  new_tokens_specials: Optional[Dict[str, str]] = None):

    # download model from hf using model_name if model_path is not provided
    if model_path is None:
        model_path = model_name

    print(f"Loading model and tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code, use_fast=True)
    if new_tokens is not None:
        tokenizer.add_tokens(new_tokens)
        if new_tokens_specials is not None:
            tokenizer.add_special_tokens(new_tokens_specials)
        model.resize_token_embeddings(len(tokenizer))
    print(f"Tokenizer loaded")
    print_tokenizer(tokenizer)

    task_map = {
        "causal_lm": AutoModelForCausalLM,
        "masked_lm": AutoModelForMaskedLM,
        "seq_cls": AutoModelForSequenceClassification,
        "tok_cls": AutoModelForTokenClassification,
    }
    ctor = task_map.get(task)
    if ctor is None:
        raise ValueError(f"Unknown task: {task}")

    model = ctor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    print(f"Model loaded")
    print_hf_model(model)
    print(f"Model wrapped with {ctor}")
    return tokenizer, model

def print_hf_model(model):
        cfg = model.config.to_dict()
        print("== Model ==")
        print(f"model_type        : {cfg.get('model_type')}")
        print(f"n_layer           : {cfg.get('n_layer') or cfg.get('num_hidden_layers')}")
        print(f"n_head            : {cfg.get('n_head')  or cfg.get('num_attention_heads')}")
        print(f"n_embd/hidden     : {cfg.get('n_embd')  or cfg.get('hidden_size')}")
        print(f"vocab_size        : {cfg.get('vocab_size')} (tokenizer: {len(tokenizer)})")
        print(f"max_position      : {cfg.get('n_positions') or cfg.get('max_position_embeddings')}")
        print(f"pad_token_id      : {cfg.get('pad_token_id')}  eos_token_id: {cfg.get('eos_token_id')}")
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("== Params ==")
        print(f"total/trainable   : {total:,} / {trainable:,}")
