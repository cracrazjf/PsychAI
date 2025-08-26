from typing import Dict, Optional
try:
    from transformers import (
    AutoTokenizer, AutoConfig,
    AutoModelForCausalLM,          # next-token LM
    AutoModelForMaskedLM,          # BERT-style
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
except Exception as e:
    raise ImportError("transformers is required. Install with extras: psychai[simple-nn]") from e


class ModelManager:
    def __init__(self, model_name: str, trust_remote_code: bool = True, task_type: str = "causal_lm"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.trust_remote_code = trust_remote_code
        self.task_type = task_type

    def load_model(self, new_tokens: Optional[List[str]] = None, new_tokens_specials: Optional[Dict[str, str]] = None):
        self.free_memory()
        if self.task_type != "custom":
            pass
        else:
            self.load_model_from_hf(new_tokens, new_tokens_specials)
            
    def add_new_tokens(self, new_tokens: List[str], specials: Optional[Dict[str, str]] = None):
        added = self.tokenizer.add_tokens(new_tokens)
        if specials is not None:
            added_specials = self.tokenizer.add_special_tokens(specials)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def summarize_hf_lm(self):
        cfg = self.model.config.to_dict()
        print("== Model ==")
        print(f"model_type        : {cfg.get('model_type')}")
        print(f"n_layer           : {cfg.get('n_layer') or cfg.get('num_hidden_layers')}")
        print(f"n_head            : {cfg.get('n_head')  or cfg.get('num_attention_heads')}")
        print(f"n_embd/hidden     : {cfg.get('n_embd')  or cfg.get('hidden_size')}")
        print(f"vocab_size        : {cfg.get('vocab_size')} (tokenizer: {len(self.tokenizer)})")
        print(f"max_position      : {cfg.get('n_positions') or cfg.get('max_position_embeddings')}")
        print(f"pad_token_id      : {cfg.get('pad_token_id')}  eos_token_id: {cfg.get('eos_token_id')}")
        print("== Tokens ==")
        print(f"specials          : {self.tokenizer.special_tokens_map}")
        print(f"added_tokens      : {len(self.tokenizer.get_added_vocab())}")
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("== Params ==")
        print(f"total/trainable   : {total:,} / {trainable:,}")

    def load_model_from_hf(self, new_tokens: Optional[List[str]] = None, new_tokens_specials: Optional[Dict[str, str]] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code, use_fast=True)

        task_map = {
            "causal_lm": AutoModelForCausalLM,
            "masked_lm": AutoModelForMaskedLM,
            "seq_cls": AutoModelForSequenceClassification,
            "tok_cls": AutoModelForTokenClassification,
        }
        ctor = task_map.get(self.task_type)
        if ctor is None:
            raise ValueError(f"Unknown task: {self.task_type}")

        self.model = ctor.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        if new_tokens is not None:
            if new_tokens_specials is not None:
                self.add_new_tokens(new_tokens, new_tokens_specials)
            else:
                self.add_new_tokens(new_tokens)

        self.summarize_hf_lm()

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
        self.model_name = None
        print("✅ Cache cleared")