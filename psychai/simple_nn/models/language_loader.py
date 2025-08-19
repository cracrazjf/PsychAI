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

def add_new_tokens(model, tokenizer, new_tokens, specials: Optional[Dict[str, str]] = None):
    """
    Add new tokens to the tokenizer.
    """
    added = tokenizer.add_tokens(new_tokens)
    if specials is not None:
        added_specials = tokenizer.add_special_tokens(specials)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def summarize_hf_lm(model, tokenizer):
    cfg = model.config.to_dict()
    print("== Model ==")
    print(f"model_type        : {cfg.get('model_type')}")
    print(f"n_layer           : {cfg.get('n_layer') or cfg.get('num_hidden_layers')}")
    print(f"n_head            : {cfg.get('n_head')  or cfg.get('num_attention_heads')}")
    print(f"n_embd/hidden     : {cfg.get('n_embd')  or cfg.get('hidden_size')}")
    print(f"vocab_size        : {cfg.get('vocab_size')} (tokenizer: {len(tokenizer)})")
    print(f"max_position      : {cfg.get('n_positions') or cfg.get('max_position_embeddings')}")
    print(f"pad_token_id      : {cfg.get('pad_token_id')}  eos_token_id: {cfg.get('eos_token_id')}")
    print("== Tokens ==")
    print(f"specials          : {tokenizer.special_tokens_map}")
    print(f"added_tokens      : {len(tokenizer.get_added_vocab())}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("== Params ==")
    print(f"total/trainable   : {total:,} / {trainable:,}")

def load_pretrained_hf_language(config):
    model_name = config.MODEL_NAME
    trust_remote_code = config.TRUST_REMOTE_CODE
    task_type = config.TASK_TYPE

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=True)

    task_map = {
        "causal_lm": AutoModelForCausalLM,
        "masked_lm": AutoModelForMaskedLM,
        "seq_cls": AutoModelForSequenceClassification,
        "tok_cls": AutoModelForTokenClassification,
    }
    ctor = task_map.get(task_type)
    if ctor is None:
        raise ValueError(f"Unknown task: {task_type}")

    model = ctor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if config.NEW_TOKENS is not None:
        if config.NEW_TOKENS_SPECIALS is not None:
            model, tokenizer = add_new_tokens(model, tokenizer, config.NEW_TOKENS, config.NEW_TOKENS_SPECIALS)
        else:
            model, tokenizer = add_new_tokens(model, tokenizer, config.NEW_TOKENS)

    summarize_hf_lm(model, tokenizer)
    return model, tokenizer