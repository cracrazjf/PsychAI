from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TokenizerConfig:
    mode: str                    # 'word' | 'char' | 'phoneme' | 'bpe' | 'unigram'
    language: str = "en-us"      # only used for phoneme mode
    vocab_size: int = 8000       # ignored for char/word if you want full vocab
    lowercase: bool = True
    strip_accents: bool = True
    split_punctuation: bool = False
    byte_level: bool = False     # for bpe/unigram; True gives GPT-2 style byte-level
    special_tokens: List[str] = None  # e.g., ["<pad>","<unk>","<bos>","<eos>"]
    unk_token: str = "<unk>"
    bos_token: Optional[str] = "<bos>"
    eos_token: Optional[str] = "<eos>"
    pad_token: Optional[str] = "<pad>"

    def __init__(self, **overrides):
        print(overrides)
        for k, v in overrides.items():
            setattr(self, k, v)

    def __post_init__(self):
        if self.special_tokens is None:
            toks = []
            if self.pad_token: toks.append(self.pad_token)
            toks.append(self.unk_token)
            if self.bos_token: toks.append(self.bos_token)
            if self.eos_token: toks.append(self.eos_token)
            self.special_tokens = toks
