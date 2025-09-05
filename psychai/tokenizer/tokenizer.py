from psychai.config.tokenizer import TokenizerConfig
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence as PreSeq
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import (
    BpeTrainer, UnigramTrainer, WordLevelTrainer
)
from typing import Iterable, List, Optional, Dict


def build_tokenizer(mode: str,
                    *,
                    vocab_size: int,
                    lowercase: bool=False,
                    strip_accents: bool=False,
                    split_punctuation: bool=False,
                    byte_level: bool=False,
                    special_tokens: Optional[List[str]]=None,
                    unk_token: Optional[str]=None,
                    bos_token: Optional[str]=None,
                    eos_token: Optional[str]=None,
                    pad_token: Optional[str]=None,
                    corpus: Optional[Iterable[str]] = None, 
                    vocab: Optional[Dict[str, int]] = None) -> PreTrainedTokenizerFast:
    # 1) Normalizer
    norm = []
    if lowercase: norm.append(Lowercase())
    if strip_accents: norm.append(StripAccents())
    normalizer = Sequence([NFD()] + norm) if norm else None

    # 2) Pre-tokenizer and model/trainer by mode
    mode = mode.lower()
    if mode == "word":
        # Word-level: each whitespace-delimited token is a piece
        tk = Tokenizer(WordLevel(vocab=vocab, unk_token=unk_token))
        if split_punctuation:
            tk.pre_tokenizer = PreSeq([Whitespace(), Punctuation()])
        else:
            tk.pre_tokenizer = Whitespace()
        if normalizer: tk.normalizer = normalizer
        
        if vocab is None:
            trainer = WordLevelTrainer(
            special_tokens=special_tokens,
            vocab_size=vocab_size,
            # Optionally ignore words shorter than min_frequency: min_frequency=2,
            )
            tk.train_from_iterator(corpus, trainer=trainer)

    elif mode == "char":
        # Character-level: train CharBPETokenizer with merges=0 (or use Unigram with tiny pieces)
        # Simpler: build a WordLevel over characters
        chars = set()
        for line in corpus:
            text = line.lower() if lowercase else line
            chars.update(list(text))
        # Ensure iterator not consumed: pass a list or re-iterate corpus outside if needed
        tk = Tokenizer(WordLevel(vocab=vocab, unk_token=unk_token))
        tk.pre_tokenizer = Whitespace()
        if normalizer: tk.normalizer = normalizer
        # Build vocab manually: each char is a token
        vocab = {c: i for i, c in enumerate(special_tokens + sorted(chars))}
        tk.model = WordLevel(vocab=vocab, unk_token=unk_token)

    elif mode == "phoneme":
        # Convert to phoneme strings first
        def ph_iter():
            for t in corpus:
                yield to_phonemes(t)
        tk = Tokenizer(Unigram())
        tk.pre_tokenizer = Whitespace()
        if normalizer: tk.normalizer = normalizer
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            unk_token=unk_token,
        )
        tk.train_from_iterator(ph_iter(), trainer=trainer)

    elif mode == "bpe":
        tk = Tokenizer(BPE(unk_token=unk_token))
        if byte_level:
            tk.pre_tokenizer = ByteLevel()
        else:
            if split_punctuation:
                tk.pre_tokenizer = PreSeq([Whitespace(), Punctuation()])
            else:
                tk.pre_tokenizer = Whitespace()
        if normalizer and not byte_level:
            tk.normalizer = normalizer  # GPT-2 byte-level typically skips extra normalization
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=True,
        )
        tk.train_from_iterator(corpus, trainer=trainer)

    elif mode == "unigram":
        tk = Tokenizer(Unigram())
        if byte_level:
            tk.pre_tokenizer = ByteLevel()
        else:
            if split_punctuation:
                tk.pre_tokenizer = PreSeq([Whitespace(), Punctuation()])
            else:
                tk.pre_tokenizer = Whitespace()
        if normalizer and not byte_level:
            tk.normalizer = normalizer
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            unk_token=unk_token,
            show_progress=True,
        )
        tk.train_from_iterator(corpus, trainer=trainer)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 3) Wrap as a HF fast tokenizer with special tokens wired up
    fast = PreTrainedTokenizerFast(tokenizer_object=tk)
    if pad_token: fast.add_special_tokens({"pad_token": pad_token})
    fast.add_special_tokens({"unk_token": unk_token})
    if bos_token: fast.add_special_tokens({"bos_token": bos_token})
    if eos_token: fast.add_special_tokens({"eos_token": eos_token})
    return fast

def print_tokenizer(tokenizer):
    v2i = tokenizer.get_vocab()
    print("Vocab size:", len(v2i))

    # print in id order
    for tok, idx in sorted(v2i.items(), key=lambda kv: kv[1]):
        print(f"{idx:5d}  {tok}")

    # special tokens
    print("\nSpecial tokens map:", tokenizer.special_tokens_map)        # names -> tokens
    print("All special tokens:", tokenizer.all_special_tokens)          # list of tokens
    print("All special ids:", tokenizer.all_special_ids)                # list of ids
    print("BOS/EOS/PAD/UNK:",
        tokenizer.bos_token, tokenizer.bos_token_id,
        tokenizer.eos_token, tokenizer.eos_token_id,
        tokenizer.pad_token, tokenizer.pad_token_id,
        tokenizer.unk_token, tokenizer.unk_token_id)

    # tokens you added after init (if any)
    print("Added vocab:", tokenizer.get_added_vocab())  