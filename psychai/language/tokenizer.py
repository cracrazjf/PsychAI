from tokenizers import pre_tokenizers as pre
from tokenizers import normalizers as norm
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram, WordLevel
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer
from tokenizers import AddedToken
from tokenizers import decoders
from transformers import PreTrainedTokenizerFast
from typing import Dict, List, Tuple, Iterable, Optional


def make_normalizer(
    *,
    preset: str | None = None,        # "bert" or "nmt" or None
    unicode_form: str | None = "NFKC",# "NFD","NFKD","NFC","NFKC", or None
    lowercase: bool = False,
    strip_accents: bool = False,
    strip: bool = False,
    strip_left: bool = False,
    strip_right: bool = False,
    replaces: list[tuple[str, str]] | None = None,
    bert_handle_chinese_chars: bool = False,
    bert_clean_text: bool = False 
):
    steps = []

    # 1) Presets (optional)
    if preset is not None:
        p = preset.lower()
        if p == "bert":
            steps.append(
                norm.BertNormalizer(
                    lowercase=lowercase,
                    strip_accents=strip_accents,
                    clean_text=bert_clean_text,
                    handle_chinese_chars=bert_handle_chinese_chars,
                )
            )
            # When using BertNormalizer, you typically don’t need separate lowercase/strip_accents below
            lowercase = False
            strip_accents = False
        elif p == "nmt":
            steps.append(norm.Nmt())
        else:
            raise ValueError(f"Unknown preset: {preset!r}. Use 'bert', 'nmt', or None.")

    # 2) Unicode normalization (optional)
    if unicode_form:
        uf = unicode_form.upper()
        form_map = {
            "NFD": norm.NFD(),
            "NFKD": norm.NFKD(),
            "NFC": norm.NFC(),
            "NFKC": norm.NFKC(),
        }
        if uf not in form_map:
            raise ValueError(f"Unknown unicode_form: {unicode_form!r}")
        steps.append(form_map[uf])

    # 3) Simple toggles
    if lowercase:
        steps.append(norm.Lowercase())
    if strip_accents:
        steps.append(norm.StripAccents())
    if strip:
        steps.append(norm.Strip(left=strip_left, right=strip_right))

    # 4) Multiple Replace rules
    if replaces:
        for (pattern, content) in replaces:
            steps.append(norm.Replace(pattern, content))

    # Return a single step or a composed sequence
    if not steps:
        return norm.Sequence([])  # no-op
    if len(steps) == 1:
        return steps[0]
    return norm.Sequence(steps)

def make_pretokenizer(
    *,
    use_whitespace: bool = False,
    use_punctuation: bool = False,
    use_byte_level: bool = False,
    use_metaspace: bool = False,
    metaspace_replacement: str = "▁",
    metaspace_add_prefix_space: bool = False,
    split_digits: bool = False,
    # --- custom rules ---
    split_underscores: bool = False,
    split_hyphens: bool = False,
    split_camelcase: bool = False,
    split_hashtags: bool = False,
    split_emojis: bool = False,
    split_num_units: bool = False,
    custom_splits: list | None = None,  # extra regexes [(pattern, behavior), ...]
):
    steps = []

    # --- standard ---
    if use_whitespace:
        steps.append(pre.Whitespace())
    if use_punctuation:
        steps.append(pre.Punctuation())
    if use_byte_level:
        steps.append(pre.ByteLevel())
    if use_metaspace:
        steps.append(pre.Metaspace(
            replacement=metaspace_replacement,
            # add_prefix_space=metaspace_add_prefix_space
        ))
    if split_digits:
        steps.append(pre.Digits(individual_digits=True))

    # --- custom rules ---
    if split_underscores:
        steps.append(pre.Split("_", behavior="isolated"))
    if split_hyphens:
        steps.append(pre.Split("-", behavior="isolated"))
    if split_camelcase:
        steps.append(pre.Split(r"(?<=[a-z])(?=[A-Z])", behavior="isolated", regex=True))
    if split_hashtags:
        steps.append(pre.Split(r"#\w+", behavior="isolated", regex=True))
    if split_emojis:
        steps.append(pre.Split(r"[\U0001F600-\U0001F64F]", behavior="isolated", regex=True))  
    if split_num_units:
        steps.append(pre.Split(r"(\d+)([a-zA-Z]+)", behavior="isolated", regex=True))

    # --- user extra rules ---
    if custom_splits:
        for pat, behavior in custom_splits:
            steps.append(pre.Split(pat, behavior=behavior))

    # return composite
    if not steps:
        return pre.Whitespace()  # default
    if len(steps) == 1:
        return steps[0]
    return pre.Sequence(steps)

def train_tokenizer(
    *,
    model_type: str = "bpe", 
    files: list[str],                   
    vocab_size: int = 999999999,
    min_frequency: int = 1,
    normalizer: Optional[norm.Normalizer] = None,
    pretokenizer: Optional[pre.PreTokenizer] = None,
    decoder_type: str = "metaspace",
    decoder_kwargs: dict = {},
    special_tokens: list[str] = ["<unk>", "<pad>"],
    protected_terms: list[str] = []
):
    if model_type == "bpe":
        model = BPE(unk_token="<unk>")
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(special_tokens),
            show_progress=True,
        )
    elif model_type == "wordpiece":
        model = WordPiece(unk_token="<unk>")
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(special_tokens),
        )
    elif model_type == "unigram":
        model = Unigram()
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=list(special_tokens),
            show_progress=True,
        )
    elif model_type == "wordlevel":
        model = WordLevel(unk_token="<unk>")
        trainer = WordLevelTrainer(
            vocab_size=vocab_size,
            special_tokens=list(special_tokens),
            min_frequency=min_frequency,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    tok = Tokenizer(model)
    if normalizer is not None:
        tok.normalizer = normalizer
    if pretokenizer is not None:
        tok.pre_tokenizer = pretokenizer

    added = [
        AddedToken(t, single_word=True, lstrip=False, rstrip=False, normalized=True)
        for t in protected_terms
    ]
    tok.add_tokens(added)  # adds them as regular (non-special) tokens

    tok.train(files, trainer)

    if decoder_type == "metaspace":
        replacement = decoder_kwargs.get("replacement", "▁")
        tok.decoder = decoders.Metaspace(replacement=replacement)
    elif decoder_type == "bytes":
        tok.decoder = decoders.Bytes()
    elif decoder_type == "wordpiece":
        prefix = decoder_kwargs.get("prefix", "##")
        tok.decoder = decoders.WordPiece(prefix=prefix)
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")
    
    return tok

def create_custom_tokenizer(vocab: Dict[str, int], pretokenizer: Optional[pre.PreTokenizer] = None):
    model = WordLevel(vocab=vocab, unk_token="<unk>")
    tok = Tokenizer(model)
    if pretokenizer is not None:
        tok.pre_tokenizer = pretokenizer
    return tok

def wrap_tokenizer(
    tokenizer: Tokenizer, 
    unk_token="<unk>", 
    pad_token="<pad>", 
    bos_token=None, 
    eos_token=None, 
    sep_token=None, 
    cls_token=None
):
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=unk_token,
        pad_token=pad_token,
        bos_token=bos_token,
        eos_token=eos_token,
        sep_token=sep_token,
        cls_token=cls_token,
    )
    return fast_tok

def print_tokenizer(tokenizer):
    v2i = tokenizer.get_vocab()
    print("Vocab size:", len(v2i))

    # print in id order
    # for tok, idx in sorted(v2i.items(), key=lambda kv: kv[1]):
    #     print(f"{idx:5d}  {tok}")

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