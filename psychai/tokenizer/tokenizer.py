from tokenizers import pre_tokenizers as pre
from tokenizers import normalizers as norm
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram, WordLevel
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer
from tokenizers import AddedToken
from transformers import PreTrainedTokenizerFast
import ftfy
import re
from typing import Dict, List, Tuple, Iterable, Optional

# --- Core regexes ---
RE_URL      = re.compile(r"https?://\S+")
RE_EMAIL    = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_NUM      = re.compile(r"\b\d[\d,]*\.?\d*\b")
RE_MONEY    = re.compile(r"(?<!\w)(?:\$|€|£|¥|₹)\s?\d[\d,]*\.?\d*(?!\w)")
RE_PERCENT  = re.compile(r"\b\d+(\.\d+)?\s?%\b")
RE_DATE     = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")
RE_TIME     = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?:\s?[APap][Mm])?\b")
# Social
RE_USER     = re.compile(r"@[A-Za-z0-9_]{1,30}")
RE_HASHTAG  = re.compile(r"#[\w]+")
RE_EMOJI    = re.compile(r"[\U0001F300-\U0001FAFF]")  # broad emoji range
RE_LAUGH    = re.compile(r"(?:ha|he|hi|ho|ha)+|l+o+l+", re.IGNORECASE)
RE_EMPH_P   = re.compile(r"([!?\.])\1{2,}")           # !!!, ???, ...
RE_EMPH_CH  = re.compile(r"([A-Za-z])\1{2,}")         # sooo, cooool
# Biomedical
RE_GENE     = re.compile(r"\b[A-Z]{2,6}\d+\b")        # e.g., IL2, BRCA1
RE_UNIT     = re.compile(r"\b(?:mg|kg|ml|mm|cm|km|μg|ng|mM|μM|nM|%)\b", re.IGNORECASE)
# Code
RE_STRING   = re.compile(r"(\".*?\"|'.*?')")          # naive (no escapes)
RE_OPERATOR = re.compile(r"([=+\-/*%<>!&|^]=|==|!=|<=|>=|//|\*\*|->|::)")

RULES: List[Tuple[str, re.Pattern, str]] = [
    ("URL",     RE_URL,     "<URL>"),
    ("EMAIL",   RE_EMAIL,   "<EMAIL>"),
    ("NUM",     RE_NUM,     "<NUM>"),
    ("MONEY",   RE_MONEY,   "<MONEY>"),
    ("PERCENT", RE_PERCENT, "<PERCENT>"),
    ("DATE",    RE_DATE,    "<DATE>"),
    ("TIME",    RE_TIME,    "<TIME>"),
    ("USER",    RE_USER,    "<USER>"),
    ("HASHTAG", RE_HASHTAG, "<HASHTAG>"),
    ("EMOJI",   RE_EMOJI,   "<EMOJI>"),
    ("LAUGH",   RE_LAUGH,   "<LAUGH>"),
    ("EMPH",    RE_EMPH_P,  "<EMPH>"),
    ("EMPH",    RE_EMPH_CH, "<EMPH>"),
    ("GENE",    RE_GENE,    "<GENE>"),
    ("UNIT",    RE_UNIT,    "<UNIT>"),
    ("STRING",  RE_STRING,  "<STR>"),
    ("OPERATOR", RE_OPERATOR, "<OP>"),
]

# --- Preset definitions ---
PRESETS: Dict[str, List[Tuple[str, re.Pattern, str]]] = {
    "general": [
        ("MONEY",   RE_MONEY,   "<MONEY>"),
        ("PERCENT", RE_PERCENT, "<PERCENT>"),
        ("DATE",    RE_DATE,    "<DATE>"),
        ("TIME",    RE_TIME,    "<TIME>"),
        ("NUM",     RE_NUM,     "<NUM>"),
        ("URL",     RE_URL,     "<URL>"),
        ("EMAIL",   RE_EMAIL,   "<EMAIL>"),
    ],
    "social": [
        ("URL",     RE_URL,     "<URL>"),
        ("EMAIL",   RE_EMAIL,   "<EMAIL>"),
        ("USER",    RE_USER,    "<USER>"),
        ("HASHTAG", RE_HASHTAG, "<HASHTAG>"),
        ("EMOJI",   RE_EMOJI,   "<EMOJI>"),
        ("LAUGH",   RE_LAUGH,   "<LAUGH>"),
        ("EMPH",    RE_EMPH_P,  "<EMPH>"),
        ("EMPH",    RE_EMPH_CH, "<EMPH>"),
        ("NUM",     RE_NUM,     "<NUM>"),
    ],
    "biomedical": [
        ("MONEY",   RE_MONEY,   "<MONEY>"),
        ("PERCENT", RE_PERCENT, "<PERCENT>"),
        ("DATE",    RE_DATE,    "<DATE>"),
        ("TIME",    RE_TIME,    "<TIME>"),
        ("UNIT",    RE_UNIT,    "<UNIT>"),
        ("GENE",    RE_GENE,    "<GENE>"),
        ("NUM",     RE_NUM,     "<NUM>"),
        ("URL",     RE_URL,     "<URL>"),
        ("EMAIL",   RE_EMAIL,   "<EMAIL>"),
    ],
    "code": [
        ("STRING",  RE_STRING,  "<STR>"),
        ("URL",     RE_URL,     "<URL>"),
        ("NUM",     RE_NUM,     "<NUM>"),
        ("OP",      RE_OPERATOR,"<OP>"),
        ("EMAIL",   RE_EMAIL,   "<EMAIL>"),
    ],
}

def apply_placeholders(
    text: str,
    *,
    rules: Optional[str] = None,
    preset: Optional[str] = None,
    reversible: bool = False,
    extra_rules: Optional[Iterable[Tuple[str, re.Pattern, str]]] = None,
    protected_terms: Optional[Iterable[str]] = None,
) -> Tuple[str, List[Tuple[str, str]]]:
    
    rules = []
    if preset is None:
        if rules is not None:
            for rule in rules:
                if rule not in RULES:
                    raise ValueError(f"Unknown rule: {rule}. Choose from {list(RULES)}")
                else:
                    rules.append(RULES[rule])
    else:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(PRESETS)}")
        rules = list(PRESETS[preset])

    if extra_rules:
        rules.extend(extra_rules)

    mapping: List[Tuple[str, str]] = []
    counters: Dict[str, int] = {}

    # Protect exact surface forms (skip replacement inside them)
    protected_terms = set(protected_terms or [])
    # Quick mask for protected terms to avoid accidental replacement inside them
    masks: List[Tuple[str, str]] = []
    for i, term in enumerate(sorted(protected_terms, key=len, reverse=True), 1):
        placeholder = f"<__PROTECT_{i}__>"
        if term in text:
            text = text.replace(term, placeholder)
            masks.append((placeholder, term))

    # Apply rules in order
    for tag, pattern, ph in rules:
        if reversible:
            def sub_fn(m):
                counters[tag] = counters.get(tag, 0) + 1
                indexed = f"<{tag}_{counters[tag]}>"
                mapping.append((indexed, m.group(0)))
                return indexed
            text = pattern.sub(sub_fn, text)
        else:
            text = pattern.sub(ph, text)

    # Unmask protected terms
    for placeholder, term in masks:
        text = text.replace(placeholder, term)

    return text, mapping


def restore_placeholders(text: str, mapping: List[Tuple[str, str]]) -> str:
    """Reverse placeholders using the mapping from apply_placeholders(..., reversible=True)."""
    for ph, original in mapping:
        text = text.replace(ph, original)
    return text


def make_normalizer(
    *,
    preset: str | None = None,        # "bert" or "nmt" or None
    unicode_form: str | None = "NFKC",# "NFD","NFKD","NFC","NFKC", or None
    lowercase: bool = False,
    strip_accents: bool = False,
    strip: bool = True,
    strip_left: bool = True,
    strip_right: bool = True,
    replaces: list[tuple[str, str]] | None = None,  # [(pattern, content), ...]
    bert_handle_chinese_chars: bool = True,         # only if preset="bert"
    bert_clean_text: bool = True                    # only if preset="bert"
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
    metaspace_add_prefix_space: bool = True,
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
            add_prefix_space=metaspace_add_prefix_space
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
        # 😀–🙏 range as an example; can be extended
    if split_num_units:
        steps.append(pre.Split(r"(\d+)([a-zA-Z]+)", behavior="isolated", regex=True))

    # --- user extra rules ---
    if custom_splits:
        for pat, behavior in custom_splits:
            steps.append(pre.Split(pat, behavior=behavior, regex=True))

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
    vocab_size: int = 32000,
    min_frequency: int = 2,
    normalizer: Optional[norm.Normalizer] = None,
    pretokenizer: Optional[pre.PreTokenizer] = None,
    special_tokens: list[str] = ["<unk>", "<pad>", "<bos>", "<eos>"],
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