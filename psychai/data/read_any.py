from __future__ import annotations

import csv
import io
import json
import numpy as np
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, TextIO

try:
    import zstandard as zstd  # type: ignore
except Exception:  # pragma: no cover
    zstd = None  # type: ignore

__all__ = [
    "read_csv",
    "read_jsonl",
    "read_json",
    "infer_file_type",
]

def _open_text(path: str | Path, *, encoding: str = "utf-8", errors: str = "ignore") -> TextIO:
    p = Path(path)
    suffixes = "".join(p.suffixes[-2:])  # handle .jsonl.gz, .csv.zst, etc.

    if suffixes.endswith(".gz"):
        import gzip
        return io.TextIOWrapper(gzip.open(p, "rb"), encoding=encoding, errors=errors)

    if suffixes.endswith(".zst"):
        if zstd is None:
            raise RuntimeError("zstandard not installed; cannot read .zst files. Try `pip install zstandard`.")
        fh = p.open("rb")
        dctx = zstd.ZstdDecompressor()  # type: ignore
        return io.TextIOWrapper(dctx.stream_reader(fh), encoding=encoding, errors=errors)

    # Plain text
    return p.open("r", encoding=encoding, errors=errors)


def _default_delimiter_for(path: str | Path, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    p = Path(path)
    if p.suffix == ".tsv" or "".join(p.suffixes[-2:]).endswith(".tsv.gz") or "".join(p.suffixes[-2:]).endswith(".tsv.zst"):
        return "\t"
    return ","

def infer_file_type(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return "csv"
    elif ext == ".jsonl":
        return "jsonl"
    elif ext == ".json":
        return "json"
    else:
        raise ValueError(f"Invalid file type: {ext}")

def read_csv(
    path: str | Path,
    *,
    delimiter: Optional[str] = None,
    encoding: str = "utf-8",
    errors: str = "ignore",
    skip_blank: bool = True,
) -> Iterator[Dict]:
    delim = _default_delimiter_for(path, delimiter)
    with _open_text(path, encoding=encoding, errors=errors) as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            if skip_blank and not any(v and str(v).strip() for v in row.values()):
                continue
            yield dict(row)


def read_jsonl(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    errors: str = "ignore",
    skip_blank: bool = True,
) -> Iterator[Dict]:
    with _open_text(path, encoding=encoding, errors=errors) as f:
        for i, line in enumerate(f):
            if skip_blank and not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:  # pragma: no cover
                raise ValueError(f"Invalid JSON on line {i+1} of {path}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Line {i+1} of {path} is not a JSON object (got {type(obj).__name__}).")
            yield obj


def read_json(
    path: str | Path,
    *,
    encoding: str = "utf-8",
    errors: str = "ignore",
) -> Iterator[Dict]:
    with _open_text(path, encoding=encoding, errors=errors) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:  # pragma: no cover
            raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if isinstance(data, list):
        for i, obj in enumerate(data):
            if not isinstance(obj, dict):
                raise ValueError(f"Element {i} in {path} is not a JSON object (got {type(obj).__name__}).")
            yield obj
    elif isinstance(data, dict):
        yield data
    else:
        raise ValueError(f"Top-level value in {path} must be an object or array of objects (got {type(data).__name__}).")

def read_npz(
    path: str | Path,
    *,
    allow_pickle: bool = False,
) -> Dict:
    with np.load(path, allow_pickle=allow_pickle) as z:
        print(f"Contents of {path}: {z.files}")
        return {k: z[k] for k in z.files}