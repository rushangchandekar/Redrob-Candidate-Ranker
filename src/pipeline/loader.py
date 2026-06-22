"""
Data Loader — Streaming loader for candidate JSONL files.
Supports both .jsonl (plain) and .jsonl.gz (gzipped) formats.
"""

import gzip
import json
import os
from typing import Iterator


def stream_candidates(path: str) -> Iterator[dict]:
    """
    Stream candidates one by one from a JSONL or JSONL.gz file.
    Memory-efficient: yields one candidate dict at a time.
    """
    if path.endswith(".gz"):
        opener = lambda p: gzip.open(p, "rt", encoding="utf-8")
    else:
        opener = lambda p: open(p, "r", encoding="utf-8")

    with opener(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_all_candidates(path: str) -> list[dict]:
    """
    Load all candidates into memory.
    Use only when you have enough RAM (~500 MB for 100k candidates).
    """
    return list(stream_candidates(path))


def load_sample_candidates(path: str) -> list[dict]:
    """
    Load sample candidates from a JSON array file.
    Used for quick testing with sample_candidates.json.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_candidates(path: str) -> int:
    """Count total candidates without loading all into memory."""
    count = 0
    for _ in stream_candidates(path):
        count += 1
    return count


def detect_dataset_path(base_dir: str) -> str:
    """
    Auto-detect the dataset file in the given directory.
    Checks for candidates.jsonl.gz first, then candidates.jsonl.
    """
    gz_path = os.path.join(base_dir, "candidates.jsonl.gz")
    plain_path = os.path.join(base_dir, "candidates.jsonl")
    sample_path = os.path.join(base_dir, "sample_candidates.json")

    if os.path.exists(gz_path):
        return gz_path
    elif os.path.exists(plain_path):
        return plain_path
    elif os.path.exists(sample_path):
        return sample_path
    else:
        raise FileNotFoundError(
            f"No candidate dataset found in {base_dir}. "
            "Expected candidates.jsonl.gz, candidates.jsonl, or sample_candidates.json"
        )
