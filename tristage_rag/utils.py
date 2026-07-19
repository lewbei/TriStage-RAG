"""Shared utilities for TriStage-RAG pipeline stages."""
import os
import re
import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)

# Pre-compiled regex for BM25 tokenization
BM25_TOKENIZER = re.compile(r'[^a-z0-9\s]')


def resolve_model_path(model_name: str, cache_dir: str) -> str:
    """Resolve model path, preferring local cached copies.

    Checks <cache_dir>/<basename> first, then <cache_dir>/<full_name>,
    falling back to the original model_name for remote download.
    """
    base_dir = os.path.join(cache_dir, os.path.basename(model_name))
    legacy_dir = os.path.join(cache_dir, model_name)
    if os.path.isdir(base_dir):
        return base_dir
    if os.path.isdir(legacy_dir):
        return legacy_dir
    return model_name


def get_device(device: str = "auto", use_gpu: bool = True) -> str:
    """Determine the best available device for inference."""
    if device == "auto":
        if torch.cuda.is_available() and use_gpu:
            return "cuda"
        return "cpu"
    return device


def clear_gpu_cache(device: str) -> None:
    """Clear CUDA cache if on GPU. Safe to call on CPU (no-op)."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
