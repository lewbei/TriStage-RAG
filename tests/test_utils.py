"""Tests for shared utilities."""
import os
import pytest
from unittest.mock import patch
from tristage_rag.utils import resolve_model_path, get_device, clear_gpu_cache, BM25_TOKENIZER


class TestResolveModelPath:
    def test_prefers_flat_local_dir(self, tmp_path):
        model_name = "google/embeddinggemma-300m"
        flat_dir = tmp_path / "embeddinggemma-300m"
        flat_dir.mkdir()
        result = resolve_model_path(model_name, str(tmp_path))
        assert result == str(flat_dir)

    def test_falls_back_to_legacy_dir(self, tmp_path):
        model_name = "google/embeddinggemma-300m"
        legacy_dir = tmp_path / "google" / "embeddinggemma-300m"
        legacy_dir.mkdir(parents=True)
        result = resolve_model_path(model_name, str(tmp_path))
        assert result == str(legacy_dir)

    def test_returns_original_when_no_local(self, tmp_path):
        model_name = "google/embeddinggemma-300m"
        result = resolve_model_path(model_name, str(tmp_path))
        assert result == model_name


class TestGetDevice:
    def test_auto_returns_cpu_when_no_cuda(self):
        with patch("tristage_rag.utils.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            assert get_device("auto") == "cpu"

    def test_explicit_cpu(self):
        assert get_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        assert get_device("cuda") == "cuda"


class TestClearGpuCache:
    def test_noop_on_cpu(self):
        clear_gpu_cache("cpu")  # should not raise

    def test_calls_empty_cache_on_cuda(self):
        with patch("tristage_rag.utils.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            clear_gpu_cache("cuda")
            mock_torch.cuda.empty_cache.assert_called_once()


class TestBM25Tokenizer:
    def test_removes_special_chars(self):
        assert BM25_TOKENIZER.sub(" ", "hello, world!") == "hello  world "

    def test_keeps_alphanumeric(self):
        assert BM25_TOKENIZER.sub(" ", "test123 foo-bar") == "test123 foo bar"

    def test_preserves_lowercase_letters(self):
        assert BM25_TOKENIZER.sub(" ", "abc 123") == "abc 123"
