"""Tests for Stage 3 cross-encoder reranker."""
import pytest
from unittest.mock import patch, MagicMock
from tristage_rag.stage3_reranker import CrossEncoderReranker, AdaptiveCrossEncoderReranker, Stage3Config


class TestCrossEncoderReranker:
    def test_normalize_scores(self):
        config = Stage3Config(device="cpu", use_fp16=False)
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.config = config

        scores = reranker._normalize_scores([0.5, 1.0, 0.0])
        assert scores[0] == pytest.approx(0.5, abs=0.01)
        assert scores[1] == pytest.approx(1.0, abs=0.01)
        assert scores[2] == pytest.approx(0.0, abs=0.01)

    def test_normalize_empty(self):
        config = Stage3Config(device="cpu", use_fp16=False)
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.config = config
        assert reranker._normalize_scores([]) == []

    def test_normalize_all_same(self):
        config = Stage3Config(device="cpu", use_fp16=False)
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.config = config
        scores = reranker._normalize_scores([5.0, 5.0, 5.0])
        assert all(s == 0.0 for s in scores)

    def test_rerank(self, sample_docs, sample_query):
        config = Stage3Config(
            model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
            device="cpu",
            cache_dir="./models",
            top_k_final=3,
            batch_size=2,
            use_fp16=False,
        )
        reranker = CrossEncoderReranker(config)

        candidates = [
            {"doc_id": i, "document": doc, "score": 0.5, "stage2_score": 0.5}
            for i, doc in enumerate(sample_docs[:4])
        ]
        results = reranker.rerank(sample_query, candidates)
        assert len(results) <= 3
        assert all("stage3_score" in r for r in results)


class TestAdaptiveCrossEncoderReranker:
    def test_adaptive_batch_size_short_docs(self):
        config = Stage3Config(device="cpu", use_fp16=False, batch_size=32)
        reranker = AdaptiveCrossEncoderReranker.__new__(AdaptiveCrossEncoderReranker)
        reranker.config = config
        reranker.max_text_length = 128

        texts = ["short"] * 10
        bs = reranker._adaptive_batch_size(texts)
        assert bs == 32  # Short docs -> full batch

    def test_adaptive_batch_size_long_docs(self):
        config = Stage3Config(device="cpu", use_fp16=False, batch_size=32)
        reranker = AdaptiveCrossEncoderReranker.__new__(AdaptiveCrossEncoderReranker)
        reranker.config = config
        reranker.max_text_length = 128

        texts = ["word " * 250] * 5  # ~250 words each
        bs = reranker._adaptive_batch_size(texts)
        assert bs < 32  # Long docs -> smaller batch

    def test_adaptive_batch_size_empty(self):
        config = Stage3Config(device="cpu", use_fp16=False, batch_size=32)
        reranker = AdaptiveCrossEncoderReranker.__new__(AdaptiveCrossEncoderReranker)
        reranker.config = config
        reranker.max_text_length = 128

        bs = reranker._adaptive_batch_size([])
        assert bs == 32

    def test_rerank_does_not_mutate_shared_config(self, sample_docs, sample_query):
        """Adaptive reranker must not mutate self.config.batch_size (thread-safety)."""
        config = Stage3Config(
            model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
            device="cpu",
            cache_dir="./models",
            top_k_final=3,
            batch_size=2,
            use_fp16=False,
        )
        reranker = AdaptiveCrossEncoderReranker(config)
        original_batch_size = config.batch_size

        long_docs = ["word " * 250] * 4  # forces _adaptive_batch_size < config.batch_size
        candidates = [
            {"doc_id": i, "document": doc, "score": 0.5, "stage2_score": 0.5}
            for i, doc in enumerate(long_docs)
        ]
        reranker.rerank(sample_query, candidates)

        assert config.batch_size == original_batch_size, (
            "AdaptiveCrossEncoderReranker.rerank mutated the shared config.batch_size "
            "— this is a thread-safety bug. The override must be passed as a call arg."
        )


class TestEarlyExitFlag:
    """Tests for the explicit last_early_exit flag (replaces fragile inference)."""

    def test_last_early_exit_set_true_on_early_exit(self):
        """When early-exit triggers, last_early_exit must be True."""
        from unittest.mock import patch
        config = Stage3Config(
            device="cpu", use_fp16=False,
            early_exit_enabled=True,
            early_exit_threshold=0.01,   # tiny gap -> always triggers
            early_exit_min_candidates=3,
            top_k_final=2,
        )
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.config = config
        import logging
        reranker.logger = logging.getLogger("test")
        reranker.last_early_exit = False
        reranker.predict = lambda *a, **k: None  # should never be called on early exit

        # Build candidates with a large top1-median gap so early exit triggers.
        candidates = [
            {"doc_id": i, "document": f"doc {i}", "stage2_score": 0.99 if i == 0 else 0.1}
            for i in range(5)
        ]
        results = reranker.rerank("q", candidates)
        assert reranker.last_early_exit is True
        assert len(results) == 2  # top_k_final
        # Early-exit results keep stage2 ordering, no stage3_score injected
        assert "stage3_score" not in results[0]

    def test_last_early_exit_set_false_on_empty(self):
        """Empty candidates must set last_early_exit=False, not leave it stale."""
        config = Stage3Config(device="cpu", use_fp16=False)
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.config = config
        import logging
        reranker.logger = logging.getLogger("test")
        reranker.last_early_exit = True  # stale True
        reranker.predict = lambda *a, **k: []

        results = reranker.rerank("q", [])
        assert results == []
        assert reranker.last_early_exit is False
