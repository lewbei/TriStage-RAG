"""Tests for Stage 2 ColBERT scorer."""
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from tristage_rag.stage2_rescorer import ColBERTScorer, Stage2Config


class TestColBERTScorer:
    def test_maxsim_score(self):
        """Test MaxSim scoring with known embeddings."""
        config = Stage2Config(device="cpu", use_fp16=False)
        scorer = ColBERTScorer.__new__(ColBERTScorer)
        scorer.config = config
        scorer.logger = MagicMock()

        # Query: 3 tokens, Doc: 4 tokens, dim=8
        query_emb = torch.randn(1, 3, 8)
        doc_emb = torch.randn(1, 4, 8)
        score = scorer._maxsim_score(query_emb, doc_emb)
        assert isinstance(score.item(), float)
        assert -1.0 <= score.item() <= 1.0

    def test_rescore_candidates(self, sample_docs, sample_query):
        """Test that rescore returns candidates with stage2_score."""
        config = Stage2Config(
            model_name="lightonai/GTE-ModernColBERT-v1",
            device="cpu",
            cache_dir="./models",
            top_k_candidates=3,
            batch_size=2,
            use_fp16=False,
        )
        scorer = ColBERTScorer(config)

        candidates = [
            {"doc_id": i, "document": doc, "score": 0.5, "stage1_score": 0.5}
            for i, doc in enumerate(sample_docs[:3])
        ]
        results = scorer.rescore_candidates(sample_query, candidates)
        assert len(results) > 0
        assert all("stage2_score" in r for r in results)
        # Results should be sorted by stage2_score
        scores = [r["stage2_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestCachePersistence:
    """Round-trip tests for export_cache / import_cache.

    These exercise the exact serialize/deserialize path that
    RetrievalPipeline.save_index/load_index uses (including the int8
    quantized-tuple branch), WITHOUT requiring model downloads. We bypass
    __init__ and inject synthetic cache entries directly.
    """

    def _make_scorer(self, precision: str):
        config = Stage2Config(
            device="cpu", use_fp16=False,
            cache_precision=precision,
            top_k_candidates=3,
        )
        scorer = ColBERTScorer.__new__(ColBERTScorer)
        scorer.config = config
        scorer.logger = MagicMock()
        scorer._doc_embeddings = {}
        return scorer

    def test_export_import_roundtrip_float(self):
        """Float32 tensors survive export → numpy → import with identical values."""
        scorer = self._make_scorer("float32")
        # doc_id -> (emb_tensor, seq_len)
        emb = torch.randn(5, 8, dtype=torch.float32)
        scorer._doc_embeddings = {0: (emb, 5)}

        exported = scorer.export_cache()
        # Simulate the pipeline's numpy conversion (save path)
        import pickle
        on_disk = {did: (e.numpy(), sl) for did, (e, sl) in exported.items()}
        blob = pickle.dumps(on_disk)
        loaded = pickle.loads(blob)
        # Simulate the pipeline's restore path (torch.from_numpy)
        restored = {did: (torch.from_numpy(e), sl) for did, (e, sl) in loaded.items()}

        scorer2 = self._make_scorer("float32")
        scorer2.import_cache(restored)
        assert scorer2.has_cached_docs()
        out_emb, out_sl = scorer2._doc_embeddings[0]
        assert out_sl == 5
        assert torch.allclose(out_emb.float(), emb)

    def test_export_import_roundtrip_int8(self):
        """The int8 (quantized, scale) tuple branch survives a full round-trip."""
        # Seed for determinism: int8 quantization of unbounded randn can exceed
        # a tight atol on unlucky draws (values near the per-channel clamp
        # boundary lose up to ~scale/2). Seeding pins a known-good draw.
        torch.manual_seed(42)
        scorer = self._make_scorer("int8")
        # Build a genuine int8 entry via the production quantizer
        float_emb = torch.randn(5, 8, dtype=torch.float32)
        quant = scorer._quantize_int8(float_emb)  # (int8_tensor, scale_vec)
        scorer._doc_embeddings = {7: (quant, 5)}

        exported = scorer.export_cache()
        # Save path: pipeline nests tuples -> ((int8_np, scale_np), seq_len)
        import pickle
        on_disk = {}
        for did, (emb, sl) in exported.items():
            q, scale = emb  # int8 tuple
            on_disk[did] = ((q.numpy(), scale.numpy()), sl)
        blob = pickle.dumps(on_disk)
        loaded = pickle.loads(blob)
        # Restore path: pipeline detects int8 dtype and rebuilds tuples
        restored = {}
        for did, (emb_data, sl) in loaded.items():
            if (isinstance(emb_data, tuple) and len(emb_data) == 2
                    and hasattr(emb_data[0], 'dtype') and emb_data[0].dtype == 'int8'):
                restored[did] = (
                    (torch.from_numpy(emb_data[0]), torch.from_numpy(emb_data[1])),
                    sl,
                )
            else:
                restored[did] = (torch.from_numpy(emb_data), sl)

        scorer2 = self._make_scorer("int8")
        scorer2.import_cache(restored)
        # Dequantize and confirm values match within int8 precision tolerance.
        # NOTE: _dequantize takes the FULL cache entry (emb_or_tuple, seq_len),
        # matching the real caller in rescore_candidates (`self._dequantize(cached)`).
        # Tolerance is relative to the per-channel scale, since int8 quantization
        # error is bounded by scale/2 per element. atol=0.05 + rtol=1e-2 covers
        # the seeded draw comfortably while still catching gross corruption.
        out_entry = scorer2._doc_embeddings[7]
        deq = scorer2._dequantize(out_entry)
        assert torch.allclose(deq, float_emb, atol=0.05, rtol=1e-2), (
            "int8 round-trip lost too much precision"
        )

    def test_import_cache_replaces_not_merges(self):
        """import_cache must replace the cache entirely, not merge into existing."""
        scorer = self._make_scorer("float32")
        scorer._doc_embeddings = {0: (torch.zeros(3, 4), 3)}
        scorer.import_cache({1: (torch.ones(3, 4), 3)})
        assert set(scorer._doc_embeddings.keys()) == {1}, (
            "import_cache merged instead of replacing"
        )

    def test_has_cached_docs(self):
        scorer = self._make_scorer("float32")
        assert scorer.has_cached_docs() is False
        scorer._doc_embeddings = {0: (torch.zeros(3, 4), 3)}
        assert scorer.has_cached_docs() is True
        scorer.import_cache({})
        assert scorer.has_cached_docs() is False

    def test_dequantize_handles_int8_full_cache_entry(self):
        """Regression: _dequantize must accept the FULL cache entry
        ``(emb_or_tuple, seq_len)``, including when the emb part is itself an
        int8 tuple. Previously it did ``cached[0].dtype`` which raised
        ``AttributeError: 'tuple' object has no attribute 'dtype'`` whenever
        int8 precision was used (the default).
        """
        scorer = self._make_scorer("int8")
        float_emb = torch.randn(5, 8, dtype=torch.float32)
        quant = scorer._quantize_int8(float_emb)  # (int8_tensor, scale_vec)
        # Full cache entry as produced by add_documents / restored by import_cache
        full_entry = (quant, 5)

        deq = scorer._dequantize(full_entry)
        assert torch.allclose(deq, float_emb, atol=1e-2)

    def test_dequantize_handles_float_full_cache_entry(self):
        """_dequantize also works for non-quantized (float) full cache entries."""
        scorer = self._make_scorer("float32")
        float_emb = torch.randn(5, 8, dtype=torch.float32)
        full_entry = (float_emb, 5)
        deq = scorer._dequantize(full_entry)
        assert torch.equal(deq, float_emb)
