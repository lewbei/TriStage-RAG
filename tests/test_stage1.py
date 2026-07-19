"""Tests for Stage 1 retriever."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from tristage_rag.stage1_retriever import Stage1Retriever, Stage1Config, BM25Index


class TestBM25Index:
    def test_fit_and_search(self):
        docs = [
            "machine learning is great",
            "deep learning uses neural networks",
            "the sun is a star",
        ]
        idx = BM25Index()
        idx.fit(docs)
        results = idx.search("machine learning", top_k=2)
        assert len(results) <= 2
        # Doc 0 should rank highest
        assert results[0][0] == 0

    def test_inverted_index_populated(self):
        docs = ["hello world", "hello python"]
        idx = BM25Index()
        idx.fit(docs)
        assert "hello" in idx.inverted_index
        assert len(idx.inverted_index["hello"]) == 2

    def test_empty_query_returns_empty(self):
        docs = ["hello world"]
        idx = BM25Index()
        idx.fit(docs)
        results = idx.search("xyz123_nonexistent", top_k=5)
        assert results == []

    def test_tokenize(self):
        idx = BM25Index()
        tokens = idx.tokenize("Hello, World! 123")
        assert tokens == ["hello", "world", "123"]


class TestStage1Retriever:
    def test_add_and_search(self, stage1_config, sample_docs):
        retriever = Stage1Retriever(stage1_config)
        retriever.add_documents(sample_docs)
        results = retriever.search("machine learning", top_k=3)
        assert len(results) > 0
        assert all("document" in r for r in results)
        assert all("score" in r for r in results)

    def test_metadata_independence(self, stage1_config, sample_docs):
        """Verify the mutable metadata bug fix."""
        retriever = Stage1Retriever(stage1_config)
        retriever.add_documents(sample_docs)
        # Mutate one metadata entry
        retriever.doc_metadata[0]["key"] = "value"
        assert retriever.doc_metadata[1] == {}

    def test_save_and_load_index(self, stage1_config, sample_docs, tmp_path):
        retriever = Stage1Retriever(stage1_config)
        retriever.add_documents(sample_docs)
        idx_path = str(tmp_path / "test_index.pkl")
        retriever.save_index(idx_path)

        retriever2 = Stage1Retriever(stage1_config)
        retriever2.load_index(idx_path)
        assert len(retriever2.documents) == len(sample_docs)

    def test_empty_search_raises(self, stage1_config):
        retriever = Stage1Retriever(stage1_config)
        with pytest.raises(ValueError, match="No documents indexed"):
            retriever.search("test")

    def test_get_stats(self, stage1_config, sample_docs):
        retriever = Stage1Retriever(stage1_config)
        retriever.add_documents(sample_docs)
        stats = retriever.get_stats()
        assert stats["total_documents"] == len(sample_docs)
        assert stats["embedding_dimension"] > 0
