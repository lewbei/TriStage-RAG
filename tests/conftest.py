"""Shared fixtures for TriStage-RAG tests."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# Sample documents for testing
SAMPLE_DOCS = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing allows computers to understand text.",
    "The solar system has eight planets orbiting the sun.",
    "Quantum computing leverages quantum mechanical phenomena.",
]

SAMPLE_QUERY = "What is machine learning?"


@pytest.fixture
def sample_docs():
    return SAMPLE_DOCS.copy()


@pytest.fixture
def sample_query():
    return SAMPLE_QUERY


@pytest.fixture
def mock_embedding_model():
    """Mock SentenceTransformer that returns random embeddings."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 384
    model.get_embedding_dimension.return_value = 384

    def mock_encode(texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        embeddings = np.random.randn(n, 384).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)

    model.encode = mock_encode
    return model


@pytest.fixture
def stage1_config(tmp_path):
    from tristage_rag.stage1_retriever import Stage1Config
    return Stage1Config(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        cache_dir=str(tmp_path / "models"),
        index_dir=str(tmp_path / "faiss_index"),
        top_k_candidates=5,
        batch_size=4,
        enable_bm25=True,
        use_fp16=False,
    )


@pytest.fixture
def stage2_config():
    from tristage_rag.stage2_rescorer import Stage2Config
    return Stage2Config(
        model_name="lightonai/GTE-ModernColBERT-v1",
        device="cpu",
        cache_dir="./models",
        top_k_candidates=5,
        batch_size=4,
        max_seq_length=128,
        use_fp16=False,
    )


@pytest.fixture
def stage3_config():
    from tristage_rag.stage3_reranker import Stage3Config
    return Stage3Config(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        device="cpu",
        cache_dir="./models",
        top_k_final=3,
        batch_size=4,
        max_length=128,
        use_fp16=False,
    )
