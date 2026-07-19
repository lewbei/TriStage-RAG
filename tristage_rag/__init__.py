"""TriStage-RAG: a 3-stage retrieval pipeline.

Stage 1: fast candidate generation (FAISS dense + BM25, fused via RRF)
Stage 2: multi-vector rescoring (ColBERT-style MaxSim over an int8-quantized
         per-token embedding cache pre-computed at index time)
Stage 3: cross-encoder reranking with early-exit and adaptive batching

The public API is intentionally small: most users only need
:class:`~tristage_rag.RetrievalPipeline` and :class:`~tristage_rag.PipelineConfig`.
Stage implementation classes (``Stage1Retriever``, ``ColBERTScorer``,
``CrossEncoderReranker``, ``AdaptiveCrossEncoderReranker``) are importable from
their submodules but are considered internal.
"""

from .retrieval_pipeline import RetrievalPipeline, PipelineConfig
from .stage1_retriever import Stage1Config
from .stage2_rescorer import Stage2Config
from .stage3_reranker import Stage3Config
from .__version__ import __version__

__all__ = [
    "RetrievalPipeline",
    "PipelineConfig",
    "Stage1Config",
    "Stage2Config",
    "Stage3Config",
    "__version__",
]
