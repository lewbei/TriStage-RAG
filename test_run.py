#!/usr/bin/env python3
"""Quick test script for TriStage-RAG pipeline"""
import sys
import os
import time
import logging

# Suppress noisy logs
logging.basicConfig(level=logging.WARNING, format='%(name)s - %(levelname)s - %(message)s')

# Allow running directly without `pip install -e .`; no-op once installed.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tristage_rag.stage1_retriever import Stage1Retriever, Stage1Config
from tristage_rag.stage2_rescorer import ColBERTScorer, Stage2Config
from tristage_rag.stage3_reranker import CrossEncoderReranker, Stage3Config

# Test documents
DOCS = [
    "The quick brown fox jumps over the lazy dog. This classic pangram contains every letter of the English alphabet.",
    "Python is a high-level programming language known for its simplicity and readability. It supports multiple paradigms.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming.",
    "The solar system consists of the Sun and everything that orbits it, including eight planets and their moons.",
    "Deep learning uses neural networks with many layers to model complex patterns in data. It has revolutionized AI.",
    "Natural language processing allows computers to understand, interpret, and generate human language.",
    "Retrieval-augmented generation combines information retrieval with text generation for more accurate responses.",
    "The theory of relativity, developed by Albert Einstein, describes the relationship between space, time, and gravity.",
    "Quantum computing leverages quantum mechanical phenomena to process information in fundamentally new ways.",
    "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities.",
]

QUERY = "What is machine learning and how does it relate to artificial intelligence?"

def main():
    print("=" * 60)
    print("TriStage-RAG Test Run")
    print("=" * 60)

    # Stage 1 (public model - embeddinggemma-300m requires HF auth)
    print("\n[Stage 1] Loading embedding model (all-MiniLM-L6-v2)...")
    t0 = time.time()
    s1_cfg = Stage1Config(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        cache_dir="./models",
        index_dir="./faiss_index",
        top_k_candidates=10,
        batch_size=8,
        enable_bm25=True,
        use_fp16=False,
    )
    stage1 = Stage1Retriever(s1_cfg)
    print(f"  Model loaded in {time.time()-t0:.2f}s")

    print(f"  Indexing {len(DOCS)} documents...")
    t0 = time.time()
    stage1.add_documents(DOCS)
    print(f"  Indexed in {time.time()-t0:.2f}s")

    print(f"  Searching: '{QUERY}'")
    t0 = time.time()
    s1_results = stage1.search(QUERY, top_k=10)
    print(f"  Stage 1 returned {len(s1_results)} candidates in {time.time()-t0:.3f}s")
    for i, r in enumerate(s1_results[:3]):
        print(f"    #{i+1} score={r['score']:.4f} doc={r['document'][:60]}...")

    # Stage 2
    print("\n[Stage 2] Loading ColBERT model (lightonai/GTE-ModernColBERT-v1)...")
    t0 = time.time()
    s2_cfg = Stage2Config(
        model_name="lightonai/GTE-ModernColBERT-v1",
        device="cpu",
        cache_dir="./models",
        top_k_candidates=5,
        batch_size=4,
        max_seq_length=192,
        use_fp16=False,
    )
    stage2 = ColBERTScorer(s2_cfg)
    print(f"  Model loaded in {time.time()-t0:.2f}s")

    # Rescore top 5 from stage 1
    s2_input = s1_results[:5]
    print(f"  Rescoring {len(s2_input)} candidates...")
    t0 = time.time()
    s2_results = stage2.rescore_candidates(QUERY, s2_input)
    print(f"  Stage 2 returned {len(s2_results)} candidates in {time.time()-t0:.3f}s")
    for i, r in enumerate(s2_results[:3]):
        print(f"    #{i+1} score={r.get('stage2_score', 0):.4f} doc={r['document'][:60]}...")

    # Stage 3
    print("\n[Stage 3] Loading cross-encoder (ms-marco-MiniLM-L6-v2)...")
    t0 = time.time()
    s3_cfg = Stage3Config(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        device="cpu",
        cache_dir="./models",
        top_k_final=5,
        batch_size=8,
        max_length=256,
        use_fp16=False,
    )
    stage3 = CrossEncoderReranker(s3_cfg)
    print(f"  Model loaded in {time.time()-t0:.2f}s")

    # Rerank top 5 from stage 2
    s3_input = s2_results[:5]
    print(f"  Reranking {len(s3_input)} candidates...")
    t0 = time.time()
    final = stage3.rerank(QUERY, s3_input)
    print(f"  Stage 3 returned {len(final)} final results in {time.time()-t0:.3f}s")

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    for i, r in enumerate(final):
        print(f"\n#{i+1}")
        print(f"  S1={r.get('stage1_score', 0):.4f}  S2={r.get('stage2_score', 0):.4f}  S3={r.get('stage3_score', 0):.4f}")
        print(f"  {r['document'][:120]}...")

    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
