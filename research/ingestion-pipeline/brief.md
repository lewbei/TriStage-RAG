# Research Brief: Fast Multi-Stage RAG Ingestion

**Date**: 2026-07-08
**Depth**: standard
**Audience**: Developer building a 3-stage RAG pipeline (dense + ColBERT + cross-encoder)

## Core Question

How to make bulk ingestion (10k-100k documents) fast and memory-efficient for a 3-stage RAG pipeline that runs:
1. Stage 1: SentenceTransformer encoding → FAISS + BM25
2. Stage 2: ColBERT-style per-token encoding → cached embeddings
3. Stage 3: Cross-encoder (query-time only)

## Current Bottlenecks

- **Ingestion**: 10k docs takes ~5.5 min (CPU). Stage 2 (ColBERT) is 3-4x slower than Stage 1.
- **Storage**: ColBERT cache is 3.7 GB for 10k docs (float32, ~128 tokens/doc, 768-dim)
- **Query latency**: ~550ms on CPU (ColBERT MaxSim over 50 candidates)

## Scope

- Focus on ingestion-time optimizations (not query-time)
- CPU and GPU approaches
- Must maintain retrieval quality (no accuracy loss)

## Angles

1. SentenceTransformer batch encoding optimization (GPU batching, mixed precision, model selection)
2. ColBERT indexing: pre-computation, compression (PQ, residual), incremental updates
3. Parallel ingestion patterns in production RAG systems
4. FAISS indexing strategies for different corpus sizes
5. Memory-efficient embedding storage (quantization, mmap, memory-mapped caches)
