# Fast Multi-Stage RAG Ingestion — Research Report

**Date**: 2026-07-08
**Scope**: How to make bulk ingestion (10k-100k documents) fast and memory-efficient for a 3-stage RAG pipeline

---

## Executive Summary

The research reveals that **no major RAG framework runs pipeline stages in parallel for a single document** — parallelism exists at the document level (batch processing) and system level (horizontal distribution), not at the stage level [F3]. This means our parallel Stage 1+Stage 2 approach was fighting the wrong battle.

The real wins come from three directions:

1. **Switch FAISS from IVFFlat to HNSW** — eliminates training time, faster queries, no quality loss for 10k-100k docs [F4]
2. **Compress ColBERT cache** — residual quantization (nbits=2) achieves 7x reduction; int8 scalar achieves 4x with minimal quality loss [F2, F5]
3. **Use static embedding models** — 100-400x faster on CPU than MiniLM at 85%+ quality [F1]

---

## Key Findings

### 1. FAISS: Switch to HNSW

| Index Type | Build Time | Query Time | Recall@1 | Memory |
|-----------|-----------|-----------|----------|--------|
| Flat (current for <10K) | Instant | O(N) | 100% | 1x |
| IVFFlat (current for >=10K) | 240s (1M) | 0.140ms | ~90% | 1x |
| **HNSW** | **No training** | **0.020ms** | **98.87%** | **3x** |

For 10k-100k docs, HNSW is unambiguously recommended [F4]. It requires **no training** (saves the IVF train step), is 7x faster than IVFFlat, and achieves near-perfect recall. Memory cost is ~3x raw vectors, which is fine for 10k-100k docs.

Decision tree [F4]:
- <10k docs: Flat (brute-force, instant build)
- 10k-100k docs: **HNSW** (no training, best speed/accuracy)
- 100k-1M docs: IVFFlat with K=4-16*sqrt(N) centroids
- Memory-constrained: PQ (98% compression but ~50% recall)

### 2. ColBERT Cache: Residual Compression

The ColBERT paper's residual compression stores:
- **Centroid ID**: int32 (4 bytes)
- **Quantized residual**: nbits per dimension, packed into uint8

For dim=128, nbits=2: **~36 bytes/embedding** vs 256 bytes uncompressed (float16) — a **7x reduction** [F2].

| Format | Bytes/embed | 10k docs (128 tokens/doc) | Compression |
|--------|------------|--------------------------|-------------|
| float32 (original) | 512 | 3.7 GB | 1x |
| float16 | 256 | 1.85 GB | 2x |
| int8 scalar | 128 | ~930 MB | 4x |
| **nbits=2 residual** | **~36** | **~530 MB** | **7x** |
| binary | 16 | ~230 MB | 16x |

Key insight: ColBERTv2 default is nbits=2, computed from a 5% heldout sample using quantile-based buckets (not uniform quantization) [F2]. This is more robust than pure binarization.

### 3. SentenceTransformer Encoding Speed

| Model | CPU (sentences/sec) | GPU (sentences/sec) | Speedup |
|-------|--------------------|--------------------|---------|
| all-MiniLM-L6-v2 (22M, 384d) | 1,739 | 16,942 | 10x |
| **Static embeddings** (MRL-based) | **100,000+** | N/A | **100-400x vs MiniLM** |

Static embedding models achieve 100-400x faster CPU inference than MiniLM at 85%+ quality [F1]. These are lookup-based models that avoid the Transformer forward pass entirely.

Other acceleration paths [F1]:
- **Multi-process encoding**: `model.start_multi_process_pool()` for multi-GPU/multi-CPU
- **ONNX backend**: `backend="onnx"` with CUDAExecutionProvider
- **torch.compile()**: Available for SentenceTransformer modules
- **Batch size tuning**: Optimal value is hardware-dependent; benchmark on your data

### 4. Parallel Ingestion: Document-Level, Not Stage-Level

No major RAG framework (LlamaIndex, Vespa) runs pipeline stages in parallel for a single document [F3]. The standard pattern is:
- **Sequential transformations** per document (LlamaIndex IngestionPipeline)
- **Parallel at document level** (batch processing, multiprocessing)
- **Horizontal distribution** across nodes (Vespa content nodes)

LlamaIndex's main scaling strategy is **caching per node+transformation pair** — subsequent runs skip unchanged chunks [F3].

### 5. Memory-Mapped Storage

NumPy memmap provides zero-copy access to large embedding arrays on disk — only accessed pages load into RAM [F5]. This is ideal for ColBERT caches that exceed available memory.

---

## Recommended Implementation Plan

### Priority 1: Switch FAISS to HNSW (immediate)
- Replace IndexIVFFlat with IndexHNSWFlat for 10k+ docs
- No training step needed, faster ingestion
- Set M=32 (default), efConstruction=200

### Priority 2: Compress ColBERT cache with int8 (quick win)
- Use `sentence_transformers.util.quantize_embeddings(emb, precision="int8")` for 4x compression
- Minimal quality loss, easy to implement
- Target: 3.7 GB to ~930 MB

### Priority 3: Reduce ColBERT dim to 128 (if using custom model)
- ColBERTv2 uses dim=128 by default (vs 768 from BERT)
- If our model supports it, 6x storage reduction
- May require model fine-tuning

### Priority 4: Evaluate static embeddings (if CPU-only)
- If ingestion speed is critical and GPU unavailable
- 100-400x faster at 85%+ quality
- Trade-off: slightly lower retrieval quality

### Priority 5: Add per-node caching (incremental ingestion)
- Cache embeddings per document+transformation pair
- Skip re-encoding on re-ingestion
- Aligns with LlamaIndex's production pattern

---

## Open Questions

1. What is the actual recall-vs-storage curve for ColBERT multi-vector embeddings under int8 quantization?
2. Can FAISS Scalar Quantizer be directly applied to ColBERT's per-token embedding cache?
3. What is the combined effect of Matryoshka truncation + int8 on ColBERT MaxSim scores?

---

## Sources

| # | Source | Type | URL |
|---|--------|------|-----|
| F1-1 | SBERT Multi-GPU docs | primary | sbert.net/examples/.../computing-embeddings |
| F1-6 | HuggingFace Static Embeddings blog | primary | huggingface.co/blog/static-embeddings |
| F1-7 | all-MiniLM-L6-v2 model card | primary | huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| F2-1 | ColBERTv2 paper (arXiv:2112.01488) | primary | arxiv.org/abs/2112.01488 |
| F2-3 | ColBERT residual.py source | primary | github.com/stanford-futuredata/ColBERT |
| F2-7 | PLAID paper (arXiv:2205.09707) | primary | arxiv.org/abs/2205.09707 |
| F3-1 | LlamaIndex IngestionPipeline docs | primary | docs.llamaindex.ai/en/stable/.../ingestion_pipeline |
| F3-4 | Vespa document processors docs | primary | docs.vespa.ai/en/applications/document-processors |
| F4-1 | FAISS Guidelines wiki | primary | github.com/facebookresearch/faiss/wiki/Guidelines |
| F4-5 | FAISS Indexing 1M vectors wiki | primary | github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors |
| F5-1 | NumPy memmap docs | primary | numpy.org/doc/stable/reference/generated/numpy.memmap |
| F5-5 | Matryoshka Representation Learning blog | primary | huggingface.co/blog/matryoshka |
