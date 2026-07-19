#!/usr/bin/env python3
"""Test TriStage-RAG against the enterprise RAG benchmark with proper chunking."""
import sys
import os
import json
import time
import pandas as pd
from typing import List, Dict

# Allow running directly without `pip install -e .`; no-op once installed.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tristage_rag.stage1_retriever import Stage1Retriever, Stage1Config
from tristage_rag.stage2_rescorer import ColBERTScorer, Stage2Config
from tristage_rag.stage3_reranker import CrossEncoderReranker, Stage3Config

BENCHMARK_PATH = "benchmark/enterprise_rag_sample/data.parquet"


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def chunk_documents(docs: List[Dict], chunk_size: int = 512, overlap: int = 64) -> List[Dict]:
    """Chunk a list of documents, preserving metadata."""
    all_chunks = []
    for doc in docs:
        full_text = doc.get("full_text", "")
        title = doc.get("title", "")
        doc_id = doc.get("doc_id", "")

        text_chunks = chunk_text(full_text, chunk_size, overlap)

        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "text": chunk,
                "doc_id": doc_id,
                "title": title,
                "chunk_idx": i,
                "artifact_type": doc.get("artifact_type", ""),
                "project_scope": doc.get("project_scope", ""),
            })
    return all_chunks


def load_benchmark():
    """Load the benchmark dataset."""
    df = pd.read_parquet(BENCHMARK_PATH)
    return df


def main():
    print("=" * 70)
    print("TriStage-RAG Enterprise Benchmark Test (with chunking)")
    print("=" * 70)

    # Load benchmark
    df = load_benchmark()
    print(f"\nLoaded {len(df)} benchmark queries")

    # Initialize pipeline
    print("\n[Pipeline] Initializing stages...")
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

    s3_cfg = Stage3Config(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        device="cpu",
        cache_dir="./models",
        top_k_final=3,
        batch_size=4,
        max_length=256,
        use_fp16=False,
    )
    stage3 = CrossEncoderReranker(s3_cfg)
    print(f"  Pipeline initialized in {time.time()-t0:.2f}s")

    # Run each query
    results = []
    for idx, row in df.iterrows():
        query = row["query_text"]
        gold_answer = row["gold_answer"]
        difficulty = row["difficulty"]
        reasoning_type = row["reasoning_type"]
        evidence_doc_ids = json.loads(row["evidence_doc_ids"])

        # Parse source documents from bundle
        source_docs = json.loads(row["source_document_bundle_json"])

        # Chunk documents
        chunks = chunk_documents(source_docs, chunk_size=384, overlap=48)
        chunk_texts = [c["text"] for c in chunks]

        print(f"\n--- Query {idx+1}/{len(df)} [{difficulty}] ---")
        print(f"  Q: {query}")
        print(f"  Gold: {gold_answer}")
        print(f"  Docs: {len(source_docs)} -> Chunks: {len(chunks)}")

        # Clear and re-index for each query (different doc sets)
        stage1.documents.clear()
        stage1.doc_metadata.clear()
        stage1.faiss_index = None
        stage1.bm25_index = None

        # Index chunks
        t0 = time.time()
        stage1.add_documents(chunk_texts)
        index_time = time.time() - t0

        # Stage 1: Fast retrieval
        t0 = time.time()
        s1_results = stage1.search(query, top_k=min(8, len(chunks)))
        s1_time = time.time() - t0

        if not s1_results:
            print(f"  No results from Stage 1")
            results.append({
                "query_id": row["query_id"],
                "query": query,
                "gold_answer": gold_answer,
                "top_result": "",
                "score": 0,
                "term_overlap": 0,
                "difficulty": difficulty,
                "reasoning_type": reasoning_type,
                "num_chunks": len(chunks),
            })
            continue

        # Stage 2: Rescoring
        t0 = time.time()
        s2_results = stage2.rescore_candidates(query, s1_results[:min(5, len(s1_results))])
        s2_time = time.time() - t0

        # Stage 3: Reranking
        t0 = time.time()
        final = stage3.rerank(query, s2_results[:min(3, len(s2_results))])
        s3_time = time.time() - t0

        # Get top result
        top_result = final[0]["document"] if final else ""
        top_score = final[0].get("stage3_score", 0) if final else 0

        # Check if any retrieved chunk contains the gold answer key terms
        gold_terms = set(gold_answer.lower().split())
        all_retrieved_text = " ".join([r.get("document", "") for r in final])
        result_terms = set(all_retrieved_text.lower().split())
        overlap = len(gold_terms & result_terms) / len(gold_terms) if gold_terms else 0

        # Check if retrieved chunks come from correct evidence documents
        retrieved_doc_ids = set()
        for r in final:
            for c in chunks:
                if c["text"] == r.get("document", ""):
                    retrieved_doc_ids.add(c["doc_id"])
                    break

        evidence_match = len(retrieved_doc_ids & set(evidence_doc_ids)) / len(evidence_doc_ids) if evidence_doc_ids else 0

        print(f"  Top chunk: {top_result[:120]}...")
        print(f"  Score: {top_score:.4f} | Term overlap: {overlap:.1%} | Evidence match: {evidence_match:.1%}")
        print(f"  Times: idx={index_time:.3f}s s1={s1_time:.3f}s s2={s2_time:.3f}s s3={s3_time:.3f}s")

        results.append({
            "query_id": row["query_id"],
            "query": query,
            "gold_answer": gold_answer,
            "top_result": top_result,
            "score": top_score,
            "term_overlap": overlap,
            "evidence_match": evidence_match,
            "difficulty": difficulty,
            "reasoning_type": reasoning_type,
            "num_chunks": len(chunks),
        })

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    df_results = pd.DataFrame(results)
    print(f"\nTotal queries: {len(df_results)}")
    print(f"Average term overlap: {df_results['term_overlap'].mean():.1%}")
    print(f"Average evidence match: {df_results['evidence_match'].mean():.1%}")
    print(f"\nBy difficulty:")
    print(df_results.groupby("difficulty")[["term_overlap", "evidence_match"]].mean().to_string())
    print(f"\nBy reasoning type:")
    print(df_results.groupby("reasoning_type")[["term_overlap", "evidence_match"]].mean().to_string())

    # Save results
    output_path = "benchmark/enterprise_rag_sample/results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
