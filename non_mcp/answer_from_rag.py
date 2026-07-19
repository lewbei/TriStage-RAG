#!/usr/bin/env python3
"""
Generate an answer from retrieved context (full RAG):
  - Runs the three-stage retriever (S1 dense+BM25 → S2 ColBERT → S3 cross-encoder)
  - Builds a compact prompt from the top-k passages
  - Calls a small HF generator (default: flan-t5-small) to produce an answer

Usage (from repo root):
  python non_mcp\answer_from_rag.py --question "your question" --device cpu --top-k 4 \
    --gen-model google/flan-t5-small

If you need to ingest documents first (PDF/DOCX/txt/md), use:
  - Web UI:  python non_mcp\webui\app.py  (then embed on /embed)
  - Script:  python non_mcp\embed_and_query.py --docs-dir C:\\path\\to\\docs
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Allow running directly without `pip install -e .`; no-op once installed.
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from non_mcp.main import AppConfig, ThreeStageRetrievalSystem
from non_mcp.generation import SimpleGenerator, GenerationConfig


def best_contexts(result: Dict[str, Any], k: int) -> List[str]:
    res = (result or {}).get("results") or []
    texts: List[str] = []
    for r in res[:k]:
        t = (r.get("document") or "").strip()
        if t:
            texts.append(t)
    return texts


def main():
    ap = argparse.ArgumentParser(description="Answer a question using TriStage retrieval + generator")
    ap.add_argument("--question", required=True, help="The question to answer")
    ap.add_argument("--device", default="cpu", help="cpu or cuda for retrieval and generation")
    ap.add_argument("--top-k", type=int, default=4, help="#contexts to include in the prompt")
    ap.add_argument("--gen-model", default="google/flan-t5-small", help="HF model id for generation")
    args = ap.parse_args()

    # Build system config (BM25 enabled by default)
    cfg = AppConfig(
        models_dir=str(ROOT / "models"),
        data_dir=str(ROOT / "data"),
        index_dir=str(ROOT / "faiss_index"),
        max_results=max(10, args.top_k),
        enable_bm25=True,
        device=args.device,
        log_level="INFO",
    )
    system = ThreeStageRetrievalSystem(cfg)

    # Run retrieval
    retr = system.search(args.question, top_k=max(10, args.top_k))
    ctxs = best_contexts(retr, args.top_k)
    if not ctxs:
        print("No retrieved contexts. Embed documents first via web UI or embed_and_query.py")
        sys.exit(2)

    # Generate answer
    gcfg = GenerationConfig(
        model_name=args.gen_model,
        device="cuda" if args.device == "cuda" else "cpu",
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.95,
        use_fp16=(args.device == "cuda"),
    )
    gen = SimpleGenerator(gcfg)
    answer = gen.generate(args.question, ctxs)

    print("\nQuestion:")
    print(args.question)
    print("\nAnswer:")
    print(answer)
    print("\n--- Contexts used (top-k) ---")
    for i, c in enumerate(ctxs, 1):
        print(f"[Context {i}] {c[:500].replace('\n', ' ')}...")


if __name__ == "__main__":
    main()

