#!/usr/bin/env python3
"""
A/B comparison: Gemma-only (dense) vs Gemma+BM25 fusion on your documents.

Usage examples (from repo root):
  python non_mcp\ab_compare.py --docs-dir "C:\\path\\to\\docs" --query "your question"
  python non_mcp\ab_compare.py --docs-dir "C:\\path\\to\\docs" --queries-file queries.txt --top-k 5

Notes
- Reads .txt and .md files in --docs-dir recursively.
- Builds two systems sharing the same models but separate data/index dirs.
- Prints Stage 1/2/3 timings and top-1 result for each.
"""

from __future__ import annotations

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# Allow running directly without `pip install -e .`; no-op once installed.
THIS = Path(__file__).resolve()
_REPO_ROOT = THIS.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from non_mcp.main import AppConfig, ThreeStageRetrievalSystem


def _read_text_file(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def load_docs_from_dir(docs_dir: Path) -> List[str]:
    texts: List[str] = []
    for p in docs_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            t = _read_text_file(p)
            if t and t.strip():
                texts.append(t)
    return texts


def build_system(base_dirs: Path, enable_bm25: bool, device: str = "cpu") -> ThreeStageRetrievalSystem:
    base_dirs.mkdir(parents=True, exist_ok=True)
    models_dir = str((THIS.parents[1] / "models").resolve())
    data_dir = str((base_dirs / "data").resolve())
    index_dir = str((base_dirs / "index").resolve())

    cfg = AppConfig(
        models_dir=models_dir,
        data_dir=data_dir,
        index_dir=index_dir,
        max_results=20,
        enable_bm25=enable_bm25,
        device=device,
        log_level="INFO",
    )
    return ThreeStageRetrievalSystem(cfg)


def print_one_result(tag: str, out: Dict[str, Any], top_k: int) -> None:
    print(f"\n[{tag}] Stage timings: S1={out['stage1_time']:.3f}s S2={out['stage2_time']:.3f}s S3={out['stage3_time']:.3f}s Total={out['total_time']:.3f}s")
    hits = out.get("results", [])
    if not hits:
        print(f"[{tag}] No results.")
        return
    best = hits[0]
    preview = (best.get("document") or "")[:160].replace("\n", " ")
    print(f"[{tag}] Top-1 final_score={best.get('final_score'):.4f} (s1={best.get('stage1_score', 0):.4f}, s2={best.get('stage2_score', 0):.4f}, s3={best.get('stage3_score', 0):.4f})")
    print(f"[{tag}] {preview}...")


def main():
    ap = argparse.ArgumentParser(description="A/B compare Gemma-only vs Gemma+BM25 on your docs")
    ap.add_argument("--docs-dir", required=True, help="Directory containing .txt/.md docs")
    ap.add_argument("--query", help="Single query to test")
    ap.add_argument("--queries-file", help="File with one query per line")
    ap.add_argument("--top-k", type=int, default=5, help="Results to show per system")
    ap.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    ap.add_argument("--workdir", default="./ab_runs", help="Where to put temp data/index dirs")
    args = ap.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        print(f"ERROR: docs-dir not found: {docs_dir}")
        sys.exit(2)

    # Load documents
    docs = load_docs_from_dir(docs_dir)
    if not docs:
        print("ERROR: no .txt/.md documents found to index.")
        sys.exit(3)

    # Prepare two systems with separate state dirs
    workdir = Path(args.workdir).resolve()
    sys_off = build_system(workdir / "bm25_off", enable_bm25=False, device=args.device)
    sys_on = build_system(workdir / "bm25_on", enable_bm25=True, device=args.device)

    # Index docs into both
    sys_off.add_documents(docs, source=str(docs_dir))
    sys_on.add_documents(docs, source=str(docs_dir))

    # Collect queries
    queries: List[str] = []
    if args.query:
        queries = [args.query]
    elif args.queries_file:
        qp = Path(args.queries_file)
        if not qp.exists():
            print(f"ERROR: queries-file not found: {qp}")
            sys.exit(4)
        with open(qp, "r", encoding="utf-8") as f:
            for line in f:
                q = line.strip()
                if q:
                    queries.append(q)
    else:
        print("Enter a query (blank line to stop):")
        q = input("> ").strip()
        if q:
            queries = [q]
    if not queries:
        print("No queries provided.")
        sys.exit(0)

    # Run A/B for each query
    for q in queries:
        print("\n" + "=" * 80)
        print(f"Query: {q}")
        try:
            out_off = sys_off.search(q, top_k=args.top_k)
            out_on = sys_on.search(q, top_k=args.top_k)
        except Exception as e:
            print(f"ERROR during search: {e}")
            continue

        print_one_result("Gemma-only (BM25 off)", out_off, args.top_k)
        print_one_result("Gemma+BM25 fusion", out_on, args.top_k)

        # Quick comparative note
        s_off = (out_off.get("results") or [{}])[0].get("final_score") if out_off.get("results") else None
        s_on = (out_on.get("results") or [{}])[0].get("final_score") if out_on.get("results") else None
        if s_off is not None and s_on is not None:
            better = "fusion" if s_on > s_off else ("dense-only" if s_off > s_on else "tie")
            print(f"Winner (top-1 final_score): {better}")


if __name__ == "__main__":
    main()

