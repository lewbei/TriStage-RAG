#!/usr/bin/env python3
"""
Embed repo documents (txt/md/json/pdf/docx) and run queries through the
three-stage pipeline (Stage 1: Gemma+FAISS+BM25, Stage 2: ColBERT, Stage 3: Cross-Encoder).

Usage examples (from repo root):
  python non_mcp\embed_and_query.py --docs-dir "C:\\Users\\lewka\\deep_learning\\rag_mcp\\documents" \
    --device cpu --top-k 5 --queries "machine learning" "neural networks"

If --queries is not provided, you will be prompted for a single query.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import time

# Allow running directly without `pip install -e .`; no-op once installed.
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from non_mcp.main import AppConfig, ThreeStageRetrievalSystem


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        window = text[start:end]
        if end < n:
            last_period = window.rfind('.')
            last_newline = window.rfind('\n')
            cut = max(last_period, last_newline)
            if cut > 0 and (end - (start + cut)) < 200:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, 0)
    return chunks


def extract_text_from_path(p: Path) -> str:
    name = p.name.lower()
    try:
        if name.endswith(('.txt', '.md', '.markdown')):
            return p.read_text(encoding='utf-8', errors='ignore')
        if name.endswith('.json'):
            import json as _json
            data = _json.loads(p.read_text(encoding='utf-8', errors='ignore'))
            if isinstance(data, list):
                return "\n\n".join([str(x) for x in data if str(x).strip()])
            if isinstance(data, dict) and 'documents' in data:
                return "\n\n".join([str(x) for x in data['documents'] if str(x).strip()])
            return ""
        if name.endswith('.pdf'):
            from pypdf import PdfReader
            reader = PdfReader(str(p))
            pages = [pg.extract_text() or '' for pg in reader.pages]
            return "\n\n".join(pages)
        if name.endswith('.docx'):
            from docx import Document
            doc = Document(str(p))
            paras = [q.text for q in doc.paragraphs if q.text and q.text.strip()]
            return "\n".join(paras)
        return p.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ''


def embed_directory(
    system: ThreeStageRetrievalSystem,
    docs_dir: Path,
    chunk_size: int = 1000,
    overlap: int = 200,
    max_files: Optional[int] = None,
    max_chunks: Optional[int] = None,
) -> int:
    count = 0
    files_seen = 0
    for ext in ("*.txt", "*.md", "*.markdown", "*.json", "*.pdf", "*.docx"):
        for path in docs_dir.rglob(ext):
            if max_files is not None and files_seen >= max_files:
                return count
            text = extract_text_from_path(path)
            if not text:
                continue
            files_seen += 1
            chunks = chunk_text(text, chunk_size, overlap)
            if not chunks:
                continue
            if max_chunks is not None and count + len(chunks) > max_chunks:
                # Trim to remaining budget
                remain = max(0, max_chunks - count)
                if remain == 0:
                    return count
                chunks = chunks[:remain]
            system.add_documents(chunks, source=str(path))
            count += len(chunks)
            if max_chunks is not None and count >= max_chunks:
                return count
    return count


def run_queries(system: ThreeStageRetrievalSystem, queries: List[str], top_k: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for q in queries:
        out = system.search(q, top_k)
        results.append({"query": q, "out": out})
    return results


def main():
    ap = argparse.ArgumentParser(description="Embed docs and run queries with the three-stage pipeline")
    ap.add_argument("--docs-dir", default=str(ROOT / "documents"), help="Folder with txt/md/json/pdf/docx")
    ap.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--queries", nargs='*', help="Queries to run")
    ap.add_argument("--max-files", type=int, default=None, help="Max files to embed (for quick runs)")
    ap.add_argument("--max-chunks", type=int, default=None, help="Max chunks to embed (for quick runs)")
    args = ap.parse_args()

    cfg = AppConfig(
        models_dir=str(ROOT / "models"),
        data_dir=str(ROOT / "data"),
        index_dir=str(ROOT / "faiss_index"),
        max_results=args.top_k,
        enable_bm25=True,
        device=args.device,
        log_level="INFO",
    )
    system = ThreeStageRetrievalSystem(cfg)

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        print(f"ERROR: docs-dir not found: {docs_dir}")
        sys.exit(2)

    print(f"Embedding from: {docs_dir}")
    t0 = time.time()
    added = embed_directory(
        system,
        docs_dir,
        chunk_size=1000,
        overlap=200,
        max_files=args.max_files,
        max_chunks=args.max_chunks,
    )
    t1 = time.time()
    print(f"Embedded chunks: {added} in {t1 - t0:.2f}s")

    # Run queries
    queries: List[str] = []
    if args.queries:
        queries = args.queries
    else:
        try:
            q = input("Enter a query: ").strip()
            if q:
                queries = [q]
        except EOFError:
            pass
    if not queries:
        print("No queries provided; exiting.")
        return

    def safe_print_line(s: str) -> None:
        try:
            print(s)
        except UnicodeEncodeError:
            enc = (getattr(sys.stdout, 'encoding', None) or 'utf-8')
            try:
                sys.stdout.write(s.encode(enc, errors='ignore').decode(enc, errors='ignore') + "\n")
            except Exception:
                sys.stdout.write((s.encode('ascii', errors='ignore').decode('ascii', errors='ignore') + "\n"))

    for entry in run_queries(system, queries, args.top_k):
        q = entry["query"]
        out = entry["out"] or {}
        safe_print_line("\n" + "=" * 80)
        safe_print_line(f"Query: {q}")
        safe_print_line(f"Timings: S1={out.get('stage1_time', 0):.3f}s S2={out.get('stage2_time', 0):.3f}s S3={out.get('stage3_time', 0):.3f}s Total={out.get('total_time', 0):.3f}s")
        res = out.get("results") or []
        if not res:
            safe_print_line("No results.")
            continue
        for i, r in enumerate(res[:args.top_k], start=1):
            preview = (r.get("document") or "").replace("\n", " ")[:200]
            safe_print_line(f"{i}. final={r.get('final_score'):.4f} s1={r.get('stage1_score', 0):.4f} s2={r.get('stage2_score', 0):.4f} s3={r.get('stage3_score', 0):.4f}")
            safe_print_line(f"   {preview}...")


if __name__ == "__main__":
    main()
