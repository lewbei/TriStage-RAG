r"""
Quick Stage-3 response script (non-MCP)

Runs the full tri-stage retrieval pipeline (FAISS/BM25 -> ColBERT -> Cross-Encoder)
and returns the top passage from the final stage (stage-3). No LLM generation.

Usage (from repo root):
    python non_mcp\respond_stage3.py "your question"
    python non_mcp\respond_stage3.py "your question" path\to\docs

If a docs folder is provided on first run, files (*.txt, *.md) are ingested and
the Stage 1 index is persisted to faiss_index/ for future runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Allow running directly without `pip install -e .`; no-op once installed.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tristage_rag.retrieval_pipeline import RetrievalPipeline


def _read_text_file(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def ensure_ingest_from_dir(pipeline: RetrievalPipeline, docs_dir: Path) -> int:
    """Optionally ingest .txt/.md files from a directory into the index.

    Returns the number of documents added. Also persists the Stage 1 index for reuse.
    """
    if not docs_dir.exists():
        return 0

    texts: List[str] = []
    meta: List[Dict[str, Any]] = []
    for p in docs_dir.rglob("*"):
        if p.suffix.lower() in {".txt", ".md"} and p.is_file():
            content = _read_text_file(p)
            if content and content.strip():
                texts.append(content)
                meta.append({
                    "source": str(p),
                    "title": p.stem,
                })

    if texts:
        pipeline.add_documents(texts, metadata=meta)
        # Persist so future runs can skip reingest
        try:
            pipeline.save_index()
        except Exception:
            # Non-fatal if save fails
            pass
    return len(texts)


def respond_from_stage3(question: str, docs_dir: Optional[str] = None) -> str:
    """Return the top stage-3 passage for a question.

    Optionally ingests documents from a folder on first run.
    """
    pipeline = RetrievalPipeline()

    if docs_dir:
        try:
            added = ensure_ingest_from_dir(pipeline, Path(docs_dir))
        except Exception:
            # Continue even if ingest fails; maybe an index already exists
            pass
    else:
        # Try to load any existing index so search won't fail on fresh runs
        try:
            pipeline.load_index()
        except Exception:
            pass

    try:
        out: Dict[str, Any] = pipeline.search(question)
    except Exception:
        return (
            "I ran all three stages, but couldn't use the index (likely empty or missing). "
            "Pass a docs folder on first run or embed via non_mcp\\main.py/webui."
        )
    hits: List[Dict[str, Any]] = (out or {}).get("results", [])

    if not hits:
        return "I ran all three stages, but couldn't find anything relevant in the index."

    best = hits[0]  # already reranked by stage-3
    passage = (best.get("document") or "").strip()
    if not passage:
        return "The last stage returned an empty passage."

    meta = best.get("metadata", {}) if isinstance(best.get("metadata"), dict) else {}
    src = meta.get("title") or meta.get("source") or ""

    # Prefer stage3_score; fall back to stage2 or stage1 score fields if needed
    score = best.get("stage3_score")
    if score is None:
        score = best.get("stage2_score")
    if score is None:
        score = best.get("score")

    footer = ""
    if src or score is not None:
        footer = f"\n\n[source: {src}] [stage3_score: {score}]"
    return passage + footer


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "usage:\n  python non_mcp\\respond_stage3.py \"your question\"\n"
            "  python non_mcp\\respond_stage3.py \"your question\" path\\to\\docs"
        )
        sys.exit(2)

    question = sys.argv[1]
    docs_dir = sys.argv[2] if len(sys.argv) > 2 else None
    print(respond_from_stage3(question, docs_dir))
