# TriStage-RAG Improvement Plan

**Date**: 2026-07-07
**Scope**: Speed & Efficiency + Code Quality & Tests

---

## Phase 1: Shared Utilities & Base Class (Foundation)

### 1A. Create `src/utils.py` — Common utilities
**Files**: Create `src/utils.py`, update `src/__init__.py`

Extract duplicated logic into reusable functions:

```python
# src/utils.py

def resolve_model_dir(model_name: str, cache_dir: str) -> str:
    """Resolve local model path: flat dir → legacy nested dir → fallback name."""
    base_dir = os.path.join(cache_dir, os.path.basename(model_name))
    legacy_dir = os.path.join(cache_dir, model_name)
    if os.path.isdir(base_dir):
        return base_dir
    elif os.path.isdir(legacy_dir):
        return legacy_dir
    return model_name

def get_device(device: str, use_gpu_if_available: bool = True) -> str:
    """Determine best device for inference."""
    if device == "auto":
        if torch.cuda.is_available() and use_gpu_if_available:
            return "cuda"
        return "cpu"
    return device

def clear_gpu_memory(device: str):
    """Clear CUDA cache if on GPU."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
```

**Duplication removed**: 5 copies of model dir resolution, 2 copies of `_get_device()`, 2 copies of `clear_gpu_memory()`.

**Effort**: 0.5h

---

### 1B. Create `src/stage_base.py` — Shared base class
**Files**: Create `src/stage_base.py`

```python
class StageBase:
    """Shared lifecycle for Stage 2 and Stage 3."""

    def __init__(self, device: str, use_gpu_if_available: bool = True):
        self.device = get_device(device, use_gpu_if_available)

    def clear_gpu_memory(self):
        clear_gpu_memory(self.device)

    def __del__(self):
        try:
            self.clear_gpu_memory()
        except Exception:  # No bare except
            pass
```

Make `ColBERTScorer` and `CrossEncoderReranker` inherit from `StageBase`.

**Effort**: 0.5h

---

### 1C. Fix shared reference bug
**File**: `src/stage1_retriever.py:302`

```python
# BEFORE (bug: all dicts share one reference)
metadata = [{}] * len(documents)

# AFTER
metadata = [{} for _ in range(len(documents))]
```

**Effort**: 0.1h

---

## Phase 2: Code Quality Fixes

### 2A. Replace deprecated `torch.cuda.amp.autocast()`
**Files**: `src/stage1_retriever.py:235`, `src/stage2_rescorer.py:152,219`, `src/stage3_reranker.py:165`

```python
# BEFORE
with torch.cuda.amp.autocast():

# AFTER
with torch.amp.autocast("cuda"):
```

**Effort**: 0.2h

---

### 2B. Remove bare except clauses
**Files**: `src/stage2_rescorer.py:350`, `src/stage3_reranker.py:317`, `src/retrieval_pipeline.py:643`, `non_mcp/main.py:487`

```python
# BEFORE
except:

# AFTER
except Exception:
```

**Effort**: 0.1h

---

### 2C. Remove global `warnings.filterwarnings('ignore')`
**Files**: `src/stage2_rescorer.py:13`, `src/stage3_reranker.py:13`

Delete the global suppress lines. If specific warnings need suppression, scope them locally.

**Effort**: 0.1h

---

### 2D. Remove unused imports
**Files**: All stage files + `retrieval_pipeline.py`

| File | Remove |
|------|--------|
| `src/stage1_retriever.py` | `json`, `field`, `ThreadPoolExecutor` |
| `src/stage2_rescorer.py` | `ThreadPoolExecutor` |
| `src/stage3_reranker.py` | `ThreadPoolExecutor`, `math` |
| `src/retrieval_pipeline.py` | `Union` (if unused) |

**Effort**: 0.2h

---

### 2E. Fix hardcoded `use_fp16=False` in non_mcp/main.py
**File**: `non_mcp/main.py:175,188,201`

Replace hardcoded `use_fp16=False` with config-driven value, or at minimum set to `True` (matching the pipeline defaults).

**Effort**: 0.2h

---

## Phase 3: Performance Optimizations (High Impact)

### 3A. BM25: Build inverted index + compile regex
**File**: `src/stage1_retriever.py` — `BM25Index` class

**Changes**:
1. Compile regex once at class level or `__init__`
2. Build inverted index (`doc_freqs_by_term: Dict[str, List[int]]`) during `fit()`
3. Replace O(N) brute-force `search()` with inverted-index lookup
4. Use `heapq.nlargest(top_k, ...)` instead of full sort
5. Compute IDF in single pass during `fit()` using inverted index

```python
import re

_TOKENIZE_RE = re.compile(r'[^a-z0-9\s]')

class BM25Index:
    def __init__(self, k1=1.2, b=0.75):
        # ... existing ...
        self.inverted_index = {}  # term → set of doc_indices

    def tokenize(self, text):
        text = text.lower()
        text = _TOKENIZE_RE.sub(' ', text)
        return text.split()

    def fit(self, documents):
        # ... existing tokenization ...
        # Build inverted index (single pass O(N) instead of O(V*N))
        self.inverted_index = defaultdict(set)
        for doc_idx, tokens in enumerate(tokenized_docs):
            for token in set(tokens):
                self.inverted_index[token].add(doc_idx)

        # IDF via inverted index (single pass)
        for term, doc_set in self.inverted_index.items():
            df = len(doc_set)
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query, top_k=10):
        query_tokens = set(self.tokenize(query))
        # Collect only docs that match at least one query token
        candidate_docs = set()
        for token in query_tokens:
            candidate_docs |= self.inverted_index.get(token, set())

        # Score only candidates using heapq
        scores = []
        for doc_idx in candidate_docs:
            score = self.score(query, doc_idx)
            scores.append((doc_idx, score))

        return heapq.nlargest(top_k, scores, key=lambda x: x[1])
```

**Speedup**: O(N log k) instead of O(N log N); avoids scoring non-matching docs.

**Effort**: 1.5h

---

### 3B. FAISS IVF threshold + retrain on additions
**File**: `src/stage1_retriever.py:256-277`

**Changes**:
1. Raise IVF threshold from 1000 to 10000
2. Add retrain logic when vectors are added to IVF index

```python
def _create_faiss_index(self, embeddings):
    d = embeddings.shape[1]
    if len(embeddings) > 10000:  # Raised from 1000
        quantizer = faiss.IndexFlatIP(d)
        self.faiss_index = faiss.IndexIVFFlat(quantizer, d, self.config.nlist, faiss.METRIC_INNER_PRODUCT)
        self.faiss_index.train(embeddings)
        self.faiss_index.add(embeddings)
        self.faiss_index.nprobe = self.config.nprobe
    else:
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(embeddings)

def add_documents(self, documents, metadata=None):
    # ... encode ...
    if self.faiss_index is None:
        self._create_faiss_index(embeddings)
    elif isinstance(self.faiss_index, faiss.IndexIVFFlat) and self.faiss_index.is_trained:
        # For IVF: try add, if needed retrain
        try:
            self.faiss_index.add(embeddings)
        except RuntimeError:
            # Need to retrain with all vectors
            all_embeddings = np.vstack([existing_embeddings, embeddings])
            self._create_faiss_index(all_embeddings)
    else:
        self.faiss_index.add(embeddings)
```

**Effort**: 1h

---

### 3C. BM25 incremental index instead of full rebuild
**File**: `src/stage1_retriever.py:316-322`

**Change**: After Phase 3A's inverted index, update incrementally:

```python
def _update_bm25_incremental(self, new_docs_start_idx: int):
    """Update BM25 index for newly added documents only."""
    for doc_idx in range(new_docs_start_idx, len(self.documents)):
        tokens = self.tokenize(self.documents[doc_idx])
        self.doc_freqs.append(defaultdict(int))
        for token in tokens:
            self.doc_freqs[doc_idx][token] += 1
            self.inverted_index[token].add(doc_idx)
        self.doc_lens.append(len(tokens))
        self.vocabulary.update(tokens)
    self.corpus_size = len(self.documents)
    self.avg_doc_len = sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 0
    # Recompute IDF only for affected terms
    for term in self.vocabulary:
        df = len(self.inverted_index.get(term, set()))
        self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
```

**Effort**: 1h

---

### 3D. ColBERT: Batch GPU sync elimination
**File**: `src/stage2_rescorer.py:225-231`

**Change**: Remove per-doc `.sum().item()` GPU sync; use vectorized operation:

```python
# BEFORE
for j in range(len(batch_docs)):
    attention_mask = encoded["attention_mask"][j]
    seq_length = attention_mask.sum().item()  # GPU sync per doc!
    token_embeddings = outputs.last_hidden_state[j, :seq_length, :]
    all_embeddings.append(token_embeddings)

# AFTER — no GPU sync in inner loop
masks = encoded["attention_mask"]  # [batch, seq_len]
lengths = masks.sum(dim=1)        # [batch] — single GPU sync
for j in range(len(batch_docs)):
    token_embeddings = outputs.last_hidden_state[j, :lengths[j], :]
    all_embeddings.append(token_embeddings)
```

**Speedup**: Eliminates N GPU→CPU syncs per batch.

**Effort**: 0.3h

---

### 3E. Cross-encoder: Batch CPU transfer
**File**: `src/stage3_reranker.py:178`

**Change**: Move `.cpu().tolist()` outside the batch loop:

```python
# BEFORE (inside batch loop)
batch_scores = scores.cpu().tolist()  # Sync per batch
all_scores.extend(batch_scores)

# AFTER — accumulate on GPU, transfer once
all_gpu_scores.append(scores)

# After loop:
if all_gpu_scores:
    all_scores = torch.cat(all_gpu_scores).cpu().tolist()
```

**Effort**: 0.3h

---

### 3F. Remove unnecessary `dict.copy()` in rescore/rerank loops
**Files**: `src/stage2_rescorer.py:279,288`, `src/stage3_reranker.py:251`

**Change**: Build new dicts with additional keys instead of copying:

```python
# BEFORE
updated_candidate = candidate.copy()
updated_candidate["stage2_score"] = score_float
updated_candidate["stage"] = "stage2"

# AFTER
updated_candidate = {**candidate, "stage2_score": score_float, "stage": "stage2"}
```

**Effort**: 0.2h

---

## Phase 4: Test Suite

### 4A. Create `tests/conftest.py` + `tests/test_bm25.py`
**Files**: Create `tests/conftest.py`, `tests/test_bm25.py`

```python
# tests/conftest.py
import pytest
from src.stage1_retriever import BM25Index

@pytest.fixture
def bm25_index():
    index = BM25Index()
    docs = [
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks",
        "natural language processing enables text understanding",
        "retrieval augmented generation combines retrieval and generation",
    ]
    index.fit(docs)
    return index

# tests/test_bm25.py
class TestBM25:
    def test_tokenize_lowercases(self, bm25_index):
        assert bm25_index.tokenize("Hello World") == ["hello", "world"]

    def test_tokenize_removes_punctuation(self, bm25_index):
        assert bm25_index.tokenize("AI, ML & NLP!") == ["ai", "ml", "nlp"]

    def test_fit_builds_index(self, bm25_index):
        assert bm25_index.corpus_size == 4
        assert len(bm25_index.inverted_index) > 0

    def test_search_returns_top_k(self, bm25_index):
        results = bm25_index.search("machine learning", top_k=2)
        assert len(results) == 2
        assert results[0][1] >= results[1][1]

    def test_search_relevant_doc_ranked_first(self, bm25_index):
        results = bm25_index.search("neural networks deep learning", top_k=1)
        assert results[0][0] == 1  # doc index 1 about deep learning

    def test_search_empty_query(self, bm25_index):
        results = bm25_index.search("", top_k=5)
        assert len(results) == 0 or all(s == 0 for _, s in results)
```

**Effort**: 1h

---

### 4B. Create `tests/test_stage1_retriever.py`
**Files**: Create `tests/test_stage1_retriever.py`

Mock `SentenceTransformer` and `faiss` for unit tests:

```python
# tests/test_stage1_retriever.py
from unittest.mock import patch, MagicMock
import numpy as np

class TestStage1Retriever:
    def test_add_documents_metadata_not_shared(self):
        """Regression test for mutable shared reference bug."""
        from src.stage1_retriever import Stage1Retriever, Stage1Config
        # ... mock model ...
        config = Stage1Config(device="cpu", use_fp16=False)
        with patch("src.stage1_retriever.SentenceTransformer"):
            retriever = Stage1Retriever(config)
            retriever.add_documents(["doc1", "doc2", "doc3"])
            # Metadata dicts must be independent
            assert retriever.doc_metadata[0] is not retriever.doc_metadata[1]

    def test_search_empty_index_raises(self):
        # Test that search raises ValueError when no docs indexed
        ...

    def test_model_dir_resolution(self):
        """Test that resolve_model_dir checks flat then legacy."""
        ...
```

**Effort**: 1h

---

### 4C. Create `tests/test_utils.py`
**Files**: Create `tests/test_utils.py`

```python
class TestResolveModelDir:
    def test_flat_dir_preferred(self, tmp_path):
        flat = tmp_path / "my-model"
        flat.mkdir()
        from src.utils import resolve_model_dir
        assert resolve_model_dir("org/my-model", str(tmp_path)) == str(flat)

    def test_legacy_dir_fallback(self, tmp_path):
        legacy = tmp_path / "org/my-model"
        legacy.mkdir(parents=True)
        from src.utils import resolve_model_dir
        assert resolve_model_dir("org/my-model", str(tmp_path)) == str(legacy)

    def test_returns_name_if_not_found(self, tmp_path):
        from src.utils import resolve_model_dir
        assert resolve_model_dir("org/my-model", str(tmp_path)) == "org/my-model"
```

**Effort**: 0.5h

---

### 4D. Create `tests/test_pipeline.py`
**Files**: Create `tests/test_pipeline.py`

Integration-style tests with mocked stages:

```python
class TestRetrievalPipeline:
    def test_search_calls_all_stages(self):
        # Mock stage1/2/3, verify pipeline orchestrates them
        ...

    def test_batch_search_returns_list(self):
        ...
```

**Effort**: 1h

---

## Phase 5: Cleanup

### 5A. Delete empty `test_run.py` (or keep as demo script, separate from test suite)

### 5B. Add `pytest.ini` / `pyproject.toml` section for test configuration

```ini
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
```

**Effort**: 0.2h

---

## Summary Table

| Group | Description | Effort | Risk |
|-------|-------------|--------|------|
| 1A | Create `src/utils.py` | 0.5h | Low |
| 1B | Create `src/stage_base.py` | 0.5h | Low |
| 1C | Fix mutable shared reference bug | 0.1h | Low |
| 2A | Replace deprecated `torch.cuda.amp.autocast()` | 0.2h | Low |
| 2B | Fix bare except clauses | 0.1h | Low |
| 2C | Remove global warnings suppression | 0.1h | Low |
| 2D | Remove unused imports | 0.2h | Low |
| 2E | Fix hardcoded use_fp16=False | 0.2h | Low |
| 3A | BM25 inverted index + compiled regex | 1.5h | Medium |
| 3B | FAISS IVF threshold + retrain | 1.0h | Medium |
| 3C | BM25 incremental index | 1.0h | Medium |
| 3D | ColBERT batch GPU sync elimination | 0.3h | Low |
| 3E | Cross-encoder batch CPU transfer | 0.3h | Low |
| 3F | Remove dict.copy() overhead | 0.2h | Low |
| 4A | BM25 unit tests | 1.0h | Low |
| 4B | Stage1 retriever tests | 1.0h | Low |
| 4C | Utils tests | 0.5h | Low |
| 4D | Pipeline integration tests | 1.0h | Low |
| 5A-B | Cleanup + test config | 0.2h | Low |
| **Total** | | **~10.5h** | |

---

## Recommended Execution Order

```
Phase 1 (Foundation)     → Phase 2 (Quality)     → Phase 3 (Performance)    → Phase 4 (Tests)    → Phase 5 (Cleanup)
├─ 1A: utils.py          ├─ 2A: autocast         ├─ 3A: BM25 inverted      ├─ 4A: BM25 tests   ├─ 5A: cleanup
├─ 1B: stage_base.py     ├─ 2B: bare except      ├─ 3B: FAISS IVF          ├─ 4B: stage1 tests ├─ 5B: pytest config
└─ 1C: shared ref bug    ├─ 2C: warnings         ├─ 3C: BM25 incremental   ├─ 4C: utils tests
                         ├─ 2D: unused imports   ├─ 3D: ColBERT sync       └─ 4D: pipeline tests
                         └─ 2E: use_fp16         ├─ 3E: cross-encoder
                                                └─ 3F: dict.copy()
```

Each phase is independently testable. Phase 1 enables Phase 2 (base class refactors). Phase 3 (performance) is independent of Phase 2 but benefits from the cleaner structure. Phase 4 tests can be written alongside each phase.
