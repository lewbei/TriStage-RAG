# Known Issues

This file tracks known limitations and deferred work. It is intentionally honest:
the core retrieval pipeline (`tristage_rag/`) is well-engineered, but the surrounding
surface has rough edges. Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

Items marked ✅ in the history below have been resolved; they're kept for context.

---

## Correctness / structural

1. **Duplicate pipeline implementation.** `non_mcp/main.py::ThreeStageRetrievalSystem`
   (834 lines) reimplements orchestration that already lives in
   `tristage_rag.retrieval_pipeline.RetrievalPipeline`. It should wrap/subclass the
   core instead of duplicating stage wiring and top-k defaults. Two sources of truth.

2. **Result-schema divergence.** The core pipeline emits `stage3_score`; the
   `non_mcp` implementation emits `final_score`. Scripts that cross the two
   (e.g. reading `final_score` from a `RetrievalPipeline` result) will silently
   miss the field. Unify on one schema.

3. ✅ **(Resolved)** Thread-safety in adaptive batching.
   `AdaptiveCrossEncoderReranker.rerank` previously mutated the shared
   `self.config.batch_size` in a `try/finally`. Fixed: `predict`/`rerank` now
   accept a per-call `batch_size` override; the shared config is never mutated.
   Regression test: `tests/test_stage3.py::TestAdaptiveCrossEncoderReranker::test_rerank_does_not_mutate_shared_config`.

4. ✅ **(Resolved)** Fragile early-exit detection.
   `RetrievalPipeline.search` previously inferred early-exit via
   `not any("stage3_score" in r ...)`, which falsely reported an early exit
   whenever Stage 3 returned `[]` for an unrelated reason. Fixed: the reranker
   now sets an explicit `last_early_exit` flag that the pipeline reads.
   Regression tests: `tests/test_stage3.py::TestEarlyExitFlag`.

5. **`batch_search` doesn't actually batch.**
   `RetrievalPipeline.batch_search` just loops `search()` per query. No shared
   encoding, no concurrency.

## Engineering hygiene

6. **Multiple `logging.basicConfig` calls.** The pipeline and the web UI each
   call `logging.basicConfig`. After the first call it's a no-op, so subsequent
   handlers are silently lost when these coexist. Should use a single configured
   logger / `dictConfig`.

7. **`print()` and bare `except Exception: pass`** in several non-core scripts
   (`respond_stage3.py`, `webui/app.py`, `non_mcp/main.py`). Replace with
   `logging` and specific exceptions.

8. **Hardcoded Flask secret key.** `non_mcp/webui/app.py` defaults to
   `"dev-secret"` (overridable via `NON_MCP_WEBUI_SECRET`). The UI binds to
   `127.0.0.1` only, so impact is local, but the default should fail loudly.

9. ✅ **(Resolved)** Dead config keys.
   Removed `PipelineConfig.max_memory_usage_gb` and `Stage1Config.max_text_length`
   (both declared, never read). `Stage2Config.pooling_method` is kept — it's
   surfaced in `get_model_info()` as metadata even though not enforced.

10. ✅ **(Resolved)** Stale benchmark config + IVF keys.
    `non_mcp/pipeline_config.yaml` had dead `nlist`/`nprobe` (Stage 1 switched to
    HNSW). Replaced with the actual `hnsw_m` / `hnsw_ef_search` /
    `hnsw_ef_construction` / `hnsw_threshold` keys.

11. ✅ **(Resolved)** Stage 2 `top_k` inconsistency.
    `PipelineConfig.stage2_top_k=50`, `Stage2Config.top_k_candidates=100`, and
    `benchmark/config.yaml` stage2 `top_k: 100` disagreed. Aligned all three to
    **50** (per the SemEval-2026 finding, `papers/2605.12028v1.pdf`).

12. ✅ **(Resolved)** Encapsulation leak in save/load.
    `RetrievalPipeline.save_index`/`load_index` reached into
    `stage2._doc_embeddings` directly. Fixed: `ColBERTScorer` now exposes
    `has_cached_docs()` / `export_cache()` / `import_cache()`, which the pipeline
    uses instead.

13. **Unwired env vars.** The README's predecessor documented `TRISTAGE_DEVICE`,
    `TRISTAGE_LOW_MEMORY`, `TRISTAGE_SAMPLE_SIZE`, and `LOG_LEVEL`, but the code
    reads from YAML configs (the benchmark loader explicitly disables env
    overrides). These have been removed from the current README; either wire them
    into the code or drop entirely.

## Tests

14. **No tests for** `save_index`/`load_index` round-trip (the int8 restore path
    is non-trivial), the parallel `add_documents` path, or the OOM fallback
    cascades. *The save/load path now goes through the public `export_cache`/
    `import_cache` API, which makes adding these tests easier.*

15. **`non_mcp/` is untested.** Neither `ThreeStageRetrievalSystem` nor the web UI
    have automated tests.

## Deployment

16. **No containerization or CI.** No `Dockerfile`, no `docker-compose.yml`, no
    `.github/workflows/`. A `pytest`-on-push CI workflow is the natural next step.

17. **No HTTP health endpoint.** If you ever expose this over HTTP, add a
    liveness/readiness route for orchestrators (k8s).
