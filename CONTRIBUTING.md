# Contributing to TriStage-RAG

Thanks for your interest! This is an active research and development project.
The core pipeline is in `tristage_rag/`; everything else is a surface (benchmark
harness, standalone CLI, web UI).

## Development setup

```bash
git clone <repository-url>
cd TriStage-RAG
pip install -e ".[all]"     # core + benchmark + web UI + dev tools
cp .env.example .env        # add HUGGING_FACE_HUB_TOKEN if using gated models
```

## Running tests

```bash
pytest                # fast unit tests (BM25, utils, normalization, adaptive batching)
pytest -m slow        # integration tests that download real models (network/GPU needed)
pytest tests/ -v      # everything, verbose
```

The `slow` marker is registered in `pyproject.toml`. Fast unit tests run without
network or GPU.

## Code style

- **Python ≥ 3.9.** Use `from __future__ import annotations` only when needed.
- **Type hints** on all public functions and dataclass fields (the core already
  follows this).
- **Docstrings** on public classes and non-trivial functions. `PipelineConfig`
  is a good reference for the expected level of detail.
- **Prefer `logging` over `print`** in library code (`tristage_rag/`,
  `non_mcp/`). See `KNOWN_ISSUES.md` for the current `basicConfig` situation.
- **No bare `except Exception: pass`.** Catch specific exceptions or re-raise.
- **Line length ~100** (configured under `[tool.ruff]` in `pyproject.toml`).

## Before submitting a pull request

1. Create a feature branch (the repo isn't on a default-branch-protection model yet,
   but please branch anyway).
2. Add or update tests for your change. Fast unit tests preferred; mark anything
   that downloads models with `@pytest.mark.slow`.
3. Run `pytest` and ensure the fast suite passes cleanly (no
   `PytestUnknownMarkWarning`).
4. If your change touches packaging, verify `pip install -e ".[all]"` still works
   and `python -c "import tristage_rag; print(tristage_rag.__version__)"` succeeds.
5. Update `README.md` and `KNOWN_ISSUES.md` if your change affects behavior or
   resolves an open issue.

## Areas that need help

The most impactful open items are tracked in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).
A short list of good first issues:

- Consolidate `non_mcp/main.py::ThreeStageRetrievalSystem` to wrap `RetrievalPipeline`.
- Add tests for `save_index`/`load_index` round-trip (covers the int8 ColBERT path).
- Wire `TRISTAGE_DEVICE` / `TRISTAGE_LOW_MEMORY` env vars (or remove them from docs).
- Add a GitHub Actions workflow that runs `pytest` on push.

## Reporting bugs

Open an issue with: the command you ran, the expected vs actual behavior, the
full traceback, your OS / Python / GPU, and whether you installed via
`pip install -e ".[all]"` or ran scripts directly.

## License

By contributing you agree your contributions are licensed under the project's
[MIT license](LICENSE).
