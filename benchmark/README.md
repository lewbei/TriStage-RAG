# TriStage-RAG MTEB Benchmark

Complete MTEB (Massive Text Embedding Benchmark) evaluation for the TriStage-RAG pipeline on the LIMIT dataset.

Tested and aligned with MTEB v2.0.0. End-to-end tri-stage is executed even when the evaluator provides (query, document) pairs (MTEB v2 behavior). The wrapper indexes unique documents once and runs the full pipeline per query before assigning scores back to the requested pairs.

## Overview

This benchmark provides a complete MTEB-compatible implementation for evaluating the 3-stage retrieval pipeline:
- **Stage 1**: Dense embeddings with `google/embeddinggemma-300m`
- **Stage 2**: ColBERT reranking with `lightonai/GTE-ModernColBERT-v1`  
- **Stage 3**: Cross-encoder scoring with `cross-encoder/ms-marco-MiniLM-L6-v2`

## Quick Start

### Prerequisites

Recommended: install from this repo’s requirements (pins MTEB 2.0.0)

```cmd
pip install -r requirements.txt
```

The benchmark will automatically download both the dataset and required models if they don't exist locally.

**Note**: The `google/embeddinggemma-300m` model is gated and requires a Hugging Face token. You can:
1. Set the `HF_TOKEN` environment variable, or
2. Use the `--hf-token` argument, or  
3. Use `--low-mem` mode to download alternative models that don't require authentication

Or install the specific MTEB version explicitly

```cmd
pip install "mteb==2.0.0"
```

Notes
- The custom LIMIT tasks in `benchmark/limit_mteb_tasks.py` are required; they load the local dataset and provide the `relevant_docs` dict format MTEB expects.
- On large pairwise runs, initial indexing can take time; the wrapper avoids re-indexing the same set of docs within a process run.

Alternatively, install from the v2.0.0 tag

```cmd
pip install git+https://github.com/embeddings-benchmark/mteb@v2.0.0
```

### GPU (CUDA) acceleration

If you have an NVIDIA GPU and want to run on CUDA:

1) Install PyTorch with CUDA wheels (match your CUDA/driver version)

```cmd
# Example for CUDA 12.1 wheels (adjust as needed)
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2) Verify CUDA is detected

```cmd
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"
```

3) Run the benchmark with GPU

```cmd
python benchmark\run_mteb_evaluation.py --tasks LIMITSmallRetrieval --limit-path benchmark\limit_dataset\limit-small --device cuda
```

Notes
- Stage 2 and Stage 3 default to fp16 when GPU is available.
- If you see CUDA OOM, try lowering batch sizes in `benchmark/config.yaml` or use `--sample-size`.
- For laptops with hybrid GPUs, ensure the Python process uses the discrete GPU.

### Dataset Setup

The benchmark can automatically download the LIMIT dataset if it's not found locally. The expected structure is:
```
benchmark/limit_dataset/
├── limit-small/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels.jsonl
└── limit/
    ├── corpus.jsonl
    ├── queries.jsonl
    └── qrels.jsonl
```

#### Auto-Download Feature

If the dataset is not found, the benchmark will automatically attempt to download it from the Google DeepMind repository. No manual setup is required!

#### Manual Download (Optional)

If you prefer to download manually or need to refresh the dataset:

```cmd
# Download the small LIMIT dataset (recommended for testing)
python download_limit_dataset.py --dataset limit-small

# Download the full LIMIT dataset
python download_limit_dataset.py --dataset limit

# Validate existing dataset
python download_limit_dataset.py --validate-only --dataset limit-small

# Show dataset information
python download_limit_dataset.py --info --dataset limit-small
```

### Model Management

The benchmark automatically downloads required models if they don't exist. You can also manage models manually:

```cmd
# Check model availability
python run_mteb_evaluation.py --check-models

# Download models only (don't run evaluation)
python run_mteb_evaluation.py --download-models-only

# Download low-memory models
python run_mteb_evaluation.py --download-models-only --low-mem

# Clean up all downloaded models
python run_mteb_evaluation.py --clean-models

# Use standalone model downloader
python download_models.py --info
python download_models.py --low-memory --check-only

# Download with Hugging Face token (for gated models)
python download_models.py --hf-token YOUR_TOKEN_HERE
python run_mteb_evaluation.py --download-models-only --hf-token YOUR_TOKEN_HERE
```

### Running Evaluation

#### LIMIT-Small Dataset (Quick Test)

Run from the repo root (Windows cmd):

```cmd
python benchmark\run_mteb_evaluation.py --tasks LIMITSmallRetrieval --limit-path benchmark\limit_dataset\limit-small --device cpu
```

Run on CUDA (if available):

```cmd
python benchmark\run_mteb_evaluation.py --tasks LIMITSmallRetrieval --limit-path benchmark\limit_dataset\limit-small --device cuda
```

Results format: MTEB v2 may return results as a list or a dict. The `run_mteb_evaluation.py` script prints a short summary either way and writes full results to the output folder.

#### Full LIMIT Dataset (Complete Evaluation)
```cmd
python benchmark\run_mteb_evaluation.py --tasks LIMITRetrieval --limit-path benchmark\limit_dataset\limit --device cpu
```

#### With Sample Size (For Testing)
```cmd
python benchmark\run_mteb_evaluation.py --tasks LIMITSmallRetrieval --limit-path benchmark\limit_dataset\limit-small --sample-size 100 --device cpu
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--tasks` | MTEB tasks to evaluate | `LIMITSmallRetrieval` |
| `--limit-path` | Path to LIMIT dataset | Auto-detect |
| `--output` | Results output folder | `benchmark/mteb_results` |
| `--device` | Device to run models on | `auto` |
| `--cache-dir` | Model cache directory | `../models` (uses top-level models) |
| `--index-dir` | FAISS index directory | `./faiss_index` |
| `--sample-size` | Sample size for evaluation | `None` (full) |
| `--log-level` | Logging level | `INFO` |
| `--low-mem` | Use low-memory settings | `False` |
| `--stage1-model` | Override Stage 1 model | `None` |
| `--download-models-only` | Only download models | `False` |
| `--check-models` | Only check model availability | `False` |
| `--clean-models` | Clean up downloaded models | `False` |
| `--hf-token` | Hugging Face token for gated models | `HF_TOKEN` env var |

## Available Tasks

### Hugging Face Token Setup

The `google/embeddinggemma-300m` model is gated and requires authentication:

1. **Get a token**: Visit https://huggingface.co/settings/tokens
2. **Request access**: Go to https://huggingface.co/google/embeddinggemma-300m and request access
3. **Use the token**:
   ```bash
   # Set environment variable
   export HF_TOKEN=your_token_here
   
   # Or use command line argument
   python run_mteb_evaluation.py --hf-token your_token_here
   ```

4. **Alternative**: Use `--low-mem` mode to download non-gated alternative models

- `LIMITSmallRetrieval`: Small version for quick evaluation (46 documents, 1000 queries)
- `LIMITRetrieval`: Full dataset for complete evaluation (250k+ documents, 50k+ queries)

## Output

Evaluation results are saved to the specified output directory (default: `benchmark/mteb_results/`) with the following structure:
```
mteb_results/
├── LIMITSmallRetrieval/
│   ├── model_predictions.json
│   └── scores.json
└── LIMITRetrieval/
    ├── model_predictions.json
    └── scores.json
```

## Key Metrics

The evaluation reports standard retrieval metrics:
- **NDCG@10**: Normalized Discounted Cumulative Gain at 10
- **Recall@10**: Recall at 10 documents
- **MAP@10**: Mean Average Precision at 10
- **MRR@10**: Mean Reciprocal Rank at 10

## File Structure

```
benchmark/
├── run_mteb_evaluation.py          # Main evaluation script
├── tristage_mteb_model.py          # MTEB-compatible 3-stage model
├── limit_mteb_tasks.py             # LIMIT dataset task definitions
├── download_limit_dataset.py       # Dataset download script
├── limit_dataset/                   # Local LIMIT dataset (auto-downloaded)
├── models/                         # Cached model files (symlink to ../models)
├── faiss_index/                    # FAISS index storage
└── mteb_results/                   # Evaluation results
```

## Architecture

### TriStage-RAG Pipeline
1. **Stage 1 - Dense Retrieval**: 
   - Uses `google/embeddinggemma-300m` for initial document embedding
   - FAISS index for efficient similarity search
   - Returns top-k candidates (default: 500)

2. **Stage 2 - ColBERT Reranking**:
   - Uses `lightonai/GTE-ModernColBERT-v1` for contextual reranking
   - Processes top-k candidates from Stage 1
   - Returns refined candidates (default: 100)

3. **Stage 3 - Cross-Encoder Scoring**:
   - Uses `cross-encoder/ms-marco-MiniLM-L6-v2` for final scoring
   - Provides precise relevance scores
   - Returns final results (default: 20)

### MTEB Integration
- Implements complete MTEB model interface
- Compatible with MTEB v2.0.0 evaluator
- Uses local LIMIT dataset files (no remote download required)
- Proper handling of corpus/query encoding and search

Implementation notes for LIMIT tasks:
- The LIMIT tasks in `limit_mteb_tasks.py` return pure-Python structures per MTEB v2 expectations:
   - `corpus`: Dict[str, {"text", "title"}]
   - `queries`: Dict[str, str]
   - `relevant_docs`: Dict[str, Dict[str, int]]
   This avoids pytrec_eval format errors like “Expected dictionary as value”.

## Troubleshooting

### pytrec_eval: "Expected dictionary as value"
- Ensure you’re using the provided `benchmark/limit_mteb_tasks.py` (it returns `relevant_docs` as a dict-of-dicts per MTEB v2) and you run `run_mteb_evaluation.py` from this repository.
- If you previously modified tasks or are mixing MTEB versions, reinstall requirements and retry.

### “No documents indexed. Call add_documents() first.”
- This should not occur with the provided wrapper; it proactively indexes corpora for retrieval and pairs mode.
- If you see this, ensure your LIMIT dataset paths are correct and documents are non-empty.

### Memory Issues (Windows)
If you encounter paging file / out-of-memory errors:
- Run on CPU: add `--device cpu` (recommended for low-memory setups)
- Reduce sample size: `--sample-size 50`
- Stage 1 has a built-in low-memory fallback to `sentence-transformers/all-MiniLM-L6-v2` when it detects Windows paging file errors
- Optionally increase Windows virtual memory (paging file) in System settings

### Dataset Not Found
Ensure the LIMIT dataset is in the correct location:
```
benchmark/limit_dataset/limit-small/
├── corpus.jsonl
├── queries.jsonl
└── qrels.jsonl
```

### Model Loading Issues
- Check internet connection for first-time model downloads
- Verify model cache directory permissions
- Ensure sufficient disk space for model files

## Example Output

```
Using LIMIT dataset from: benchmark/limit_dataset/limit-small
Loaded 1000 queries
Loaded 46 documents
Loaded 1000 query-document relevance pairs
Initializing TriStage-RAG model for MTEB evaluation...
Model created: TriStageMTEBModel(stage1=google/embeddinggemma-300m, stage2=lightonai/GTE-ModernColBERT-v1, stage3=cross-encoder/ms-marco-MiniLM-L6-v2)
Tasks to evaluate: ['LIMITSmallRetrieval']
Starting MTEB evaluation on 1 tasks...
Evaluation completed successfully!
Results saved to: benchmark/mteb_results

Summary of results:
  LIMITSmallRetrieval: 0.4523
```

When the evaluator uses pairwise CE scoring, the wrapper still executes tri-stage retrieval internally and produces final scores from Stage 3 reranking for the requested (query, document) pairs.

## Citation

If you use this benchmark, please cite:
- **MTEB**: [Massive Text Embedding Benchmark](https://github.com/embeddings-benchmark/mteb)
- **LIMIT**: [LIMIT Dataset](https://github.com/google-deepmind/limit)
- **TriStage-RAG**: This 3-stage retrieval pipeline implementation
