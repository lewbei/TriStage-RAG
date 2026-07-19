#!/usr/bin/env python3
"""
Complete MTEB evaluation script for TriStage-RAG pipeline on LIMIT dataset
Uses local LIMIT dataset files and provides full evaluation functionality
"""

import logging
import argparse
import time
import json
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

# Import our custom model and tasks (package-relative so this works both as
# `python -m benchmark.run_mteb_evaluation` and via the installed package).
from benchmark.tristage_mteb_model import TriStageMTEBModel
from benchmark.limit_mteb_tasks import LIMITSmallRetrieval, LIMITRetrieval
from benchmark.download_limit_dataset import LIMITDatasetDownloader
from benchmark.download_models import ModelDownloader

# Try to import MTEB
try:
    import mteb
    from mteb import MTEB
    MTEB_AVAILABLE = True
except ImportError:
    print("MTEB not available. Install with: pip install mteb")
    MTEB_AVAILABLE = False

def setup_logging(level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_local_limit_dataset(limit_path: Path):
    """Load LIMIT dataset from local files"""
    queries = []
    corpus = []
    qrels = {}
    
    # Load queries
    queries_file = limit_path / "queries.jsonl"
    if queries_file.exists():
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                queries.append(data)
        print(f"Loaded {len(queries)} queries")
    
    # Load corpus
    corpus_file = limit_path / "corpus.jsonl"
    if corpus_file.exists():
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                corpus.append(data)
        print(f"Loaded {len(corpus)} documents")
    
    # Load qrels
    qrels_file = limit_path / "qrels.jsonl"
    if qrels_file.exists():
        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                query_id = str(data["query-id"])
                doc_id = str(data["corpus-id"])
                score = data.get("score", 1)
                
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = score
        print(f"Loaded {len(qrels)} query-document relevance pairs")
    
    return queries, corpus, qrels

def evaluate_with_mteb(model, tasks, output_folder="results"):
    """
    Evaluate model using MTEB framework.
    
    Args:
        model: MTEB-compatible model
        tasks: List of task names or MTEB task objects
        output_folder: Folder to save results
        
    Returns:
        Evaluation results
    """
    if not MTEB_AVAILABLE:
        raise ImportError("MTEB not available")
    
    # Create MTEB evaluation instance
    evaluation = MTEB(tasks=tasks)
    
    # Run evaluation
    print(f"Starting MTEB evaluation on {len(tasks)} tasks...")
    start_time = time.time()
    
    results = evaluation.run(
        model,
        output_folder=output_folder,
        encode_kwargs={'batch_size': 32},
        overwrite_results=True
    )
    
    evaluation_time = time.time() - start_time
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")
    
    return results

def main():
    """Main evaluation function"""
    # Load .env file if it exists
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    
    parser = argparse.ArgumentParser(description="Evaluate TriStage-RAG with MTEB on LIMIT dataset")
    parser.add_argument("--tasks", nargs="+", 
                       default=["LIMITSmallRetrieval"],
                       help="MTEB tasks to evaluate (LIMITSmallRetrieval, LIMITRetrieval)")
    parser.add_argument("--output", type=str, default="benchmark/mteb_results",
                       help="Output folder for results")
    parser.add_argument("--limit-path", type=str, default=None,
                       help="Path to LIMIT dataset directory")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run models on")
    parser.add_argument("--cache-dir", type=str, default="../models",
                       help="Model cache directory")
    parser.add_argument("--index-dir", type=str, default="./faiss_index",
                       help="FAISS index directory")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Sample size for evaluation (None for full evaluation)")
    parser.add_argument("--low-mem", action="store_true",
                       help="Use low-memory settings and smaller models to avoid OOM/paging errors")
    parser.add_argument("--stage1-model", type=str, default=None,
                       help="Override Stage 1 model name (e.g., sentence-transformers/all-MiniLM-L6-v2)")
    parser.add_argument("--download-models-only", action="store_true",
                       help="Only download models, don't run evaluation")
    parser.add_argument("--check-models", action="store_true",
                       help="Only check model availability, don't run evaluation")
    parser.add_argument("--clean-models", action="store_true",
                       help="Clean up all downloaded models")
    parser.add_argument("--hf-token", type=str, default=None,
                       help="Hugging Face token for gated models (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Get Hugging Face token from multiple sources (priority: argument > env var > .env file)
    hf_token = (
        args.hf_token or 
        os.getenv("HF_TOKEN") or 
        os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    
    # Handle model management commands
    if args.clean_models:
        print("Cleaning up models...")
        model_downloader = ModelDownloader(args.cache_dir, hf_token)
        model_downloader.clean_models()
        return
    
    if args.check_models:
        print("Checking model availability...")
        model_downloader = ModelDownloader(args.cache_dir, hf_token)
        model_info = model_downloader.get_model_info()
        print(json.dumps(model_info, indent=2))
        return
    
    if args.download_models_only:
        print("Downloading models only...")
        model_downloader = ModelDownloader(args.cache_dir, hf_token)
        if model_downloader.download_all_models(low_memory=args.low_mem):
            print("✅ Models downloaded successfully!")
        else:
            print("❌ Failed to download some models")
        return
    
    # Setup logging
    setup_logging(args.log_level)
    
    if not MTEB_AVAILABLE:
        print("Error: MTEB not available. Install with: pip install mteb")
        return
    
    # Determine LIMIT dataset path
    if args.limit_path:
        limit_path = Path(args.limit_path)
    else:
        # Try to find LIMIT dataset automatically
        possible_paths = [
            Path("./limit/limit-small"),
            Path("./limit"),
            Path("../limit"),
            Path("../limit_dataset"),
            Path("benchmark/limit"),
            Path("benchmark/limit/limit-small")
        ]
        
        limit_path = None
        for path in possible_paths:
            if path.exists():
                limit_path = path
                break
        
        # If not found, try to download it
        if not limit_path:
            print("LIMIT dataset not found. Attempting to download...")
            # Use absolute path to ensure dataset downloads inside benchmark folder
            downloader = LIMITDatasetDownloader(str(Path(__file__).parent / "limit_dataset"))
            
            # Try to download limit-small first (smaller dataset)
            if downloader.download_dataset("limit-small"):
                limit_path = Path(__file__).parent / "limit_dataset" / "limit-small"
                print(f"Successfully downloaded LIMIT-small dataset to: {limit_path}")
            else:
                # Try to download full limit dataset as fallback
                print("Failed to download limit-small, trying full limit dataset...")
                if downloader.download_dataset("limit"):
                    limit_path = Path(__file__).parent / "limit_dataset" / "limit"
                    print(f"Successfully downloaded LIMIT dataset to: {limit_path}")
                else:
                    print("Error: Failed to download LIMIT dataset automatically")
                    print("Please download it manually using:")
                    print("  python download_limit_dataset.py --dataset limit-small")
                    return
    
    print(f"Using LIMIT dataset from: {limit_path}")
    
    # Load and verify dataset
    try:
        queries, corpus, qrels = load_local_limit_dataset(limit_path)
        
        if len(queries) == 0 or len(corpus) == 0:
            print("Error: No queries or corpus found in dataset")
            return
            
        # Apply sample size if specified
        if args.sample_size:
            corpus = corpus[:args.sample_size]
            print(f"Using sample of {len(corpus)} documents")
    
    except Exception as e:
        print(f"Error loading LIMIT dataset: {e}")
        return
    
    print("Initializing TriStage-RAG model for MTEB evaluation...")
    
    # Check and download models if needed
    print("Checking model availability...")
    model_downloader = ModelDownloader(args.cache_dir, hf_token)
    
    if not model_downloader.ensure_models_available(low_memory=args.low_mem):
        print("Error: Failed to ensure models are available")
        return
    
    # Show model information
    model_info = model_downloader.get_model_info()
    print(f"Models directory: {model_info['models_dir']}")
    print(f"Total model size: {model_info['total_size_mb']} MB")
    for stage, info in model_info['models'].items():
        status = "✅ Complete" if info['complete'] else "❌ Incomplete"
        print(f"  {stage}: {info['name']} - {status}")
    
    # Create the model
    # Optional low-memory overrides
    pipeline_overrides = {}
    if args.low_mem:
        pipeline_overrides = {
            "stage1_model": "sentence-transformers/all-MiniLM-L6-v2",
            "stage1_batch_size": 32,
            "stage1_top_k": 200,
            "stage1_use_fp16": False,
            "stage2_model": "sentence-transformers/all-MiniLM-L6-v2",  # fallback for rescoring if needed
            "stage2_batch_size": 8,
            "stage2_top_k": 50,
            "stage2_use_fp16": False,
            "stage3_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # compact CE
            "stage3_batch_size": 16,
            "stage3_top_k": 10,
            "stage3_use_fp16": False,
        }
    if args.stage1_model:
        pipeline_overrides["stage1_model"] = args.stage1_model

    model = TriStageMTEBModel(
        device=args.device,
        cache_dir=args.cache_dir,
        index_dir=args.index_dir,
        pipeline_config=pipeline_overrides if pipeline_overrides else None
    )
    
    print(f"Model created: {model}")
    print(f"Pipeline info: {model.get_pipeline_info()}")
    
    # Create task objects
    mteb_tasks = []
    for task_name in args.tasks:
        if task_name == "LIMITSmallRetrieval":
            mteb_tasks.append(LIMITSmallRetrieval())
        elif task_name == "LIMITRetrieval":
            mteb_tasks.append(LIMITRetrieval())
        else:
            print(f"Warning: Unknown task {task_name}")
    
    if not mteb_tasks:
        print("Error: No valid tasks specified")
        return
    
    print(f"Tasks to evaluate: {[task.metadata.name for task in mteb_tasks]}")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    try:
        results = evaluate_with_mteb(
            model=model,
            tasks=mteb_tasks,
            output_folder=str(output_path)
        )
        
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {output_path}")
        
        # Print summary (supports MTEB v2 returning list or dict)
        print("\nSummary of results:")
        def _print_entry(name: str, score_val):
            try:
                print(f"  {name}: {float(score_val):.4f}")
            except Exception:
                print(f"  {name}: {score_val}")

        def _extract_main_score(entry):
            if not isinstance(entry, dict):
                return None
            # Try direct
            if isinstance(entry.get('main_score'), (int, float)):
                return entry['main_score']
            if isinstance(entry.get('ndcg_at_10'), (int, float)):
                return entry['ndcg_at_10']
            # Try nested scores by split
            scores = entry.get('scores')
            if isinstance(scores, dict) and scores:
                split = 'test' if 'test' in scores else next(iter(scores))
                split_scores = scores.get(split, {})
                if isinstance(split_scores, dict):
                    if isinstance(split_scores.get('ndcg_at_10'), (int, float)):
                        return split_scores['ndcg_at_10']
                    if isinstance(split_scores.get('main_score'), (int, float)):
                        return split_scores['main_score']
                    # fallback: first numeric
                    for v in split_scores.values():
                        if isinstance(v, (int, float)):
                            return v
            return None

        if isinstance(results, dict):
            for task_name, task_results in results.items():
                score_val = _extract_main_score(task_results) if isinstance(task_results, dict) else None
                if score_val is not None:
                    _print_entry(task_name, score_val)
                else:
                    print(f"  {task_name}: (see detailed scores in {output_path})")
        elif isinstance(results, list):
            for i, entry in enumerate(results):
                if isinstance(entry, dict):
                    name = entry.get('mteb_dataset_name') or entry.get('task_name') or entry.get('dataset_name') or entry.get('name') or f'Task_{i+1}'
                    score_val = _extract_main_score(entry)
                    if score_val is not None:
                        _print_entry(name, score_val)
                    else:
                        print(f"  {name}: (see detailed scores in {output_path})")
                else:
                    print(f"  Task_{i+1}: (see detailed scores in {output_path})")
        else:
            print(f"  (see detailed scores in {output_path})")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()