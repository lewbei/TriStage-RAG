#!/usr/bin/env python3
"""
Main TriStage-RAG Benchmark Runner
Single script to handle the complete workflow: dataset download, model download, and evaluation
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Import required modules (the repo is pip-installable via `pip install -e .[all]`;
# no sys.path manipulation needed).
from benchmark.download_limit_dataset import LIMITDatasetDownloader
from benchmark.download_models import ModelDownloader
from benchmark.config_loader import BenchmarkConfig

def setup_logging(level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main orchestration function"""
    # Load .env file if it exists
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    
    # Load benchmark configuration
    try:
        config = BenchmarkConfig()
        print(f"Loaded benchmark configuration from: {config.config_path}")
    except Exception as e:
        print(f"ERROR: Failed to load benchmark configuration: {e}")
        return
    
    # Setup logging
    setup_logging(config.get_log_level())
    
    print("TriStage-RAG Benchmark Runner")
    print("=" * 50)
    print(f"Configuration: {config}")
    print(f"Device: {config.get_device()}")
    print(f"Low Memory Mode: {config.is_low_memory_mode()}")
    
    # Step 1: Download LIMIT datasets if needed
    print("\nStep 1: Checking LIMIT datasets...")
    # Use benchmark-relative path for dataset storage
    benchmark_dir = Path(__file__).parent / "benchmark"
    dataset_path = benchmark_dir / config.get_dataset_path()
    # Ensure folder exists
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # If a legacy dataset folder exists at repo root, migrate it into benchmark
    legacy_top_level = Path(__file__).parent.parent / "limit_dataset"
    if legacy_top_level.exists():
        try:
            import shutil
            print(f"Found legacy dataset at {legacy_top_level}. Merging into {dataset_path} and removing legacy folder...")

            def _merge_tree(src: Path, dst: Path):
                dst.mkdir(parents=True, exist_ok=True)
                for child in src.iterdir():
                    dst_child = dst / child.name
                    if child.is_dir():
                        _merge_tree(child, dst_child)
                    else:
                        if not dst_child.exists():
                            child.replace(dst_child)

            _merge_tree(legacy_top_level, dataset_path)

            # Remove legacy folder after merge
            shutil.rmtree(legacy_top_level, ignore_errors=True)
            print("Legacy dataset folder removed.")
        except Exception as e:
            print(f"Warning: failed to migrate/remove legacy dataset folder: {e}")
    limit_small_path = dataset_path / "limit-small"
    limit_full_path = dataset_path / "limit"
    
    # Determine which datasets are needed based on configured tasks
    tasks = config.get_tasks()
    need_small = "LIMITSmallRetrieval" in tasks
    need_full = "LIMITRetrieval" in tasks
    
    print(f"Configured tasks: {tasks}")
    print(f"Need small dataset: {need_small}")
    print(f"Need full dataset: {need_full}")
    
    if config.get("benchmark.dataset.auto_download", True):
        # Use absolute path to ensure dataset downloads inside benchmark folder
        dataset_downloader = LIMITDatasetDownloader(str(dataset_path.absolute()))
        
        # Download small dataset if needed
        if need_small and not limit_small_path.exists():
            print("LIMIT-small dataset not found, downloading...")
            if dataset_downloader.download_dataset("limit-small"):
                print("SUCCESS: LIMIT-small dataset downloaded successfully")
            else:
                print("ERROR: Failed to download LIMIT-small dataset")
                return
        elif need_small:
            print("SUCCESS: LIMIT-small dataset already exists")
        
        # Download full dataset if needed
        if need_full and not limit_full_path.exists():
            print("LIMIT-full dataset not found, downloading...")
            if dataset_downloader.download_dataset("limit"):
                print("SUCCESS: LIMIT-full dataset downloaded successfully")
            else:
                print("ERROR: Failed to download LIMIT-full dataset")
                return
        elif need_full:
            print("SUCCESS: LIMIT-full dataset already exists")
    else:
        print("Auto-download disabled, skipping dataset download")
    
    # Step 2: Download models if needed
    print("\nStep 2: Checking models...")
    model_downloader = ModelDownloader(config.get_cache_dir())
    
    if config.get("benchmark.models.auto_download", True):
        if not model_downloader.ensure_models_available(low_memory=config.is_low_memory_mode()):
            print("ERROR: Failed to ensure models are available")
            return
    
    # Show model information
    model_info = model_downloader.get_model_info()
    print(f"Models directory: {model_info['models_dir']}")
    print(f"Total model size: {model_info['total_size_mb']} MB")
    for stage, info in model_info['models'].items():
        status = "COMPLETE" if info['complete'] else "INCOMPLETE"
        print(f"  {stage}: {info['name']} - {status}")
    
    # Step 3: Run benchmark
    print("\nStep 3: Running benchmark...")
    
    # Import benchmark modules after path setup
    from benchmark.tristage_mteb_model import TriStageMTEBModel
    from benchmark.limit_mteb_tasks import LIMITSmallRetrieval
    
    try:
        import mteb
        from mteb import MTEB
        MTEB_AVAILABLE = True
    except ImportError:
        print("ERROR: MTEB not available. Install with: pip install mteb")
        return
    
    # Create model with configuration overrides
    model = TriStageMTEBModel(
        device=config.get_device(),
        cache_dir=config.get_cache_dir(),
        index_dir=config.get_index_dir(),
        pipeline_config=config.get_pipeline_overrides()
    )
    
    # Create tasks based on configuration
    tasks = []
    for task_name in config.get_tasks():
        if task_name == "LIMITSmallRetrieval":
            tasks.append(LIMITSmallRetrieval())
        elif task_name == "LIMITRetrieval":
            # Import here to avoid issues if not available
            from benchmark.limit_mteb_tasks import LIMITRetrieval
            tasks.append(LIMITRetrieval())
        else:
            print(f"WARNING: Unknown task {task_name}")
    
    if not tasks:
        print("ERROR: No valid tasks specified")
        return
    
    # Create output directory
    output_path = Path(config.get_output_dir())
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    try:
        evaluation = MTEB(tasks=tasks)
        print(f"Starting MTEB evaluation on {len(tasks)} tasks...")
        print(f"Tasks: {[task.metadata.name for task in tasks]}")
        
        # Get evaluation parameters from config
        encode_kwargs = config.get("benchmark.evaluation.encode_kwargs", {'batch_size': 32})
        overwrite_results = config.get("benchmark.evaluation.overwrite_results", True)
        
        results = evaluation.run(
            model,
            output_folder=str(output_path),
            encode_kwargs=encode_kwargs,
            overwrite_results=overwrite_results
        )
        
        print("\nSUCCESS: Benchmark completed successfully!")
        print(f"Results saved to: {output_path}")
        
        # Print summary
        if isinstance(results, dict):
            for task_name, task_results in results.items():
                if isinstance(task_results, dict):
                    main_score = task_results.get('main_score')
                    if main_score is not None:
                        print(f"  {task_name}: {main_score:.4f}")
                    else:
                        print(f"  {task_name}: (see detailed scores in {output_path})")
        elif isinstance(results, list):
            for i, entry in enumerate(results):
                if isinstance(entry, dict):
                    name = entry.get('mteb_dataset_name') or entry.get('task_name') or f'Task_{i+1}'
                    main_score = entry.get('main_score')
                    if main_score is not None:
                        print(f"  {name}: {main_score:.4f}")
                    else:
                        print(f"  {name}: (see detailed scores in {output_path})")
        
        # Print sample results
        print("\n" + "="*60)
        print("SAMPLE RETRIEVAL RESULTS")
        print("="*60)
        
        # Try to load and display some sample results
        try:
            # Load predictions file if it exists
            predictions_file = output_path / "model_predictions.json"
            if predictions_file.exists():
                import json
                with open(predictions_file, 'r', encoding='utf-8') as f:
                    predictions = json.load(f)
                
                print(f"Loaded predictions from: {predictions_file}")
                
                # Show sample queries and their top results
                sample_count = min(3, len(predictions))
                print(f"\nShowing {sample_count} sample queries with top results:\n")
                
                for i, (query_id, query_data) in enumerate(list(predictions.items())[:sample_count]):
                    print(f"--- Sample {i+1} ---")
                    print(f"Query ID: {query_id}")
                    
                    # Get the actual query text from the dataset
                    try:
                        from benchmark.limit_mteb_tasks import LIMITSmallRetrieval
                        task = LIMITSmallRetrieval()
                        queries = task.queries
                        if query_id in queries:
                            print(f"Query: {queries[query_id]}")
                        else:
                            print(f"Query: [ID: {query_id}]")
                    except:
                        print(f"Query: [ID: {query_id}]")
                    
                    # Show top results
                    if 'scores' in query_data:
                        scores = query_data['scores']
                        print(f"Top {min(5, len(scores))} results:")
                        
                        for j, (doc_id, score) in enumerate(list(scores.items())[:5]):
                            # Get document text if available
                            try:
                                corpus = task.corpus
                                if doc_id in corpus:
                                    doc_text = corpus[doc_id].get('text', '')
                                    doc_preview = doc_text[:150] + "..." if len(doc_text) > 150 else doc_text
                                    print(f"  {j+1}. Doc ID: {doc_id}")
                                    print(f"     Score: {score:.4f}")
                                    print(f"     Text: {doc_preview}")
                                else:
                                    print(f"  {j+1}. Doc ID: {doc_id}, Score: {score:.4f}")
                            except:
                                print(f"  {j+1}. Doc ID: {doc_id}, Score: {score:.4f}")
                    
                    print()
            else:
                print("No predictions file found. Showing raw results structure...")
                # Try to show some information about the results structure
                if isinstance(results, dict):
                    for task_name, task_results in results.items():
                        print(f"\nTask: {task_name}")
                        if isinstance(task_results, dict):
                            print(f"  Keys: {list(task_results.keys())}")
                            if 'scores' in task_results:
                                print(f"  Scores structure: {type(task_results['scores'])}")
                                if isinstance(task_results['scores'], dict):
                                    print(f"  Score splits: {list(task_results['scores'].keys())}")
        except Exception as e:
            print(f"Could not load sample results: {e}")
            print("Check the output directory for detailed result files.")
        
    except Exception as e:
        print(f"ERROR: Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()