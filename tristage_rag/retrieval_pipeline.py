import os
import pickle
import yaml
import logging
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

# Import our stage modules
from .stage1_retriever import Stage1Retriever, Stage1Config
from .stage2_rescorer import ColBERTScorer, Stage2Config  
from .stage3_reranker import AdaptiveCrossEncoderReranker, Stage3Config

@dataclass
class PipelineConfig:
    """Configuration for the complete 3-stage retrieval pipeline.
    
    This dataclass holds all configurable parameters for the retrieval pipeline,
    including model settings, batch sizes, device selection, and performance options.
    
    Attributes:
        stage1_model (str): Model name for Stage 1 retriever (default: "google/embeddinggemma-300m").
        stage1_top_k (int): Number of top candidates from Stage 1 (default: 500).
        stage1_batch_size (int): Batch size for Stage 1 processing (default: 32).
        stage1_enable_bm25 (bool): Whether to enable BM25 fusion in Stage 1 (default: True).
        stage1_bm25_top_k (int): Top-k for BM25 in Stage 1 (default: 300).
        stage1_fusion_method (str): Fusion method for Stage 1 (default: "rrf").
        stage1_use_fp16 (bool): Use FP16 precision in Stage 1 (default: True).
        stage2_model (str): Model name for Stage 2 rescorer (default: "lightonai/GTE-ModernColBERT-v1").
        stage2_top_k (int): Number of top candidates from Stage 2 (default: 50).
        stage2_batch_size (int): Batch size for Stage 2 processing (default: 16).
        stage2_max_seq_length (int): Max sequence length for Stage 2 (default: 192).
        stage2_use_fp16 (bool): Use FP16 precision in Stage 2 (default: True).
        stage2_scoring_method (str): Scoring method for Stage 2 (default: "maxsim").
        stage3_model (str): Model name for Stage 3 reranker (default: "cross-encoder/ms-marco-MiniLM-L6-v2").
        stage3_top_k (int): Number of final top results from Stage 3 (default: 20).
        stage3_batch_size (int): Batch size for Stage 3 processing (default: 32).
        stage3_max_length (int): Max length for Stage 3 inputs (default: 256).
        stage3_use_fp16 (bool): Use FP16 precision in Stage 3 (default: True).
        stage3_early_exit_enabled (bool): Enable early exit in Stage 3 (default: True).
        stage3_early_exit_threshold (float): Score gap threshold for early exit (default: 0.15).
        device (str): Device for model inference ("auto", "cpu", "cuda", etc.) (default: "auto").
        cache_dir (str): Directory for model cache (default: "./models").
        index_dir (str): Directory for index storage (default: "./faiss_index").
        log_level (str): Logging level ("DEBUG", "INFO", etc.) (default: "INFO").
        log_file (str): Path to log file (default: "retrieval_pipeline.log").
        enable_timing (bool): Whether to track and log timing information (default: True).
        save_intermediate_results (bool): Whether to save results from each stage (default: False).
        auto_cleanup (bool): Whether to automatically clean up memory after queries (default: True).
    """
    
    # Stage 1 configuration
    stage1_model: str = "google/embeddinggemma-300m"
    stage1_top_k: int = 500
    stage1_batch_size: int = 32
    stage1_enable_bm25: bool = True
    stage1_bm25_top_k: int = 300
    stage1_fusion_method: str = "rrf"
    stage1_use_fp16: bool = True
    
    # Stage 2 configuration — top_k reduced from 100 to 50 (per SemEval 2026 finding)
    stage2_model: str = "lightonai/GTE-ModernColBERT-v1"
    stage2_top_k: int = 50
    stage2_batch_size: int = 16
    stage2_max_seq_length: int = 192
    stage2_use_fp16: bool = True
    stage2_scoring_method: str = "maxsim"
    
    # Stage 3 configuration
    stage3_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    stage3_top_k: int = 20
    stage3_batch_size: int = 32
    stage3_max_length: int = 256
    stage3_use_fp16: bool = True
    stage3_early_exit_enabled: bool = True
    stage3_early_exit_threshold: float = 0.15
    
    # General configuration
    device: str = "auto"
    cache_dir: str = "./models"
    index_dir: str = "./faiss_index"
    log_level: str = "INFO"
    log_file: str = "retrieval_pipeline.log"
    enable_timing: bool = True
    save_intermediate_results: bool = False
    
    # Memory optimization
    auto_cleanup: bool = True

class RetrievalPipeline:
    """Complete 3-stage retrieval pipeline for RAG systems.
    
    Orchestrates a three-stage retrieval process:
    1. Stage 1: Fast candidate generation using embeddings and BM25.
    2. Stage 2: Multi-vector rescoring with pre-computed ColBERT embeddings.
    3. Stage 3: Cross-encoder reranking with early exit for easy queries.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[PipelineConfig] = None):
        self.logger = logging.getLogger(__name__)
        
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = PipelineConfig()
        
        self._setup_logging()
        
        self.stage1 = None
        self.stage2 = None
        self.stage3 = None
        
        self.performance_stats = {
            "total_queries": 0,
            "avg_stage1_time": 0.0,
            "avg_stage2_time": 0.0,
            "avg_stage3_time": 0.0,
            "avg_total_time": 0.0,
            "stage_time_history": [],
            "early_exit_count": 0,
        }
        
        self.logger.info("RetrievalPipeline initialized")
    
    def _load_config(self, config_path: str) -> PipelineConfig:
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            pipeline_data = config_data.get('pipeline', {})
            
            return PipelineConfig(
                # Stage 1
                stage1_model=pipeline_data.get('stage1', {}).get('model', "google/embeddinggemma-300m"),
                stage1_top_k=pipeline_data.get('stage1', {}).get('top_k', 500),
                stage1_batch_size=pipeline_data.get('stage1', {}).get('batch_size', 32),
                stage1_enable_bm25=pipeline_data.get('stage1', {}).get('enable_bm25', True),
                stage1_bm25_top_k=pipeline_data.get('stage1', {}).get('bm25_top_k', 300),
                stage1_fusion_method=pipeline_data.get('stage1', {}).get('fusion_method', "rrf"),
                stage1_use_fp16=pipeline_data.get('stage1', {}).get('use_fp16', True),
                
                # Stage 2
                stage2_model=pipeline_data.get('stage2', {}).get('model', "lightonai/GTE-ModernColBERT-v1"),
                stage2_top_k=pipeline_data.get('stage2', {}).get('top_k', 50),
                stage2_batch_size=pipeline_data.get('stage2', {}).get('batch_size', 16),
                stage2_max_seq_length=pipeline_data.get('stage2', {}).get('max_seq_length', 192),
                stage2_use_fp16=pipeline_data.get('stage2', {}).get('use_fp16', True),
                stage2_scoring_method=pipeline_data.get('stage2', {}).get('scoring_method', "maxsim"),
                
                # Stage 3
                stage3_model=pipeline_data.get('stage3', {}).get('model', "cross-encoder/ms-marco-MiniLM-L6-v2"),
                stage3_top_k=pipeline_data.get('stage3', {}).get('top_k', 20),
                stage3_batch_size=pipeline_data.get('stage3', {}).get('batch_size', 32),
                stage3_max_length=pipeline_data.get('stage3', {}).get('max_length', 256),
                stage3_use_fp16=pipeline_data.get('stage3', {}).get('use_fp16', True),
                stage3_early_exit_enabled=pipeline_data.get('stage3', {}).get('early_exit_enabled', True),
                stage3_early_exit_threshold=pipeline_data.get('stage3', {}).get('early_exit_threshold', 0.15),
                
                # General
                device=pipeline_data.get('device', "auto"),
                cache_dir=pipeline_data.get('cache_dir', "./models"),
                index_dir=pipeline_data.get('index_dir', "./faiss_index"),
                log_level=pipeline_data.get('log_level', "INFO"),
                log_file=pipeline_data.get('log_file', "retrieval_pipeline.log"),
                enable_timing=pipeline_data.get('enable_timing', True),
                save_intermediate_results=pipeline_data.get('save_intermediate_results', False),
                auto_cleanup=pipeline_data.get('auto_cleanup', True),
            )
            
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return PipelineConfig()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler()
            ]
        )
    
    def initialize_stages(self):
        """Initialize all three stages of the pipeline"""
        self.logger.info("Initializing pipeline stages...")
        
        try:
            # Stage 1: Fast Candidate Generation
            stage1_config = Stage1Config(
                model_name=self.config.stage1_model,
                device=self.config.device,
                cache_dir=self.config.cache_dir,
                index_dir=self.config.index_dir,
                top_k_candidates=self.config.stage1_top_k,
                batch_size=self.config.stage1_batch_size,
                enable_bm25=self.config.stage1_enable_bm25,
                bm25_top_k=self.config.stage1_bm25_top_k,
                fusion_method=self.config.stage1_fusion_method,
                use_fp16=self.config.stage1_use_fp16
            )
            self.stage1 = Stage1Retriever(stage1_config)
            self.logger.info("Stage 1 initialized")
            
            # Stage 2: Multi-Vector Rescoring (pre-computed embeddings)
            stage2_config = Stage2Config(
                model_name=self.config.stage2_model,
                device=self.config.device,
                cache_dir=self.config.cache_dir,
                max_seq_length=self.config.stage2_max_seq_length,
                batch_size=self.config.stage2_batch_size,
                top_k_candidates=self.config.stage2_top_k,
                use_fp16=self.config.stage2_use_fp16,
                scoring_method=self.config.stage2_scoring_method,
            )
            self.stage2 = ColBERTScorer(stage2_config)
            self.logger.info("Stage 2 initialized")
            
            # Stage 3: Cross-Encoder Reranking (with early exit)
            stage3_config = Stage3Config(
                model_name=self.config.stage3_model,
                device=self.config.device,
                cache_dir=self.config.cache_dir,
                max_length=self.config.stage3_max_length,
                batch_size=self.config.stage3_batch_size,
                top_k_final=self.config.stage3_top_k,
                use_fp16=self.config.stage3_use_fp16,
                early_exit_enabled=self.config.stage3_early_exit_enabled,
                early_exit_threshold=self.config.stage3_early_exit_threshold,
            )
            self.stage3 = AdaptiveCrossEncoderReranker(stage3_config)
            self.logger.info("Stage 3 initialized")
            
            self.logger.info("All pipeline stages initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing pipeline stages: {e}")
            raise
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Add documents to the pipeline index.

        Stage 1 (FAISS + BM25) and Stage 2 (ColBERT pre-encoding) run
        **in parallel** since they use independent models with no shared state.
        """
        if not self.stage1:
            self.initialize_stages()
        
        self.logger.info(f"Adding {len(documents)} documents to pipeline")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _stage1_index():
            self.stage1.add_documents(documents, metadata)

        def _stage2_precompute(start_idx: int):
            doc_ids = list(range(start_idx, start_idx + len(documents)))
            self.stage2.add_documents(doc_ids, documents)

        # We need start_idx *before* Stage 1 runs (it appends to self.documents).
        # So compute it from current length, then run both in parallel.
        start_idx = len(self.stage1.documents)

        try:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest") as pool:
                f1 = pool.submit(_stage1_index)
                f2 = pool.submit(_stage2_precompute, start_idx)

                # Wait for both; propagate first error
                for future in as_completed([f1, f2]):
                    future.result()  # raises if failed

            self.logger.info("Documents added successfully (Stage 1 + Stage 2 in parallel)")
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            raise
    
    def search(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Execute complete 3-stage search pipeline.
        
        1. Stage 1: Fast candidate generation (dense + BM25).
        2. Stage 2: Multi-vector rescoring (uses pre-computed embeddings).
        3. Stage 3: Cross-encoder reranking (with early exit).
        """
        if not self.stage1 or not self.stage2 or not self.stage3:
            self.initialize_stages()
        
        top_k = top_k or self.config.stage3_top_k
        
        total_start = self._get_timing(self.config.enable_timing)
        
        try:
            self.logger.info(f"Starting 3-stage search for query: '{query[:100]}...'")
            
            # Stage 1: Fast Candidate Generation
            stage1_start = self._get_timing(self.config.enable_timing)
            stage1_results = self.stage1.search(query, self.config.stage1_top_k)
            stage1_time = time.time() - stage1_start if stage1_start else None
            
            self.logger.info(f"Stage 1 completed: {len(stage1_results)} candidates")
            
            if not stage1_results:
                return {
                    "query": query,
                    "results": [],
                    "stage1_results": [],
                    "stage2_results": [],
                    "timing": self._calculate_timing(total_start, stage1_time, None, None),
                    "performance_stats": self.performance_stats
                }
            
            # Stage 2: Multi-Vector Rescoring (pre-computed embeddings)
            stage2_start = self._get_timing(self.config.enable_timing)
            stage2_results = self.stage2.rescore_candidates(query, stage1_results)
            stage2_time = time.time() - stage2_start if stage2_start else None
            
            self.logger.info(f"Stage 2 completed: {len(stage2_results)} rescored candidates")
            
            if not stage2_results:
                return {
                    "query": query,
                    "results": [],
                    "stage1_results": stage1_results,
                    "stage2_results": [],
                    "timing": self._calculate_timing(total_start, stage1_time, stage2_time, None),
                    "performance_stats": self.performance_stats
                }
            
            # Stage 3: Cross-Encoder Reranking (with early exit)
            stage3_start = self._get_timing(self.config.enable_timing)
            final_results = self.stage3.rerank(query, stage2_results)
            stage3_time = time.time() - stage3_start if stage3_start else None

            # Read the explicit early-exit flag set by the reranker. Previously
            # this was inferred from result contents (`not any("stage3_score"
            # in r ...)`), which falsely reported an early exit whenever Stage 3
            # returned [] for an unrelated reason (e.g. predict() raised).
            early_exited = getattr(self.stage3, "last_early_exit", False)
            
            # Apply final top-k filter
            final_results = final_results[:top_k]
            
            total_time = time.time() - total_start if total_start else None
            
            self.logger.info(
                f"Stage 3 completed: {len(final_results)} final results"
                + (" (early exit)" if early_exited else "")
            )
            
            # Update performance stats
            if self.config.enable_timing:
                self._update_performance_stats(
                    stage1_time, stage2_time, stage3_time, total_time,
                    early_exited=early_exited,
                )
            
            result = {
                "query": query,
                "results": final_results,
                "stage1_results": stage1_results if self.config.save_intermediate_results else [],
                "stage2_results": stage2_results if self.config.save_intermediate_results else [],
                "timing": self._calculate_timing(total_start, stage1_time, stage2_time, stage3_time),
                "performance_stats": self.performance_stats.copy(),
                "early_exited": early_exited,
            }
            
            if self.config.auto_cleanup:
                self._cleanup_memory()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            raise
    
    def batch_search(self, queries: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Execute batch search for multiple queries."""
        results = []
        for query in queries:
            result = self.search(query, top_k)
            results.append(result)
        return results
    
    def save_index(self, index_path: Optional[str] = None):
        """Save pipeline index to disk.

        Persists Stage 1 (FAISS + BM25) **and** Stage 2 (ColBERT token
        embeddings) so that a loaded index is immediately queryable without
        re-encoding documents.
        """
        if not self.stage1:
            raise ValueError("Pipeline not initialized")
        
        if index_path is None:
            index_path = os.path.join(self.config.index_dir, "pipeline_index.pkl")
        
        # Stage 1 handles its own persistence (FAISS index + pickle)
        self.stage1.save_index(index_path)

        # Persist Stage 2 ColBERT embeddings alongside the Stage 1 index
        if self.stage2 and self.stage2.has_cached_docs():
            colbert_path = os.path.join(self.config.index_dir, "stage2_colbert_cache.pkl")
            # Convert tensors to numpy for efficient pickling. Handles both
            # plain tensors and int8 tuples (quantized, scale).
            cache_data = {}
            for doc_id, (emb, seq_len) in self.stage2.export_cache().items():
                if isinstance(emb, tuple):
                    # int8 quantized: (int8_tensor, scale_vec)
                    cache_data[doc_id] = (
                        (emb[0].numpy(), emb[1].numpy()),
                        seq_len,
                    )
                else:
                    cache_data[doc_id] = (emb.numpy(), seq_len)
            with open(colbert_path, "wb") as f:
                pickle.dump(cache_data, f)
            self.logger.info(f"Stage 2 ColBERT embeddings saved ({len(cache_data)} docs)")

        self.logger.info(f"Pipeline index saved to {index_path}")
    
    def load_index(self, index_path: Optional[str] = None):
        """Load pipeline index from disk.

        Restores Stage 1 (FAISS + BM25) and, if available, Stage 2 ColBERT
        token embeddings so the pipeline is immediately queryable.
        """
        if not self.stage1:
            self.initialize_stages()
        
        if index_path is None:
            index_path = os.path.join(self.config.index_dir, "pipeline_index.pkl")
        
        self.stage1.load_index(index_path)

        # Restore Stage 2 ColBERT embeddings if persisted
        if self.stage2:
            colbert_path = os.path.join(self.config.index_dir, "stage2_colbert_cache.pkl")
            if os.path.exists(colbert_path):
                import pickle as _pickle
                with open(colbert_path, "rb") as f:
                    cache_data = _pickle.load(f)
                # Convert numpy back to torch tensors — handles int8 tuples too
                restored = {}
                for doc_id, (emb_data, seq_len) in cache_data.items():
                    if isinstance(emb_data, tuple) and len(emb_data) == 2 and hasattr(emb_data[0], 'dtype') and emb_data[0].dtype == 'int8':
                        # Int8 quantized: (int8_numpy, scale_numpy)
                        restored[doc_id] = (
                            (torch.from_numpy(emb_data[0]), torch.from_numpy(emb_data[1])),
                            seq_len,
                        )
                    else:
                        restored[doc_id] = (torch.from_numpy(emb_data), seq_len)
                self.stage2.import_cache(restored)
                self.logger.info(
                    f"Stage 2 ColBERT embeddings restored ({len(restored)} docs)"
                )
            else:
                self.logger.warning(
                    "No ColBERT cache found — Stage 2 will encode on-the-fly until "
                    "documents are re-added via add_documents()"
                )

        self.logger.info(f"Pipeline index loaded from {index_path}")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the pipeline."""
        info = {
            "config": asdict(self.config),
            "stages_initialized": {
                "stage1": self.stage1 is not None,
                "stage2": self.stage2 is not None,
                "stage3": self.stage3 is not None
            },
            "performance_stats": self.performance_stats
        }
        
        if self.stage1:
            info["stage1_stats"] = self.stage1.get_stats()
        if self.stage2:
            info["stage2_info"] = self.stage2.get_model_info()
        if self.stage3:
            info["stage3_info"] = self.stage3.get_model_info()
        
        return info
    
    def _get_timing(self, enable_timing: bool) -> Optional[float]:
        return time.time() if enable_timing else None
    
    def _calculate_timing(self, total_start: Optional[float], stage1_time: Optional[float], 
                         stage2_time: Optional[float], stage3_time: Optional[float]) -> Dict[str, float]:
        if not self.config.enable_timing:
            return {}
        
        total_time = time.time() - total_start if total_start else None
        
        return {
            "stage1_time": stage1_time or 0.0,
            "stage2_time": stage2_time or 0.0,
            "stage3_time": stage3_time or 0.0,
            "total_time": total_time or 0.0
        }
    
    def _update_performance_stats(self, stage1_time: float, stage2_time: float,
                                   stage3_time: float, total_time: float,
                                   early_exited: bool = False):
        self.performance_stats["total_queries"] += 1
        
        alpha = 1.0 / self.performance_stats["total_queries"]
        
        self.performance_stats["avg_stage1_time"] = (
            (1 - alpha) * self.performance_stats["avg_stage1_time"] + alpha * stage1_time
        )
        self.performance_stats["avg_stage2_time"] = (
            (1 - alpha) * self.performance_stats["avg_stage2_time"] + alpha * stage2_time
        )
        self.performance_stats["avg_stage3_time"] = (
            (1 - alpha) * self.performance_stats["avg_stage3_time"] + alpha * stage3_time
        )
        self.performance_stats["avg_total_time"] = (
            (1 - alpha) * self.performance_stats["avg_total_time"] + alpha * total_time
        )
        
        if early_exited:
            self.performance_stats["early_exit_count"] += 1
        
        self.performance_stats["stage_time_history"].append({
            "stage1": stage1_time,
            "stage2": stage2_time,
            "stage3": stage3_time,
            "total": total_time,
            "early_exited": early_exited,
        })
        
        if len(self.performance_stats["stage_time_history"]) > 100:
            self.performance_stats["stage_time_history"] = self.performance_stats["stage_time_history"][-100:]
    
    def _cleanup_memory(self):
        try:
            if self.stage2:
                self.stage2.clear_gpu_memory()
            if self.stage3:
                self.stage3.clear_gpu_memory()
        except Exception as e:
            self.logger.warning(f"Error during memory cleanup: {e}")
    
    def export_config(self, config_path: str):
        config_dict = {
            "pipeline": asdict(self.config)
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        self.logger.info(f"Configuration exported to {config_path}")
    
    def __del__(self):
        try:
            self._cleanup_memory()
        except Exception:
            pass
