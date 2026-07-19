#!/usr/bin/env python3
"""
MTEB-compatible wrapper for TriStage-RAG pipeline
Implements the MTEB model interface for 3-stage retrieval evaluation
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import sys
from dataclasses import dataclass

from tristage_rag.retrieval_pipeline import RetrievalPipeline, PipelineConfig
from tristage_rag.stage3_reranker import AdaptiveCrossEncoderReranker, Stage3Config

@dataclass
class ModelCard:
    model_name: str = "TriStage-RAG"
    name: str = "TriStage-RAG"
    description: str = "3-stage retrieval pipeline with embedding, ColBERT, and cross-encoder models"
    architecture: str = "TriStage-RAG"
    framework: list = None
    model_type: str = "retrieval"
    languages: list = None
    language: list = None
    base_model_revision: str = "main"
    release_date: str = "2025-09-08"
    version: str = "1.0.0"

    def __post_init__(self):
        if self.framework is None:
            self.framework = ["PyTorch", "SentenceTransformers", "FAISS"]
        if self.languages is None:
            self.languages = ["eng-Latn"]
        if self.language is None:
            self.language = ["eng-Latn"]

class TriStageMTEBModel:
    """
    MTEB-compatible wrapper for the 3-stage TriStage-RAG pipeline.
    
    This class implements the MTEB model interface to enable evaluation
    of the complete 3-stage pipeline on MTEB tasks including LIMIT.
    """
    
    def __init__(self, 
                 pipeline_config: Optional[Dict[str, Any]] = None,
                 device: str = "auto",
                 cache_dir: str = "./models",
                 index_dir: str = "./faiss_index"):
        """
        Initialize the TriStage-RAG model for MTEB evaluation.
        
        Args:
            pipeline_config: Optional pipeline configuration
            device: Device to run models on ("auto", "cpu", "cuda")
            cache_dir: Directory for model cache
            index_dir: Directory for FAISS index storage
        """
        self.logger = logging.getLogger(__name__)

        # Normalize cache_dir and index_dir relative to repo root (parent of benchmark)
        repo_root = Path(__file__).parent.parent
        if not Path(cache_dir).is_absolute():
            cache_dir = str((repo_root / cache_dir).resolve())
        if not Path(index_dir).is_absolute():
            index_dir = str((repo_root / index_dir).resolve())

        # Initialize the 3-stage pipeline with normalized paths
        if pipeline_config:
            # Update the pipeline config to use normalized paths
            pipeline_config['cache_dir'] = cache_dir
            pipeline_config.setdefault('index_dir', index_dir)
            config = PipelineConfig(**pipeline_config)
            self.pipeline = RetrievalPipeline(config=config)
        else:
            config = PipelineConfig(cache_dir=cache_dir, index_dir=index_dir)
            self.pipeline = RetrievalPipeline(config=config)

        # Override device setting if provided
        if device != "auto":
            self.pipeline.config.device = device

        # MTEB requires these attributes
        self.similarity_metric_name = "cosine"
        # MTEB v2 expects similarity_fn_name for metadata
        self.similarity_fn_name = "cosine"
        self.max_seq_length = 512  # Will be adjusted per stage

        # Cache for encoded documents to avoid re-encoding
        self._document_cache = {}
        self._query_cache = {}
        # Map internal stage1 indices -> external corpus IDs (strings)
        self._doc_id_map = {}

        # Model card data to eliminate warning
        self.model_card_data = ModelCard()

        self.logger.info("TriStageMTEBModel initialized with top-level model path")
        # Proactively initialize stages lazily upon first encode/search
    
    def encode(self, 
               sentences: List[str], 
               task_name: str = "",
               **kwargs) -> np.ndarray:
        """
        Encode sentences using the 3-stage pipeline.
        
        For retrieval tasks, this handles both corpus and query encoding.
        For other tasks, uses Stage 1 embeddings.
        
        Args:
            sentences: List of sentences to encode
            task_name: MTEB task name (used to determine encoding strategy)
            **kwargs: Additional arguments including batch_size, prompt_name
            
        Returns:
            numpy array of embeddings
        """
        if not sentences:
            return np.array([])
        
        # Determine if this is corpus or query encoding
        is_corpus = self._is_corpus_encoding(task_name, kwargs)
        
        if is_corpus:
            return self._encode_corpus(sentences, task_name, **kwargs)
        else:
            return self._encode_queries(sentences, task_name, **kwargs)
    
    def _is_corpus_encoding(self, task_name: str, kwargs: Dict[str, Any]) -> bool:
        """Determine if current encoding is for corpus or queries"""
        # Check task name hints
        corpus_keywords = ["corpus", "document", "passage"]
        query_keywords = ["query", "question"]
        
        task_lower = task_name.lower()
        for keyword in corpus_keywords:
            if keyword in task_lower:
                return True
        
        for keyword in query_keywords:
            if keyword in task_lower:
                return False
        
        # Check prompt_name if available
        prompt_name = kwargs.get("prompt_name", "").lower()
        if prompt_name:
            for keyword in corpus_keywords:
                if keyword in prompt_name:
                    return True
            for keyword in query_keywords:
                if keyword in prompt_name:
                    return False
        
        # Default to corpus for retrieval tasks, queries otherwise
        return "retrieval" in task_name.lower()
    
    def _encode_corpus(self, documents: List[str], task_name: str, **kwargs) -> np.ndarray:
        """Encode corpus documents using Stage 1 embeddings"""
        self.logger.info(f"Encoding {len(documents)} corpus documents for task: {task_name}")
        
        # Check cache first
        cache_key = f"corpus_{task_name}_{hash(str(documents[:10]))}"
        if cache_key in self._document_cache:
            return self._document_cache[cache_key]
        
        # Ensure stages are initialized
        if not getattr(self.pipeline, 'stage1', None):
            try:
                self.pipeline.initialize_stages()
            except Exception as e:
                self.logger.warning(f"Stage initialization failed during corpus encode: {e}")
                raise

        # Add documents to pipeline if not already indexed
        self._ensure_documents_indexed(documents)
        
        # Get embeddings from Stage 1
        if hasattr(self.pipeline, 'stage1') and self.pipeline.stage1:
            batch_size = kwargs.get('batch_size', self.pipeline.config.stage1_batch_size)
            
            embeddings = self.pipeline.stage1.model.encode(
                documents,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings=True  # Important for cosine similarity
            )
            
            # Cache the result
            self._document_cache[cache_key] = embeddings
            
            return embeddings
        else:
            raise ValueError("Pipeline Stage 1 not initialized")
    
    def _encode_queries(self, queries: List[str], task_name: str, **kwargs) -> np.ndarray:
        """Encode queries using Stage 1 embeddings"""
        self.logger.info(f"Encoding {len(queries)} queries for task: {task_name}")
        
        # Check cache first
        cache_key = f"query_{task_name}_{hash(str(queries[:10]))}"
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        # Ensure stages are initialized
        if not getattr(self.pipeline, 'stage1', None):
            try:
                self.pipeline.initialize_stages()
            except Exception as e:
                self.logger.warning(f"Stage initialization failed during query encode: {e}")
                raise

        # Get embeddings from Stage 1
        if hasattr(self.pipeline, 'stage1') and self.pipeline.stage1:
            batch_size = kwargs.get('batch_size', self.pipeline.config.stage1_batch_size)
            
            embeddings = self.pipeline.stage1.model.encode(
                queries,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings=True  # Important for cosine similarity
            )
            
            # Cache the result
            self._query_cache[cache_key] = embeddings
            
            return embeddings
        else:
            raise ValueError("Pipeline Stage 1 not initialized")
    
    def _ensure_documents_indexed(self, documents: List[str]):
        """Ensure documents are indexed in the pipeline"""
        if not hasattr(self.pipeline, 'stage1') or not self.pipeline.stage1:
            try:
                self.pipeline.initialize_stages()
            except Exception as e:
                self.logger.error(f"Failed to initialize stages for indexing: {e}")
                raise
        
        # Check if documents are already indexed
        current_docs = getattr(self.pipeline.stage1, 'documents', [])
        if len(current_docs) == 0:
            self.logger.info("Indexing documents in pipeline")
            self.pipeline.add_documents(documents)
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               task_name: str = "") -> List[Dict[str, Any]]:
        """
        Perform search using the full 3-stage pipeline.
        
        This is the main method that demonstrates the power of the 3-stage approach.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            task_name: MTEB task name
            
        Returns:
            List of search results with scores and metadata
        """
        self.logger.info(f"Searching for: {query[:100]}...")
        
        # Check if pipeline has documents indexed
        if hasattr(self.pipeline, 'stage1') and self.pipeline.stage1:
            current_docs = getattr(self.pipeline.stage1, 'documents', [])
            if len(current_docs) == 0:
                self.logger.warning("No documents indexed in pipeline. Search may return empty results.")
        
        # Use the full pipeline for search
        try:
            pipeline_out = self.pipeline.search(query, top_k=top_k)
        except ValueError as e:
            if "No documents indexed" in str(e):
                self.logger.warning("Pipeline has no indexed documents. Returning empty results.")
                return []
            else:
                raise e
        
    # pipeline_out is a dict with key 'results'
        results_list = pipeline_out.get("results", []) if isinstance(pipeline_out, dict) else pipeline_out

        # Convert to MTEB-compatible format
        formatted_results = []
        for i, result in enumerate(results_list):
            internal_id = result.get("doc_id", i)
            # Map internal numeric id back to external corpus id if available
            external_id = self._doc_id_map.get(int(internal_id), str(internal_id)) if isinstance(internal_id, (int, np.integer)) else str(internal_id)
            formatted_results.append({
                "id": external_id,
                # Coerce score to float (avoid list/np types)
                "score": float(result.get("stage3_score", result.get("stage2_score", result.get("score", 0.0))) if not isinstance(result.get("stage3_score", result.get("stage2_score", result.get("score", 0.0))), list) else (result.get("stage3_score", result.get("stage2_score", result.get("score", [0.0])))[0] if result.get("stage3_score", result.get("stage2_score", result.get("score", [0.0]))) else 0.0)),
                "text": result.get("document", ""),
                "rank": i + 1,
                "stage1_score": result.get("stage1_score", 0.0),
                "stage2_score": result.get("stage2_score", 0.0), 
                "stage3_score": result.get("stage3_score", 0.0)
            })
        
        return formatted_results
    
    def predict(self, 
                queries: Union[List[str], List[tuple], List[List[str]]], 
                corpus: List[str] = None,
                top_k: int = 10,
                task_name: str = "", **kwargs) -> Union[List[List[Dict[str, Any]]], List[float]]:
        """
        Batch prediction for multiple queries against a corpus.
        
        Args:
            queries: List of query strings
            corpus: List of corpus documents (optional for some MTEB calls)
            top_k: Number of results per query
            task_name: MTEB task name
            
        Returns:
            List of result lists (one per query) or list of scores for pair inputs
        """
        # Handle pair inputs used by MTEB (list of (query, doc) pairs) using full tri-stage
        if queries and isinstance(queries[0], (tuple, list)) and len(queries[0]) in (2, 3) and corpus is None:
            from collections import defaultdict, OrderedDict
            pairs = queries  # type: ignore

            # Build unique corpus from pairs and index it once
            unique_docs = OrderedDict()
            for p in pairs:
                unique_docs.setdefault(str(p[1]), None)

            # Avoid re-adding the same doc set repeatedly across calls
            doc_set_key = hash(tuple(unique_docs.keys()))
            if getattr(self, "_last_pair_doc_key", None) != doc_set_key:
                start_idx = len(getattr(self.pipeline.stage1, 'documents', [])) if getattr(self.pipeline, 'stage1', None) else 0
                self.pipeline.add_documents(list(unique_docs.keys()))
                # Map document text -> internal id for this batch
                self._pair_doc_index = {}
                for offset, doc_text in enumerate(unique_docs.keys()):
                    self._pair_doc_index[doc_text] = start_idx + offset
                self._last_pair_doc_key = doc_set_key

            # Group pairs by query
            group_docs = defaultdict(list)  # query_text -> list of (idx, doc_text)
            for idx, p in enumerate(pairs):
                group_docs[str(p[0])].append((idx, str(p[1])))

            scores_out: List[Optional[float]] = [None] * len(pairs)
            for q, items in group_docs.items():
                idxs = [i for i, _ in items]
                docs_for_q = [d for _, d in items]
                # Run full pipeline search; limit top_k to the number of candidate docs for this query
                try:
                    results = self.pipeline.search(q, top_k=max(1, len(docs_for_q)))
                except Exception as e:
                    self.logger.warning(f"Pipeline search failed for a group ({e}); using zeros")
                    results = {"results": []}
                res_list = results.get("results", []) if isinstance(results, dict) else results
                # Map document text -> score from final stage
                res_map = {}
                for r in res_list:
                    doc_text = r.get("document", "")
                    score_val = r.get("stage3_score", r.get("stage2_score", r.get("score", 0.0)))
                    if isinstance(score_val, list):
                        score_val = score_val[0] if score_val else 0.0
                    try:
                        res_map[doc_text] = float(score_val)
                    except Exception:
                        res_map[doc_text] = 0.0
                # Assign scores back to requested docs
                for i, d in zip(idxs, docs_for_q):
                    scores_out[i] = res_map.get(d, 0.0)

            return [s if s is not None else 0.0 for s in scores_out]

        # Handle different calling patterns from MTEB
        if corpus is None:
            # MTEB might call predict with just queries for some tasks
            all_results = []
            for query in queries:
                results = self.search(query, top_k=top_k, task_name=task_name)
                all_results.append(results)
            return all_results
        else:
            # Standard case with corpus - ensure documents are indexed
            if corpus and len(corpus) > 0:
                self._ensure_documents_indexed(corpus)
            
            # Process each query
            all_results = []
            for query in queries:
                results = self.search(query, top_k=top_k, task_name=task_name)
                all_results.append(results)
            
            return all_results
    
    def search_cross_encoder(self, corpus, queries, top_k: int = 10, **kwargs):
        """
        Special method for MTEB cross-encoder search.
        Returns results in the format MTEB expects.
        """
        # Ensure corpus is indexed with proper id mapping
        def _extract_corpus(c):
            doc_ids: List[str] = []
            texts: List[str] = []
            # Case 1: dict-like mapping id -> {text,title}
            if isinstance(c, dict):
                for cid, cval in c.items():
                    doc_ids.append(str(cid))
                    if isinstance(cval, dict):
                        texts.append(cval.get("text", "") or "")
                    else:
                        texts.append(str(cval))
                return doc_ids, texts
            # Case 2: list/sequence of dicts with _id/text
            if hasattr(c, "__len__") and not isinstance(c, (str, bytes)):
                try:
                    sample = c[0] if len(c) > 0 else None
                except Exception:
                    sample = None
                if isinstance(sample, dict):
                    for row in c:
                        cid = str(row.get("_id", row.get("id", len(doc_ids))))
                        doc_ids.append(cid)
                        texts.append(row.get("text", "") or "")
                    return doc_ids, texts
            # Fallback: treat as iterable of texts
            try:
                for i, row in enumerate(c):
                    doc_ids.append(str(i))
                    texts.append(str(row))
            except TypeError:
                pass
            return doc_ids, texts

        if corpus is not None and len(corpus) > 0:
            # Support dict corpus mapping id-> {text,title}
            if isinstance(corpus, dict):
                doc_ids = [str(k) for k in corpus.keys()]
                corpus_texts = [v.get("text", "") if isinstance(v, dict) else str(v) for v in corpus.values()]
            else:
                doc_ids, corpus_texts = _extract_corpus(corpus)
            # Track current index length to build mapping
            start_idx = len(getattr(self.pipeline.stage1, 'documents', [])) if getattr(self.pipeline, 'stage1', None) else 0
            # Add documents to pipeline
            self.pipeline.add_documents(corpus_texts)
            # Build index->external-id map for this batch
            for offset, cid in enumerate(doc_ids):
                self._doc_id_map[start_idx + offset] = cid
        
        # Process queries
        all_results = {}
        for i, query in enumerate(queries if isinstance(queries, list) else list(queries)):
            # queries may be dict id->text or list of dicts
            if isinstance(queries, dict):
                query_id = str(query)
                query_text = str(queries[query])
            else:
                query_text = query.get("text", "") if isinstance(query, dict) else str(query)
                query_id = query.get("_id", str(i)) if isinstance(query, dict) else str(i)
            
            results = self.search(query_text, top_k=top_k)
            
            # Convert to MTEB expected format: {query_id: {doc_id: score}}
            query_result = {}
            for result in results:
                doc_id = str(result.get("id", ""))
                score = result.get("score", 0.0)
                # Ensure score is a single float value, not a list
                if isinstance(score, list):
                    score = score[0] if score else 0.0
                query_result[doc_id] = float(score)
            
            all_results[query_id] = query_result
        
        return all_results
    
    def __call__(self, *args, **kwargs):
        """Make the model callable for MTEB compatibility"""
        # Handle different calling patterns from MTEB
        if len(args) == 1 and isinstance(args[0], list):
            # Single argument - likely queries for encoding
            return self.encode(args[0], **kwargs)
        elif len(args) == 2 and isinstance(args[0], list) and isinstance(args[1], list):
            # Two arguments - likely (queries, corpus) for retrieval
            return self.predict(args[0], args[1], **kwargs)
        else:
            # Fall back to encode
            return self.encode(*args, **kwargs)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the 3-stage pipeline configuration"""
        return {
            "stage1_model": self.pipeline.config.stage1_model,
            "stage2_model": self.pipeline.config.stage2_model,
            "stage3_model": self.pipeline.config.stage3_model,
            "device": self.pipeline.config.device,
            "stage1_top_k": self.pipeline.config.stage1_top_k,
            "stage2_top_k": self.pipeline.config.stage2_top_k,
            "stage3_top_k": self.pipeline.config.stage3_top_k,
            "similarity_metric": self.similarity_metric_name,
            "max_seq_length": self.max_seq_length
        }
    
    def __repr__(self):
        return f"TriStageMTEBModel(stage1={self.pipeline.config.stage1_model}, stage2={self.pipeline.config.stage2_model}, stage3={self.pipeline.config.stage3_model})"


def create_tristage_model(model_name: str = "tristage-rag", **kwargs) -> TriStageMTEBModel:
    """
    Factory function to create TriStageMTEBModel instance.
    Compatible with MTEB's model loading pattern.
    
    Args:
        model_name: Model name (ignored, kept for compatibility)
        **kwargs: Additional arguments for model initialization
        
    Returns:
        TriStageMTEBModel instance
    """
    return TriStageMTEBModel(**kwargs)


def _register_with_mteb():
    try:
        # Lazy import to avoid static analysis errors when MTEB isn't installed
        _locals = {}
        exec('from mteb.models.model import Model\nfrom mteb.models.model_wrapper import ModelWrapper', globals(), _locals)
        Model = _locals['Model']
        ModelWrapper = _locals['ModelWrapper']

        class MTEBTriStageWrapper(ModelWrapper):
            """MTEB wrapper for TriStage-RAG pipeline"""

            def __init__(self, model: TriStageMTEBModel, **kwargs):
                super().__init__(model, **kwargs)
                self.model = model

            def encode(self, sentences, **kwargs):
                return self.model.encode(sentences, **kwargs)

            def predict(self, queries, corpus, **kwargs):
                return self.model.predict(queries, corpus, **kwargs)

        Model.register("tristage-rag", MTEBTriStageWrapper)
    except Exception:
        # MTEB not available, or registration not needed
        return

_register_with_mteb()