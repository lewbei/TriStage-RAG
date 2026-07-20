import os
import json
import pickle
import logging
import math
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import torch
from collections import defaultdict
from .utils import resolve_model_path, get_device, BM25_TOKENIZER
from .base_stage import BaseStage

@dataclass
class Stage1Config:
    model_name: str = "google/embeddinggemma-300m"
    device: str = "auto"
    cache_dir: str = "./models"
    index_dir: str = "./faiss_index"
    top_k_candidates: int = 500
    batch_size: int = 32
    enable_bm25: bool = True
    bm25_top_k: int = 300
    fusion_method: str = "rrf"  # "rrf" (Reciprocal Rank Fusion) or "weighted"
    rrf_k: int = 60
    dense_weight: float = 0.7
    bm25_weight: float = 0.3
    use_fp16: bool = True
    # FAISS index settings — HNSW for 10k+, Flat for smaller
    hnsw_m: int = 32           # HNSW connections per vector
    hnsw_ef_search: int = 128  # HNSW search-time precision
    hnsw_ef_construction: int = 200  # HNSW build-time precision
    hnsw_threshold: int = 10_000  # Switch from Flat to HNSW at this size

class BM25Index:
    """BM25 index with inverted index for fast query-time retrieval."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = []
        self.idf = {}
        self.doc_lens = []
        self.avg_doc_len = 0.0
        self.corpus_size = 0
        self.documents = []
        # Inverted index: token -> list of (doc_idx, term_freq)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = BM25_TOKENIZER.sub(' ', text)
        return text.split()

    def fit(self, documents: List[str]):
        self.documents = documents
        self.corpus_size = len(documents)
        self.doc_freqs.clear()
        self.doc_lens.clear()
        self.inverted_index.clear()
        idf_counts: Dict[str, int] = defaultdict(int)

        for doc_idx, doc in enumerate(documents):
            tokens = self.tokenize(doc)
            self.doc_lens.append(len(tokens))

            term_freq: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            self.doc_freqs.append(term_freq)

            # Build inverted index and count document frequency in one pass
            seen_tokens = set()
            for token, tf in term_freq.items():
                self.inverted_index[token].append((doc_idx, tf))
                if token not in seen_tokens:
                    idf_counts[token] += 1
                    seen_tokens.add(token)

        self.avg_doc_len = sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 0.0

        # Compute IDF from document frequency counts
        for token, df in idf_counts.items():
            self.idf[token] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def _score_doc(self, query_tokens: List[str], doc_idx: int) -> float:
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        score = 0.0
        for token in query_tokens:
            if token in doc_freq and token in self.idf:
                tf = doc_freq[token]
                idf = self.idf[token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += idf * (numerator / denominator)
        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        import heapq
        query_tokens = self.tokenize(query)

        # Collect only documents that contain at least one query token
        candidate_indices: set = set()
        for token in query_tokens:
            if token in self.inverted_index:
                for doc_idx, _ in self.inverted_index[token]:
                    candidate_indices.add(doc_idx)

        if not candidate_indices:
            return []

        # Score candidates only
        scored = []
        for doc_idx in candidate_indices:
            s = self._score_doc(query_tokens, doc_idx)
            if s > 0:
                scored.append((s, doc_idx))

        # Use heapq for O(N log k) top-k extraction
        return [(doc_idx, s) for s, doc_idx in heapq.nlargest(top_k, scored)]

class Stage1Retriever(BaseStage):
    """Stage 1: Fast Candidate Generation with Dense Embeddings + FAISS + Optional BM25"""

    def __init__(self, config: Stage1Config):
        super().__init__()
        self.config = config

        # Initialize model
        self.model = None
        self.embedding_dim = None

        # Initialize indexes
        self.faiss_index = None
        self.bm25_index = None
        self.documents = []
        self.doc_metadata = []

        # Ensure directories exist
        os.makedirs(self.config.cache_dir, exist_ok=True)
        os.makedirs(self.config.index_dir, exist_ok=True)

        self._load_model()

    def _load_model(self):
        """Load the embedding model"""
        try:
            self.logger.info(f"Loading Stage 1 model: {self.config.model_name}")

            device = get_device(self.config.device)

            def _load(name, dev):
                src = resolve_model_path(name, self.config.cache_dir)
                return SentenceTransformer(src, device=dev, cache_folder=self.config.cache_dir)

            try:
                self.model = _load(self.config.model_name, device)
            except Exception as e:
                err_msg = str(e).lower()
                low_mem_signatures = [
                    "paging file is too small", "out of memory",
                    "cuda out of memory", "os error 1455"
                ]
                # Also fall back when the model is gated / unreachable without
                # auth (e.g. google/embeddinggemma-300m). Failing the whole
                # pipeline here would make the default RetrievalPipeline()
                # unusable for anyone without an HF token — silently fall back
                # to the public all-MiniLM-L6-v2 instead.
                auth_signatures = [
                    "gatedrepoerror", "gated repo", "unauthorized",
                    "401 client error", "403 client error", "access is restricted",
                ]
                if any(sig in err_msg for sig in low_mem_signatures):
                    if device == "cuda":
                        self.logger.warning(f"CUDA OOM loading '{self.config.model_name}': {e}. Falling back to CPU.")
                        device = "cpu"
                        try:
                            self.model = _load(self.config.model_name, device)
                        except Exception:
                            self.logger.warning(f"CPU also failed. Falling back to all-MiniLM-L6-v2.")
                            self.config.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                            self.model = _load(self.config.model_name, device)
                    else:
                        fallback = "sentence-transformers/all-MiniLM-L6-v2"
                        self.logger.warning(f"Low memory loading '{self.config.model_name}': {e}. Falling back to '{fallback}'.")
                        self.config.model_name = fallback
                        self.model = _load(self.config.model_name, device)
                elif any(sig in err_msg for sig in auth_signatures):
                    fallback = "sentence-transformers/all-MiniLM-L6-v2"
                    self.logger.warning(
                        f"Cannot access gated/restricted model '{self.config.model_name}' "
                        f"(set HUGGING_FACE_HUB_TOKEN to use it): {e}. "
                        f"Falling back to '{fallback}'."
                    )
                    self.config.model_name = fallback
                    self.model = _load(self.config.model_name, device)
                else:
                    raise
            
            # Get embedding dimension
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            else:
                # Fallback: encode a sample to get dimension
                sample_embedding = self.model.encode("sample text", convert_to_numpy=True)
                self.embedding_dim = sample_embedding.shape[0]
            
            self.logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            self.logger.error(f"Error loading Stage 1 model: {e}")
            raise
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts"""
        try:
            # Use FP16 if enabled and supported
            if self.config.use_fp16 and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    embeddings = self.model.encode(
                        texts,
                        batch_size=self.config.batch_size,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
            else:
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error encoding batch: {e}")
            raise
    
    def _create_faiss_index(self, embeddings: np.ndarray):
        """Create FAISS index for fast similarity search.

        Uses HNSW for corpora >= hnsw_threshold (no training, fast, 98%+ recall).
        Uses flat brute-force for smaller corpora.
        """
        try:
            d = embeddings.shape[1]
            n = len(embeddings)

            if n >= self.config.hnsw_threshold:
                # HNSW: no training needed, best speed/accuracy for 10k-100k+ docs
                self.faiss_index = faiss.IndexHNSWFlat(d, self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
                self.faiss_index.hnsw.efSearch = self.config.hnsw_ef_search
                self.faiss_index.hnsw.efConstruction = self.config.hnsw_ef_construction
                self.faiss_index.add(embeddings)
            else:
                # Flat brute-force for small corpora (instant build)
                self.faiss_index = faiss.IndexFlatIP(d)
                self.faiss_index.add(embeddings)

            self.logger.info(f"FAISS index created with {n} vectors (type: {type(self.faiss_index).__name__})")
        except Exception as e:
            self.logger.error(f"Error creating FAISS index: {e}")
            raise
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Add documents to the index"""
        if not documents:
            return
        
        self.logger.info(f"Adding {len(documents)} documents to Stage 1 index")
        
        # Store documents and metadata
        start_idx = len(self.documents)
        self.documents.extend(documents)
        
        if metadata is None:
            metadata = [{} for _ in range(len(documents))]
        self.doc_metadata.extend(metadata)
        
        # Encode documents
        embeddings = self._encode_batch(documents)
        embeddings = self._normalize_embeddings(embeddings)
        
        # Create or update FAISS index
        if self.faiss_index is None:
            self._create_faiss_index(embeddings)
        else:
            # HNSW and Flat both support sequential adds — no retrain needed
            self.faiss_index.add(embeddings)
        
        # Create or update BM25 index if enabled
        if self.config.enable_bm25:
            if self.bm25_index is None:
                self.bm25_index = BM25Index()
                self.bm25_index.fit(self.documents)
            else:
                # For simplicity, recreate BM25 index
                self.bm25_index.fit(self.documents)
        
        self.logger.info(f"Documents added successfully. Total documents: {len(self.documents)}")
    
    def _reciprocal_rank_fusion(self, dense_results: List[Tuple[int, float]], 
                               bm25_results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Combine results using Reciprocal Rank Fusion"""
        scores = defaultdict(float)
        
        # Add dense scores
        for rank, (doc_idx, score) in enumerate(dense_results):
            scores[doc_idx] += 1.0 / (self.config.rrf_k + rank + 1)
        
        # Add BM25 scores
        for rank, (doc_idx, score) in enumerate(bm25_results):
            scores[doc_idx] += 1.0 / (self.config.rrf_k + rank + 1)
        
        # Sort by combined score
        fused_results = [(doc_idx, score) for doc_idx, score in scores.items()]
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        return fused_results
    
    def _weighted_fusion(self, dense_results: List[Tuple[int, float]], 
                        bm25_results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Combine results using weighted scores"""
        scores = defaultdict(float)
        
        # Normalize and add dense scores
        if dense_results:
            max_dense_score = max(score for _, score in dense_results)
            for doc_idx, score in dense_results:
                scores[doc_idx] += self.config.dense_weight * (score / max_dense_score)
        
        # Normalize and add BM25 scores
        if bm25_results:
            max_bm25_score = max(score for _, score in bm25_results)
            for doc_idx, score in bm25_results:
                scores[doc_idx] += self.config.bm25_weight * (score / max_bm25_score)
        
        # Sort by combined score
        fused_results = [(doc_idx, score) for doc_idx, score in scores.items()]
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        return fused_results
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for documents using Stage 1 retrieval.

        Dense (FAISS) and BM25 run **in parallel** when both are enabled,
        since they are independent operations.
        """
        if self.faiss_index is None:
            raise ValueError("No documents indexed. Call add_documents() first.")
        
        top_k = top_k or self.config.top_k_candidates
        
        # Encode query (shared by both retrievers)
        query_embedding = self._encode_batch([query])
        query_embedding = self._normalize_embeddings(query_embedding)

        # Run dense + BM25 in parallel
        dense_results: List[Tuple[int, float]] = []
        bm25_results: List[Tuple[int, float]] = []

        from concurrent.futures import ThreadPoolExecutor, as_completed

        futures = {}
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="stage1") as pool:
            # Dense search
            def _dense_search():
                scores, indices = self.faiss_index.search(query_embedding, top_k)
                return [(int(idx), float(s))
                        for idx, s in zip(indices[0], scores[0]) if idx >= 0]
            futures[pool.submit(_dense_search)] = "dense"

            # BM25 search (if enabled)
            if self.config.enable_bm25 and self.bm25_index is not None:
                def _bm25_search():
                    return self.bm25_index.search(query, self.config.bm25_top_k)
                futures[pool.submit(_bm25_search)] = "bm25"

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if name == "dense":
                        dense_results = result
                    else:
                        bm25_results = result
                except Exception as e:
                    self.logger.error(f"Stage 1 {name} search failed: {e}")
        
        # Combine results
        if bm25_results:
            if self.config.fusion_method == "rrf":
                fused_results = self._reciprocal_rank_fusion(dense_results, bm25_results)
            else:  # weighted
                fused_results = self._weighted_fusion(dense_results, bm25_results)
            final_results = fused_results[:top_k]
        else:
            final_results = dense_results[:top_k]
        
        # Format results
        results = []
        for doc_idx, score in final_results:
            if doc_idx < len(self.documents):
                result = {
                    "doc_id": doc_idx,
                    "document": self.documents[doc_idx],
                    "score": score,
                    "stage1_score": score,
                    "metadata": self.doc_metadata[doc_idx],
                    "stage": "stage1"
                }
                results.append(result)
        
        self.logger.info(f"Stage 1 search completed. Found {len(results)} candidates")
        return results
    
    def save_index(self, index_path: Optional[str] = None):
        """Save the index to disk"""
        if index_path is None:
            index_path = os.path.join(self.config.index_dir, "stage1_index.pkl")
        
        index_data = {
            "documents": self.documents,
            "doc_metadata": self.doc_metadata,
            "config": self.config.__dict__,
            "bm25_index": self.bm25_index
        }
        
        # Save FAISS index separately
        faiss_path = os.path.join(self.config.index_dir, "stage1_faiss.index")
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, faiss_path)
        
        # Save other data
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        self.logger.info(f"Stage 1 index saved to {index_path}")
    
    def load_index(self, index_path: Optional[str] = None):
        """Load the index from disk"""
        if index_path is None:
            index_path = os.path.join(self.config.index_dir, "stage1_index.pkl")
        
        if not os.path.exists(index_path):
            self.logger.warning(f"Index file not found: {index_path}")
            return
        
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.documents = index_data["documents"]
        self.doc_metadata = index_data["doc_metadata"]
        self.bm25_index = index_data.get("bm25_index")
        
        # Load FAISS index
        faiss_path = os.path.join(self.config.index_dir, "stage1_faiss.index")
        if os.path.exists(faiss_path):
            self.faiss_index = faiss.read_index(faiss_path)
        
        self.logger.info(f"Stage 1 index loaded from {index_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index"""
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embedding_dim,
            "faiss_index_type": type(self.faiss_index).__name__ if self.faiss_index else None,
            "bm25_enabled": self.config.enable_bm25,
            "bm25_vocabulary_size": len(self.bm25_index.inverted_index) if self.bm25_index else 0,
            "config": self.config.__dict__
        }
