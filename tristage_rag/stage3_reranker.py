import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .base_stage import BaseStage
from .utils import resolve_model_path

@dataclass
class Stage3Config:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    device: str = "auto"
    cache_dir: str = "./models"
    max_length: int = 256  # Optimized for 4GB VRAM
    batch_size: int = 32  # Safe batch size for 4GB VRAM
    top_k_final: int = 20
    use_fp16: bool = True
    use_gpu_if_available: bool = True
    activation_fxn: str = "sigmoid"  # "sigmoid" or "softmax"
    normalize_scores: bool = True
    # Early-exit: skip cross-encoder when Stage 2 scores already separate results
    early_exit_enabled: bool = True
    early_exit_threshold: float = 0.15  # min gap between top-1 and median score
    early_exit_min_candidates: int = 10  # always rerank at least this many

class CrossEncoderReranker(BaseStage):
    """Stage 3: Cross-Encoder Reranker with optional early exit.

    When ``early_exit_enabled`` is True the reranker checks whether Stage 2
    scores already produce a clear separation.  If the gap between the top
    score and the median of the top-K candidates exceeds
    ``early_exit_threshold``, the cross-encoder is skipped entirely and the
    Stage 2 ranking is returned as-is.
    """

    def __init__(self, config: Stage3Config):
        super().__init__()
        self.config = config

        # Initialize model
        self.model = None
        self.tokenizer = None
        self.device = self._init_device(config.device, getattr(config, 'use_gpu_if_available', True))

        # Last-call introspection: True iff the most recent rerank() call took
        # the early-exit branch. Read by RetrievalPipeline to avoid inferring
        # early-exit from result contents (which is fragile when Stage 3 returns
        # [] for unrelated reasons).
        self.last_early_exit = False

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the cross-encoder model"""
        try:
            self.logger.info(f"Loading Stage 3 model: {self.config.model_name}")

            model_source = resolve_model_path(self.config.model_name, self.config.cache_dir)

            # Try to load as CrossEncoder first (preferred)
            try:
                self.model = CrossEncoder(
                    model_source,
                    device=self.device,
                    max_length=self.config.max_length,
                    cache_folder=self.config.cache_dir
                )
                self.logger.info("Loaded as SentenceTransformers CrossEncoder")
                self.use_sentence_transformers = True
            except Exception:
                self.logger.info("Falling back to HuggingFace AutoModel")
                self.use_sentence_transformers = False
                
                # Load tokenizer and model separately
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    cache_dir=self.config.cache_dir
                )
                
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_source,
                    cache_dir=self.config.cache_dir
                )
                
                # Try to load on GPU, fallback to CPU on OOM
                try:
                    self.model.to(self.device)
                    self.model.eval()
                    self.logger.info(f"HuggingFace model loaded successfully on {self.device}")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and self.device == "cuda":
                        self.logger.warning(f"CUDA OOM: {e}. Falling back to CPU.")
                        self.device = "cpu"
                        self.model.to(self.device)
                        self.model.eval()
                        self.logger.info(f"HuggingFace model loaded successfully on {self.device} (fallback)")
                    else:
                        raise
            
            # Set mixed precision if enabled
            self.use_amp = self.config.use_fp16 and self.device == "cuda"
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            self.logger.info(f"Using FP16: {self.use_amp}")
            
        except Exception as e:
            self.logger.error(f"Error loading Stage 3 model: {e}")
            raise
    
    def _predict_with_huggingface(self, queries: List[str], documents: List[str],
                                  batch_size: Optional[int] = None) -> List[float]:
        """Predict using HuggingFace AutoModel.

        ``batch_size`` overrides ``self.config.batch_size`` for this call only
        (used by the adaptive reranker) without mutating shared config state.
        """
        bs = batch_size if batch_size is not None else self.config.batch_size
        all_score_tensors = []

        for i in range(0, len(queries), bs):
            batch_queries = queries[i:i + bs]
            batch_docs = documents[i:i + bs]

            encoded = self.tokenizer(
                batch_queries, batch_docs,
                truncation=True, padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(**encoded)
                else:
                    outputs = self.model(**encoded)

                logits = outputs.logits
                if self.config.activation_fxn == "sigmoid":
                    scores = torch.sigmoid(logits).squeeze(-1)
                else:
                    scores = F.softmax(logits, dim=-1)[:, 1]

                all_score_tensors.append(scores.cpu())

        return torch.cat(all_score_tensors).tolist() if all_score_tensors else []
    
    def predict(self, query: str, documents: List[str],
                batch_size: Optional[int] = None) -> List[float]:
        """Predict relevance scores for query-document pairs.

        ``batch_size`` overrides ``self.config.batch_size`` for this call only
        (used by the adaptive reranker) without mutating shared config state.
        """
        if not documents:
            return []

        bs = batch_size if batch_size is not None else self.config.batch_size
        queries = [query] * len(documents)

        if self.use_sentence_transformers:
            sentence_pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(
                sentence_pairs,
                batch_size=bs,
                show_progress_bar=False
            ).tolist()
        else:
            scores = self._predict_with_huggingface(queries, documents, batch_size=bs)

        if self.config.normalize_scores:
            scores = self._normalize_scores(scores)

        return scores
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        if not scores:
            return scores
        
        scores_array = np.array(scores)
        
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score > min_score:
            normalized = (scores_array - min_score) / (max_score - min_score)
        else:
            normalized = np.zeros_like(scores_array)
        
        return normalized.tolist()

    # ------------------------------------------------------------------
    # Early exit logic
    # ------------------------------------------------------------------

    def _should_early_exit(self, candidates: List[Dict[str, Any]]) -> bool:
        """Return True if Stage 2 scores clearly separate the top candidates.

        Heuristic: if ``top1_score - median_score > threshold`` the cross-encoder
        is unlikely to change the ranking meaningfully.
        """
        if not self.config.early_exit_enabled:
            return False

        if len(candidates) < self.config.early_exit_min_candidates:
            return False

        scores = [c.get("stage2_score", 0.0) for c in candidates]
        if not scores:
            return False

        top1 = max(scores)
        median = float(np.median(scores))
        gap = top1 - median

        should_exit = gap > self.config.early_exit_threshold
        if should_exit:
            self.logger.info(
                f"Early exit triggered: top1={top1:.4f} median={median:.4f} gap={gap:.4f} "
                f"> threshold={self.config.early_exit_threshold}"
            )
        return should_exit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(self, query: str, candidates: List[Dict[str, Any]],
               batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder.

        If early exit is enabled and Stage 2 scores already separate the
        candidates clearly, the cross-encoder is skipped and the Stage 2
        ranking is returned unchanged.

        ``batch_size`` overrides ``self.config.batch_size`` for this call only
        (used by the adaptive reranker) without mutating shared config state.
        """
        if not candidates:
            self.last_early_exit = False
            return []

        self.logger.info(f"Reranking {len(candidates)} candidates with Stage 3")

        # --- Early exit check ---
        if self._should_early_exit(candidates):
            candidates.sort(key=lambda x: x.get("stage2_score", 0.0), reverse=True)
            top_k = candidates[: self.config.top_k_final]
            self.logger.info(
                f"Stage 3 early exit: returning top {len(top_k)} from Stage 2 ranking"
            )
            self.last_early_exit = True
            return top_k

        # --- Full cross-encoder reranking ---
        self.last_early_exit = False
        documents = [candidate["document"] for candidate in candidates]

        try:
            scores = self.predict(query, documents, batch_size=batch_size)
        except Exception as e:
            self.logger.error(f"Error reranking: {e}")
            return candidates

        # Update candidates in-place
        for candidate, score in zip(candidates, scores):
            candidate["stage3_score"] = score
            candidate["stage"] = "stage3"

        # Sort and take top-k
        candidates.sort(key=lambda x: x["stage3_score"], reverse=True)
        final_results = candidates[:self.config.top_k_final]

        self.logger.info(
            f"Stage 3 reranking completed. Top score: "
            f"{final_results[0]['stage3_score'] if final_results else 0:.4f}"
        )
        return final_results
    
    def batch_rerank(self, queries: List[str], candidates_list: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Rerank multiple query-candidate pairs in batch"""
        if not queries or not candidates_list:
            return []
        
        if len(queries) != len(candidates_list):
            raise ValueError("Number of queries must match number of candidate lists")
        
        results = []
        for query, candidates in zip(queries, candidates_list):
            reranked = self.rerank(query, candidates)
            results.append(reranked)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        info = {
            "model_name": self.config.model_name,
            "device": self.device,
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "use_fp16": self.use_amp,
            "activation_function": self.config.activation_fxn,
            "normalize_scores": self.config.normalize_scores,
            "top_k_final": self.config.top_k_final,
            "early_exit_enabled": self.config.early_exit_enabled,
            "early_exit_threshold": self.config.early_exit_threshold,
        }
        
        if self.use_sentence_transformers:
            info["model_type"] = "SentenceTransformers CrossEncoder"
        else:
            info["model_type"] = "HuggingFace AutoModel"
            info["num_labels"] = self.model.num_labels if self.model else None
        
        return info


class AdaptiveCrossEncoderReranker(CrossEncoderReranker):
    """Adaptive reranker that adjusts batch size based on input length"""
    
    def __init__(self, config: Stage3Config):
        super().__init__(config)
        self.max_text_length = config.max_length // 2
    
    def _adaptive_batch_size(self, texts: List[str]) -> int:
        """Determine optimal batch size based on text lengths"""
        if not texts:
            return self.config.batch_size
        
        avg_length = sum(len(text.split()) for text in texts) / len(texts)
        
        if avg_length > 200:
            return max(4, self.config.batch_size // 4)
        elif avg_length > 100:
            return max(8, self.config.batch_size // 2)
        elif avg_length > 50:
            return max(16, self.config.batch_size)
        else:
            return self.config.batch_size
    
    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank with adaptive batch sizing.

        Picks a batch size based on average document length and passes it down
        to ``predict`` as a per-call override — the shared ``self.config`` is
        never mutated, so this is safe under concurrent access.
        """
        if not candidates:
            return []

        documents = [candidate["document"] for candidate in candidates]
        adaptive_batch_size = self._adaptive_batch_size([query] + documents)

        return super().rerank(query, candidates, batch_size=adaptive_batch_size)
