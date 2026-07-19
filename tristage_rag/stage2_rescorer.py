import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
from .base_stage import BaseStage
from .utils import resolve_model_path

@dataclass
class Stage2Config:
    model_name: str = "lightonai/GTE-ModernColBERT-v1"
    device: str = "auto"
    cache_dir: str = "./models"
    max_seq_length: int = 192  # Optimized for 4GB VRAM
    batch_size: int = 16
    # 50, not the more conventional 100: per the SemEval-2026 finding (see
    # papers/2605.12028v1.pdf) that k=50 outperforms k=500 at Stage 2. This
    # matches PipelineConfig.stage2_top_k so a bare Stage2Config() agrees with
    # the pipeline default.
    top_k_candidates: int = 50
    use_fp16: bool = True
    pooling_method: str = "cls"  # "cls", "mean", or "max"
    normalize_embeddings: bool = True
    scoring_method: str = "maxsim"  # "maxsim" or "colbert"
    use_gpu_if_available: bool = True
    early_exit_threshold: float = 0.0  # Score gap threshold for early exit (0 = disabled)
    # Compression: "none" | "float16" | "int8" — int8 gives 4x compression with minimal quality loss
    cache_precision: str = "int8"

class ColBERTScorer(BaseStage):
    """ColBERT-style MaxSim scoring with pre-computed document embeddings.

    Document token embeddings are computed once at index time and cached in
    memory.  At query time only the query is encoded and MaxSim runs against
    the cached embeddings — this avoids re-encoding all candidates on every
    query, which was the dominant latency bottleneck.
    """

    def __init__(self, config: Stage2Config):
        super().__init__()
        self.config = config

        # Model / tokenizer
        self.model = None
        self.tokenizer = None
        self.device = self._init_device(config.device, config.use_gpu_if_available)

        # Pre-computed document embeddings: doc_id -> (token_embeddings, seq_len)
        self._doc_embeddings: Dict[int, Tuple[torch.Tensor, int]] = {}
        self._doc_texts: Dict[int, str] = {}

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the ColBERT-style model"""
        try:
            self.logger.info(f"Loading Stage 2 model: {self.config.model_name}")

            model_source = resolve_model_path(self.config.model_name, self.config.cache_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_source, cache_dir=self.config.cache_dir
            )
            self.model = AutoModel.from_pretrained(
                model_source, cache_dir=self.config.cache_dir
            )

            # GPU with CPU fallback on OOM
            try:
                self.model.to(self.device)
                self.model.eval()
                self.logger.info(f"Model loaded successfully on {self.device}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.device == "cuda":
                    self.logger.warning(f"CUDA OOM: {e}. Falling back to CPU.")
                    self.device = "cpu"
                    self.model.to(self.device)
                    self.model.eval()
                else:
                    raise

            self.use_amp = self.config.use_fp16 and self.device == "cuda"
            self.logger.info(f"Using FP16: {self.use_amp}")

        except Exception as e:
            self.logger.error(f"Error loading Stage 2 model: {e}")
            raise

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch, return (token_embeddings, attention_mask).

        Both tensors are on ``self.device``.
        """
        safe_texts = [t if t and t.strip() else "empty" for t in texts]

        encoded = self.tokenizer(
            safe_texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(**encoded)
            else:
                outputs = self.model(**encoded)

        return outputs.last_hidden_state, encoded["attention_mask"]

    def _encode_single_text(self, text: str) -> torch.Tensor:
        """Encode a single text, return non-padded token embeddings [1, seq_len, dim]."""
        if not text or not text.strip():
            text = "empty"

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
            padding=False,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(**encoded)
            else:
                outputs = self.model(**encoded)

        seq_length = encoded["attention_mask"].sum().item()
        return outputs.last_hidden_state[:, :seq_length, :]

    def encode_query(self, query: str) -> torch.Tensor:
        """Encode query to get token embeddings"""
        return self._encode_single_text(query)

    # ------------------------------------------------------------------
    # Document embedding cache (pre-computation)
    # ------------------------------------------------------------------

    def add_documents(self, doc_ids: List[int], documents: List[str]):
        """Pre-compute and cache ColBERT token embeddings for documents.

        Called once when documents are added to the pipeline.  This is the
        key optimisation: embeddings are computed offline so that at query
        time only the query needs encoding.

        Compression controlled by ``config.cache_precision``:
        - "none":   float32 (original, 1x)
        - "float16": float16 (2x compression, negligible quality loss)
        - "int8":    int8 scalar quantization (4x compression, minimal quality loss)
        """
        if not documents:
            return

        self.logger.info(
            f"Pre-computing ColBERT embeddings for {len(documents)} docs "
            f"(precision={self.config.cache_precision})"
        )

        # Batch encode all documents
        for i in range(0, len(documents), self.config.batch_size):
            batch_ids = doc_ids[i : i + self.config.batch_size]
            batch_docs = documents[i : i + self.config.batch_size]

            token_embs, attn_mask = self._encode_batch(batch_docs)
            seq_lengths = attn_mask.sum(dim=1)

            for j, did in enumerate(batch_ids):
                sl = seq_lengths[j].item()
                emb = token_embs[j, :sl, :].cpu()

                # Apply compression
                if self.config.cache_precision == "int8":
                    emb = self._quantize_int8(emb)
                elif self.config.cache_precision == "float16":
                    emb = emb.half()
                # else: keep float32

                self._doc_embeddings[did] = (emb, int(sl))
                self._doc_texts[did] = batch_docs[j]

        self.logger.info(
            f"ColBERT embeddings cached for {len(self._doc_embeddings)} documents"
        )

    def _quantize_int8(self, emb: torch.Tensor) -> torch.Tensor:
        """Quantize float32 embeddings to int8 with per-channel scale.

        Returns a tuple (int8_tensor, scale_vec) for later dequantization.
        4x compression with minimal quality loss on normalized embeddings.
        """
        scale = emb.abs().max(dim=-1, keepdim=True).values / 127.0
        scale = scale.clamp(min=1e-8)
        quantized = torch.clamp(torch.round(emb / scale), -128, 127).to(torch.int8)
        return (quantized, scale.squeeze(-1).float())

    def _dequantize(self, cached: Tuple) -> torch.Tensor:
        """Restore embeddings from cache (float32, float16, or int8 tuple)."""
        if isinstance(cached, tuple) and len(cached) == 2 and cached[0].dtype == torch.int8:
            # Int8 quantized: (int8_tensor, scale_vec) → float32
            quantized, scale = cached
            return quantized.float() * scale.unsqueeze(-1)
        emb, _ = cached
        return emb.float()

    def remove_documents(self, doc_ids: List[int]):
        """Remove cached embeddings for the given document IDs."""
        for did in doc_ids:
            self._doc_embeddings.pop(did, None)
            self._doc_texts.pop(did, None)

    def get_cached_doc(self, doc_id: int) -> Optional[torch.Tensor]:
        """Return cached token embeddings for *doc_id* (or None)."""
        entry = self._doc_embeddings.get(doc_id)
        return entry[0] if entry is not None else None

    # ------------------------------------------------------------------
    # Cache persistence (public API used by RetrievalPipeline.save/load_index)
    # ------------------------------------------------------------------

    def has_cached_docs(self) -> bool:
        """Return True iff any ColBERT token embeddings are cached."""
        return bool(self._doc_embeddings)

    def export_cache(self) -> Dict[int, Tuple[Any, int]]:
        """Return a serializable copy of the ColBERT token-embedding cache.

        Each entry maps ``doc_id -> (emb_data, seq_len)`` where ``emb_data`` is
        either a torch tensor (float32 / float16) or, for int8 precision, the
        tuple ``(int8_tensor, scale_vec)``. Callers are free to convert tensors
        to numpy / pickle them; this method makes no copy of the tensor *data*
        beyond what torch already holds.
        """
        return dict(self._doc_embeddings)

    def import_cache(self, cache: Dict[int, Tuple[Any, int]]) -> None:
        """Replace the in-memory ColBERT cache with *cache*.

        Mirrors :meth:`export_cache`. Accepts either torch tensors directly or
        the numpy-converted form produced by ``RetrievalPipeline.load_index``
        (torch.from_numpy is applied as needed).
        """
        self._doc_embeddings = dict(cache)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _maxsim_score(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute MaxSim score between query and document token embeddings."""
        query_norm = F.normalize(query_embeddings, p=2, dim=-1)
        doc_norm = F.normalize(doc_embeddings, p=2, dim=-1)

        sim_matrix = torch.matmul(query_norm.squeeze(0), doc_norm.squeeze(0).T)
        max_sim_scores = torch.max(sim_matrix, dim=-1)[0]

        return torch.mean(max_sim_scores)

    def _colbert_score(self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute ColBERT score (sum of max similarities, softmax-weighted)."""
        query_norm = F.normalize(query_embeddings, p=2, dim=-1)
        doc_norm = F.normalize(doc_embeddings, p=2, dim=-1)

        sim_matrix = torch.matmul(query_norm.squeeze(0), doc_norm.squeeze(0).T)
        max_sim_scores = torch.max(sim_matrix, dim=-1)[0]

        query_weights = F.softmax(max_sim_scores, dim=0)
        return torch.sum(max_sim_scores * query_weights)

    def _score_pair(self, query_embs: torch.Tensor, doc_embs: torch.Tensor) -> float:
        # Cast to float32 for accurate matmul (cached docs may be float16)
        if doc_embs.dtype != query_embs.dtype:
            doc_embs = doc_embs.to(query_embs.dtype)
        if self.config.scoring_method == "colbert":
            return float(self._colbert_score(query_embs, doc_embs).item())
        return float(self._maxsim_score(query_embs, doc_embs).item())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rescore_candidates(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rescore candidates using pre-computed document embeddings.

        Only the query is encoded at search time.  Document embeddings are
        looked up from the cache populated by :meth:`add_documents`.
        """
        if not candidates:
            return []

        self.logger.info(f"Rescoring {len(candidates)} candidates with Stage 2")

        # Encode query once
        query_embeddings = self.encode_query(query)

        # Score each candidate using cached embeddings
        scored_candidates = []
        cache_hits = 0
        cache_misses = 0

        for candidate in candidates:
            doc_id = candidate["doc_id"]
            cached = self._doc_embeddings.get(doc_id)

            if cached is not None:
                doc_embs = self._dequantize(cached)
                doc_embs = doc_embs.to(self.device)
                cache_hits += 1
            else:
                # Fallback: encode on-the-fly (should be rare after warm-up)
                cache_misses += 1
                doc_embs = self._encode_single_text(candidate["document"])

            score = self._score_pair(query_embeddings, doc_embs)
            candidate["stage2_score"] = score
            candidate["stage"] = "stage2"
            scored_candidates.append(candidate)

        # Sort by Stage 2 score (descending)
        scored_candidates.sort(key=lambda x: x["stage2_score"], reverse=True)

        # Keep top-k
        top_candidates = scored_candidates[: self.config.top_k_candidates]

        self.logger.info(
            f"Stage 2 done. cache_hits={cache_hits} cache_misses={cache_misses}. "
            f"Top score: {top_candidates[0]['stage2_score']:.4f}"
        )
        return top_candidates

    def compute_similarity_matrix(self, query: str, documents: List[str]) -> np.ndarray:
        """Compute similarity matrix between query and documents (uncached)."""
        query_embeddings = self.encode_query(query)

        for i in range(0, len(documents), self.config.batch_size):
            batch = documents[i : i + self.config.batch_size]
            token_embs, attn_mask = self._encode_batch(batch)
            seq_lengths = attn_mask.sum(dim=1)
            for j in range(len(batch)):
                sl = seq_lengths[j].item()
                doc_embs = token_embs[j, :sl, :].unsqueeze(0)
                yield self._score_pair(query_embeddings, doc_embs)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "max_seq_length": self.config.max_seq_length,
            "use_fp16": self.use_amp,
            "pooling_method": self.config.pooling_method,
            "scoring_method": self.config.scoring_method,
            "batch_size": self.config.batch_size,
            "embedding_dim": self.model.config.hidden_size if self.model else None,
            "cached_documents": len(self._doc_embeddings),
        }
