# ------------------------------------------------------------------------------------------
# EmbeddingModel.py:
# Custom adapter for ONNX BGE-M3 embeddings (quantized INT8) to work with LangChain + FAISS.
# ------------------------------------------------------------------------------------------
from __future__ import annotations
from typing import Optional, List
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from langchain_core.embeddings import Embeddings
import logging
import os

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class EmbeddingModel(Embeddings):
    """
    Minimal embeddings adapter for the BGE-M3 model (ONNX, INT8).
    
    Responsibilities:
      - Tokenize text using HuggingFace AutoTokenizer.
      - Run ONNX inference via Optimum ORT.
      - Perform mean pooling over token embeddings with attention mask.
      - Apply L2 normalization (important for cosine similarity in FAISS).
    
    Implements LangChain's abstract Embeddings API:
      • embed_documents(texts) -> List[List[float]]
      • embed_query(text) -> List[float]
    
    Also provides convenience aliases:
      • embed_multiple_documents(texts)
      • embed_single_document(text)
    """

    def __init__(
        self,
        model_id: str,
        onnx_file_name: Optional[str] = "model_quantized.onnx",
        max_seq_len: int = 8192,
        normalize: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        logger.info("Initializing EmbeddingModel...")

        # Load tokenizer (always uses base BAAI/bge-m3)
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", cache_dir=cache_dir)
        logger.debug("Tokenizer loaded successfully from 'BAAI/bge-m3'.")

        # Try to locate and load the ONNX model file
        candidates = [
            onnx_file_name,
            "model_quantized.onnx",
            "model.onnx",
            "onnx/model_quantized.onnx",
            "onnx/model.onnx",
        ]
        last_err = None
        self.model = None
        logger.info("Attempting to load ONNX model from provided candidates...")
        for name in [n for n in candidates if n]:
            try:
                self.model = ORTModelForFeatureExtraction.from_pretrained(
                    model_id, file_name=name, cache_dir=cache_dir
                )
                logger.info(f"ONNX model loaded successfully from file: {name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load model from {name}: {e}")
                last_err = e

        # Fail if no model could be loaded        
        if self.model is None:
            logger.error(f"Could not load ONNX model from {model_id}: {last_err}")
            raise RuntimeError(f"Could not load ONNX model from {model_id}: {last_err}")

        self.max_seq_len = max_seq_len
        self.normalize = normalize

        # Ensure normalization is always on because FAISS IndexFlatIP expects unit vectors
        if not self.normalize:
            raise ValueError(
                "RAG uses IndexFlatIP and relies on unit-norm embeddings. "
                "Instantiate EmbeddingModel(normalize=True)."
            )
        
        logger.info("EmbeddingModel initialized successfully.")

    # ---------------------------------------------------------------------------------------
    # Internal utilities
    # ---------------------------------------------------------------------------------------   
    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """Apply row-wise L2 normalization (makes vectors unit length)."""
        logger.debug("Performing L2 normalization on embeddings...")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        return matrix / norms

    def _mean_pool(self, last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Mean-pool token embeddings, masking out padding tokens."""
        logger.debug("Applying mean pooling to last_hidden_state...")
        mask = np.expand_dims(attention_mask, -1).astype(np.float32)
        masked = last_hidden_state * mask
        summed = masked.sum(axis=1)
        denom = np.clip(mask.sum(axis=1), 1e-9, None)
        pooled = summed / denom
        if self.normalize:
            pooled = self._l2_normalize(pooled)
        return pooled

    def _forward(self, inputs: dict) -> np.ndarray:
        """Run forward pass through ONNX model and extract last_hidden_state."""
        logger.debug("Running forward pass through ONNX model...")
        outputs = self.model(**inputs)
        last_hidden = None

        # Handle different output formats
        if isinstance(outputs, dict):
            last_hidden = outputs.get("last_hidden_state", None)
        if last_hidden is None and hasattr(outputs, "last_hidden_state"):
            last_hidden = outputs.last_hidden_state
        if last_hidden is None and isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            last_hidden = outputs[0]
        if last_hidden is None:
            logger.error("ORT model did not return last_hidden_state.")
            raise RuntimeError("ORT model did not return last_hidden_state.")
        return last_hidden

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """Tokenize, run forward pass, mean-pool, normalize, return embeddings as Python lists."""
        logger.info(f"Encoding {len(texts)} text(s) into embeddings...")

        # Tokenize with truncation/padding to max_seq_len
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="np",
        )
 
        # Forward pass + mean pooling
        last_hidden = self._forward(inputs)
        pooled = self._mean_pool(last_hidden, inputs["attention_mask"])  

        # Make sure output is FAISS-friendly (contiguous, float32)
        pooled = np.ascontiguousarray(pooled, dtype=np.float32)
        logger.debug("Embeddings generated successfully.")
        return pooled.tolist()

    # ----------------------------------------------------------------------------------------------------------------
    # LangChain API
    # ----------------------------------------------------------------------------------------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (multi-row embeddings)."""
        logger.info(f"Embedding {len(texts)} document(s)...")
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query (returns one vector)."""
        logger.info(f"Embedding single query: '{text[:50]}...'")
        return self._encode([text])[0]

    # ----------------------------------------------------------------------------------------------------------------
    # Convenience aliases (Public APIs)
    # ----------------------------------------------------------------------------------------------------------------
    def embed_multiple_documents(self, texts: List[str]) -> List[List[float]]:  
        """Alias for embed_documents (used for JSON batches)."""
        logger.info(f"Embedding multiple documents: {len(texts)} items.")
        return self.embed_documents(texts)

    def embed_single_document(self, text: str) -> List[float]:  
        """Alias for embed_query (used for user query strings)."""
        logger.info(f"Embedding single document: '{text[:50]}...'")
        return self.embed_query(text)
