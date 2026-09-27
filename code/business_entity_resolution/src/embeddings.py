"""
ML Challenge 2026: Business Entity Resolution
Embedding Generation Module

Generates vector representations for business records combining
business name, business address, and country for semantic candidate retrieval.
Supports lightweight dense transformer models and sparse hashing vectors.
"""

import numpy as np


def build_composite_text(record):
    """
    Construct a unified textual representation from record fields:
    business_name | business_address | country
    """
    name = str(record.get("business_name", "") or "").strip()
    addr = str(record.get("business_address", "") or "").strip()
    country = str(record.get("country", "") or "").strip()
    return f"{name} | {addr} | {country}"


class RecordEmbedder:
    """
    Generates normalized embedding vectors for semantic similarity and FAISS indexing.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                print(f"Warning: Could not load SentenceTransformer ({e}). Falling back to hashing.", flush=True)
                self._model = False

    def encode(self, texts, batch_size=64, normalize=True):
        """
        Encode a list of text strings into L2-normalized float32 numpy vectors.
        """
        self._load_model()
        if self._model:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            return embeddings.astype(np.float32)
        else:
            # Deterministic character-ngram hashing fallback if model unavailable
            from sklearn.feature_extraction.text import HashingVectorizer
            hv = HashingVectorizer(n_features=256, analyzer="char_wb", ngram_range=(3, 4), norm="l2")
            return hv.transform(texts).toarray().astype(np.float32)
