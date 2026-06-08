"""
Shared vector store & corpus — dùng file pickle local (không cần Weaviate/Docker).
"""

import pickle
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent.parent.parent
INDEX_DIR = PROJECT_DIR / "data" / "index"
STORE_PATH = INDEX_DIR / "store.pkl"

_store_cache: dict | None = None


def get_store() -> dict:
    """Load hoặc khởi tạo store: {chunks, embeddings, model_name}."""
    global _store_cache
    if _store_cache is not None:
        return _store_cache

    if STORE_PATH.exists():
        with open(STORE_PATH, "rb") as f:
            _store_cache = pickle.load(f)
        return _store_cache

    _store_cache = {"chunks": [], "embeddings": None, "model_name": ""}
    return _store_cache


def save_store(store: dict):
    """Persist store to disk."""
    global _store_cache
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(STORE_PATH, "wb") as f:
        pickle.dump(store, f)
    _store_cache = store


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity giữa 1 query vector và ma trận embeddings."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    return (matrix / norms) @ query_norm
