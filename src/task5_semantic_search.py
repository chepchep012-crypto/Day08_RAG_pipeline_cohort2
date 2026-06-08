"""
Task 5 — Semantic Search Module (dense retrieval trên local store).
"""

import numpy as np

from src.common.store import STORE_PATH, cosine_similarity, get_store
from src.task4_chunking_indexing import EMBEDDING_DIM, EMBEDDING_MODEL, _embed_texts


def _ensure_index():
    """Build index nếu chưa có."""
    store = get_store()
    if not store.get("chunks"):
        from src.task4_chunking_indexing import run_pipeline
        run_pipeline()


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity trên local store.
    """
    _ensure_index()
    store = get_store()
    chunks = store["chunks"]
    embeddings = np.array(store["embeddings"], dtype=float)

    if not chunks:
        return []

    query_emb, _ = _embed_texts([query])
    query_vec = np.array(query_emb[0], dtype=float)

    scores = cosine_similarity(query_vec, embeddings)

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "content": chunks[idx]["content"],
            "score": float(scores[idx]),
            "metadata": chunks[idx].get("metadata", {}),
        })
    return results


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
