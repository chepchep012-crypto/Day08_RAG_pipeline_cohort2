"""
Task 6 — Lexical Search Module (BM25).
"""

import numpy as np
from rank_bm25 import BM25Okapi

from src.task4_chunking_indexing import load_documents, chunk_documents

_bm25: BM25Okapi | None = None
CORPUS: list[dict] = []


def _build_index():
    """Xây BM25 index từ standardized documents."""
    global _bm25, CORPUS

    if _bm25 is not None:
        return

    docs = load_documents()
    if not docs:
        CORPUS = []
        _bm25 = BM25Okapi([[]])
        return

    CORPUS = chunk_documents(docs)
    tokenized = [doc["content"].lower().split() for doc in CORPUS]
    _bm25 = BM25Okapi(tokenized)


def build_bm25_index(corpus: list[dict]):
    """Xây dựng BM25 index từ corpus."""
    global _bm25, CORPUS
    CORPUS = corpus
    tokenized = [doc["content"].lower().split() for doc in corpus]
    _bm25 = BM25Okapi(tokenized)
    return _bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Tìm kiếm từ khóa sử dụng BM25."""
    _build_index()

    if not CORPUS:
        return []

    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx].get("metadata", {}),
            })
    return results


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
