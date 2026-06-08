"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.
"""

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank, rerank_rrf
from src.task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Retrieval pipeline hoàn chỉnh với fallback logic."""
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    if dense_results or sparse_results:
        merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
        for item in merged:
            item["source"] = "hybrid"

        if use_reranking and merged:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
            for item in final_results:
                item["source"] = "hybrid"
        else:
            final_results = merged[:top_k]

        if final_results and final_results[0]["score"] >= score_threshold:
            return final_results[:top_k]

    fallback = pageindex_search(query, top_k=top_k)
    return fallback


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma túy",
        "Nghệ sĩ nào bị bắt vì sử dụng ma túy năm 2024",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
