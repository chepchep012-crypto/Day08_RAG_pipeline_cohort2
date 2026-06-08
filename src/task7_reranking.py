"""
Task 7 — Reranking Module (keyword overlap cross-encoder fallback, không cần API).
"""

import re
from typing import Optional


def _keyword_overlap_score(query: str, document: str) -> float:
    """Tính điểm relevance dựa trên keyword overlap (fallback reranker)."""
    query_tokens = set(re.findall(r"\w+", query.lower()))
    doc_tokens = set(re.findall(r"\w+", document.lower()))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    return overlap / len(query_tokens)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Rerank bằng keyword overlap (local, không cần Jina API)."""
    scored = []
    for c in candidates:
        rerank_score = _keyword_overlap_score(query, c["content"])
        # Kết hợp score gốc (30%) + rerank score (70%)
        combined = 0.3 * c.get("score", 0) + 0.7 * rerank_score
        scored.append({**c, "score": combined})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Maximal Marginal Relevance."""
    import numpy as np

    def _cos(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            emb = candidates[idx].get("embedding")
            relevance = (
                _cos(query_embedding, emb)
                if emb
                else candidates[idx].get("score", 0)
            )
            max_sim = 0.0
            for sel_idx in selected:
                sel_emb = candidates[sel_idx].get("embedding")
                if emb and sel_emb:
                    max_sim = max(max_sim, _cos(emb, sel_emb))

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [{**candidates[i], "score": candidates[i].get("score", 0)} for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker."""
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """Unified reranking interface."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
    else:
        raise ValueError(f"Unknown rerank method: {method}")
