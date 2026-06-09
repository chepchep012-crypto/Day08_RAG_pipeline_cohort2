"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

import os
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

# =============================================================================
# CONFIGURATION
# =============================================================================
SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh tích hợp Hybrid Search và cơ chế Fallback thông minh.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn của người dùng.
        top_k: Số lượng kết quả tinh hoa cuối cùng muốn giữ lại.
        score_threshold: Ngưỡng điểm tối thiểu để chấp nhận kết quả Hybrid.
        use_reranking: Cấu hình bật/tắt tầng Cross-Encoder Reranker.

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Chạy song song/đồng thời Semantic Search và Lexical Search
    # Lấy số lượng ứng viên rộng gấp đôi (top_k * 2) để làm phong phú dữ liệu trước khi lọc/fusion
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Step 2: Hợp nhất (Merge) kết quả của 2 luồng tìm kiếm bằng thuật toán RRF
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    
    # Gán nhãn nguồn gốc cho các tài liệu đi ra từ luồng kết hợp
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Tiến hành xếp hạng lại (Rerank) chuyên sâu nhằm tối ưu điểm số ngữ cảnh
    if use_reranking and merged:
        # Sử dụng Cross-Encoder để chấm lại điểm tương quan thực tế giữa Query và Chunk text
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        # Nếu tắt Reranking, cắt trực tiếp top_k phần tử dẫn đầu từ danh sách RRF
        final_results = merged[:top_k]

    # Step 4: Kiểm tra chất lượng điểm số (Threshold Check) và điều hướng Fallback
    # Lấy ra điểm số của ứng viên đứng đầu bảng để làm tiêu chí đánh giá chất lượng
    best_score = final_results[0]["score"] if final_results else 0.0

    if not final_results or best_score < score_threshold:
        print(f"  [⚠️ WARNING] Hybrid best score ({best_score:.3f}) "
              f"< threshold ({score_threshold}). Kích hoạt Fallback → PageIndex Vectorless.")
        
        # Gọi công cụ trích xuất cấu trúc cây phân cấp của PageIndex (Task 8) làm cứu cánh
        fallback_results = pageindex_search(query, top_k=top_k)
        
        # Đảm bảo các chunk đi từ PageIndex có trường 'source' chuẩn xác để đồng bộ với pipeline
        for r in fallback_results:
            r["source"] = "pageindex"
            
        return fallback_results[:top_k]

    # Step 5: Trả về kết quả Hybrid đạt chuẩn chất lượng
    return final_results[:top_k]