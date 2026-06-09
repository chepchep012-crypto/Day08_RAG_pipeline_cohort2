import os
import numpy as np
from typing import Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# =============================================================================
# CONFIGURATION — Thiết lập Model Reranker Local chuyên dụng
# =============================================================================
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"

# Biến lưu trữ Model và Tokenizer dạng Lazy Loading để tránh nạp lại nhiều lần làm chậm RAM
_RERANK_MODEL = None
_RERANK_TOKENIZER = None

# =============================================================================
# CHỨC NĂNG BỔ TRỢ (UTILITIES)
# =============================================================================

def _get_local_reranker():
    """Khởi tạo và lưu cache bộ đôi Model & Tokenizer cho Cross-Encoder."""
    global _RERANK_MODEL, _RERANK_TOKENIZER
    if _RERANK_MODEL is None or _RERANK_TOKENIZER is None:
        print(f"[*] Đang tải Reranker Model: {RERANK_MODEL_NAME} từ HuggingFace...")
        _RERANK_TOKENIZER = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME)
        _RERANK_MODEL = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_NAME)
        
        # Tự động chọn thiết bị tính toán tối ưu (Ưu tiên GPU CUDA nếu có)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _RERANK_MODEL.to(device)
        _RERANK_MODEL.eval() # Chuyển sang chế độ Inference
    return _RERANK_MODEL, _RERANK_TOKENIZER


def cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    """Tính toán độ tương đồng Cosine giữa hai vector embedding."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# =============================================================================
# CORE IMPLEMENTATION (3 PHƯƠNG PHÁP RERANK)
# =============================================================================

def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Xếp hạng lại các ứng viên bằng mô hình Deep Learning Cross-Encoder Local.
    """
    if not candidates:
        return []

    model, tokenizer = _get_local_reranker()
    device = model.device

    # Chuẩn bị cặp dữ liệu đầu vào dạng (Query, Document) cho mô hình Cross-Encoder
    pairs = [[query, c["content"]] for c in candidates]

    with torch.no_grad():
        # Tokenize toàn bộ các cặp văn bản đồng thời
        inputs = tokenizer(
            pairs, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(device)
        
        # Mô hình tính toán logits biểu thị mức độ liên quan ngữ cảnh sâu
        scores = model(**inputs).logits.view(-1).cpu().numpy()

    # Cập nhật lại score mới vào bản sao danh sách ứng viên
    reranked_candidates = []
    for idx, score in enumerate(scores):
        updated_item = candidates[idx].copy()
        # Chuyển đổi điểm logit về dạng xác suất sigmoid hoặc giữ nguyên phân cấp tuyến tính
        updated_item["score"] = round(float(score), 4)
        reranked_candidates.append(updated_item)

    # Sắp xếp giảm dần theo điểm số rerank mới thu được
    reranked_candidates = sorted(reranked_candidates, key=lambda x: x["score"], reverse=True)
    return reranked_candidates[:top_k]


def rerank_mmr(query_embedding: list[float], candidates: list[dict], top_k: int = 5, lambda_param: float = 0.7) -> list[dict]:
    """
    Maximal Marginal Relevance — Cân bằng giữa độ liên quan (Relevance) và độ đa dạng (Diversity).
    Ngăn chặn việc LLM nhận các đoạn văn bị trùng lặp nội dung với nhau.
    """
    if not candidates:
        return []
        
    # Kiểm tra tính sẵn sàng của mảng Vector Embedding trong các chunk
    if "embedding" not in candidates[0]:
        raise ValueError("[!] MMR đòi hỏi các chunk ứng viên phải chứa trường dữ liệu 'embedding'.")

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            # 1. Tính toán độ tương đồng giữa Query và Chunk hiện tại
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            # 2. Tìm kiếm độ tương đồng lớn nhất của chunk này với nhóm các chunk ĐÃ CHỌN trước đó
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # 3. Áp dụng công thức chuẩn MMR
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        # Đưa ứng viên tối ưu nhất vòng này vào danh sách trích xuất
        selected.append(best_idx)
        remaining.remove(best_idx)

    # Đóng gói và trả về kết quả
    results = [candidates[i].copy() for i in selected]
    return results


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion — Trộn điểm và xếp hạng lại tài liệu từ nhiều bộ tìm kiếm khác nhau 
    (Ví dụ: Trộn danh sách Semantic Search với danh sách Lexical Search).
    """
    rrf_scores = {}  # content -> rrf_score
    content_map = {} # content -> item_dict

    # Duyệt qua từng danh sách xếp hạng thành phần (ví dụ: list_1 của Dense, list_2 của BM25)
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            
            # Áp dụng công thức chuẩn: RRF(d) = Σ 1 / (k + rank(d))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            # Lưu lại dữ liệu thô để phục vụ việc map kết quả đầu ra
            if key not in content_map:
                content_map[key] = item

    # Sắp xếp các chuỗi văn bản giảm dần dựa theo điểm số RRF tổng hợp
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        final_item = content_map[content].copy()
        final_item["score"] = round(float(score), 6) # Gán điểm số RRF phối hợp
        results.append(final_item)

    return results

# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def rerank(query: str, candidates: list[dict], top_k: int = 5, method: str = "cross_encoder", **kwargs) -> list[dict]:
    """
    Giao diện điều phối Rerank thống nhất cho toàn bộ hệ thống RAG Pipeline.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
        
    elif method == "mmr":
        query_embedding = kwargs.get("query_embedding")
        if query_embedding is None:
            raise ValueError("[!] Để chạy MMR, bạn phải truyền thêm tham số 'query_embedding' dạng list[float].")
        lambda_param = kwargs.get("lambda_param", 0.7)
        return rerank_mmr(query_embedding, candidates, top_k, lambda_param)
        
    elif method == "rrf":
        ranked_lists = kwargs.get("ranked_lists")
        if ranked_lists is None:
            raise ValueError("[!] Để chạy RRF, bạn phải truyền tham số 'ranked_lists' chứa danh sách các bảng hạng.")
        smoothing_k = kwargs.get("k", 60)
        return rerank_rrf(ranked_lists, top_k, smoothing_k)
        
    else:
        raise ValueError(f"[!] Không tìm thấy phương pháp xếp hạng nào mang tên: {method}")