import os
from pathlib import Path
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi

# =============================================================================
# CONFIGURATION & GLOBAL STATE — Đồng bộ chính xác dữ liệu từ Task 4 & Task 5
# =============================================================================
VECTOR_DB_DIR = "data/vector_store"
COLLECTION_NAME = "vietnamese_legal_and_news"

# Biến toàn cục chứa tập tài liệu (Corpus) phục vụ BM25 index
CORPUS: list[dict] = []  # Mỗi phần tử dạng: {'content': str, 'metadata': dict}
GLOBAL_BM25_INDEX = None # Giữ thực thể BM25Okapi sau khi build xong

# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_corpus_from_db():
    """
    Nạp toàn bộ dữ liệu chunks văn bản hiện có từ ChromaDB vào biến toàn cục CORPUS.
    """
    global CORPUS
    if CORPUS: # Nếu đã nạp rồi thì bỏ qua không nạp lại
        return

    if not os.path.exists(VECTOR_DB_DIR):
        print(f"[!] Không tìm thấy Vector Store tại {VECTOR_DB_DIR}. Vui lòng chạy Task 4 trước.")
        return

    # Kết nối ChromaDB local để lấy text thô và metadata
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        # Lấy toàn bộ bản ghi văn bản (documents) và thông tin đi kèm (metadatas)
        all_data = collection.get(include=["documents", "metadatas"])
        
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        
        # Đóng gói vào biến CORPUS toàn cục theo đúng cấu trúc yêu cầu
        for doc, meta in zip(documents, metadatas):
            CORPUS.append({
                "content": doc,
                "metadata": meta
            })
        print(f"[*] Đã nạp thành công {len(CORPUS)} chunks văn bản từ database vào hệ thống BM25.")
    except Exception as e:
        print(f"[!] Lỗi khi nạp dữ liệu cho Lexical Search: {str(e)}")


def clean_and_tokenize(text: str) -> list[str]:
    """
    Hàm tiền xử lý và tách từ (Tokenization) cơ bản cho tiếng Việt.
    Chuyển văn bản về chữ thường và lọc bỏ các dấu câu cơ bản gây nhiễu từ khóa.
    """
    text_cleaned = text.lower()
    for punc in [".", ",", "?", "!", ":", ";", '"', "'", "(", ")", "-", "_", "/"]:
        text_cleaned = text_cleaned.replace(punc, " ")
    return text_cleaned.split()


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng cấu trúc chỉ mục BM25 từ danh sách Corpus văn bản.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    Returns:
        Thực thể BM25Okapi đã được fit dữ liệu từ vựng.
    """
    if not corpus:
        return None
        
    # Tiến hành tokenize toàn bộ tập tài liệu
    tokenized_corpus = [clean_and_tokenize(doc["content"]) for doc in corpus]
    
    # Khởi tạo thuật toán BM25Okapi nền tảng
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa chính xác (Keyword Search) sử dụng thuật toán BM25Okapi.

    Args:
        query: Câu truy vấn hoặc mã hiệu điều luật cần tìm kiếm.
        top_k: Số lượng kết quả tối đa muốn giữ lại.

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Được sắp xếp giảm dần dựa trên điểm số BM25 (Score càng lớn càng khớp).
    """
    global CORPUS, GLOBAL_BM25_INDEX
    
    # Bước 1: Bảo đảm dữ liệu corpus đã được nạp lên RAM
    load_corpus_from_db()
    if not CORPUS:
        print("[!] Tập dữ liệu CORPUS trống. Tìm kiếm thất bại.")
        return []
        
    # Bước 2: Khởi tạo/Tái sử dụng Index (Lazy Initialization để tối ưu hiệu năng)
    if GLOBAL_BM25_INDEX is None:
        GLOBAL_BM25_INDEX = build_bm25_index(CORPUS)
        
    if GLOBAL_BM25_INDEX is None:
        return []

    # Bước 3: Tokenize câu truy vấn đầu vào và thực hiện tính toán điểm số (Get scores)
    tokenized_query = clean_and_tokenize(query)
    scores = GLOBAL_BM25_INDEX.get_scores(tokenized_query)

    # Bước 4: Trích xuất chỉ mục top_k phần tử có điểm số cao nhất bằng numpy
    # Sử dụng np.argsort để lấy mảng vị trí sắp xếp tăng dần, đảo ngược thành giảm dần [::-1] 
    # rồi cắt lấy top_k phần tử đầu bảng.
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        # Chỉ lấy những tài liệu thực sự có chứa từ khóa khớp (score > 0)
        # Tránh trả về các đoạn văn vô liên quan khi tập dữ liệu lớn hơn top_k
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": round(float(scores[idx]), 4), # Ép về float thuần Python và làm tròn cho đẹp
                "metadata": CORPUS[idx]["metadata"]
            })
            
    return results