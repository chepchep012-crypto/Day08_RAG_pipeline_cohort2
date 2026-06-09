import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# =============================================================================
# CONFIGURATION — Đồng bộ chính xác với Vector Store ở Task 4
# =============================================================================
VECTOR_DB_DIR = "data/vector_store"
COLLECTION_NAME = "vietnamese_legal_and_news"
MODEL_NAME = "BAAI/bge-m3"

# =============================================================================
# IMPLEMENTATION
# =============================================================================

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity trên cơ sở dữ liệu ChromaDB.

    Args:
        query: Câu truy vấn bằng tiếng Việt thuần túy.
        top_k: Số lượng kết quả tối đa muốn trả về.

    Returns:
        List of {
            'content': str,      # Nội dung chunk văn bản
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Xếp hạng giảm dần theo score (Similarity cao nhất nằm đầu bảng).
    """
    # Kiểm tra xem Vector Store đã được khởi tạo qua Task 4 chưa
    if not os.path.exists(VECTOR_DB_DIR):
        print(f"[!] Không tìm thấy cơ sở dữ liệu tại {VECTOR_DB_DIR}. Vui lòng chạy Task 4 trước.")
        return []

    # Kết nối tới cơ sở dữ liệu ChromaDB cục bộ
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    # Bước 1: Khởi tạo hàm nhúng cục bộ (Sử dụng cấu hình CPU an toàn cho máy local)
    # ChromaDB nhận embedding_function này và tự động đảm nhiệm việc biến đổi `query` thành vector ngầm định
    bge_embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME,
        config_kwargs={"device": "cpu"}
    )

    try:
        # Lấy Collection dữ liệu ra cùng với hàm nhúng tương ứng
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=bge_embedding_function
        )

        # Bước 2: Truy vấn không gian Vector (Query Vector Store)
        # Chúng ta truyền trực tiếp chuỗi văn bản vào tham số `query_texts`.
        # Bộ công cụ ChromaDB sẽ tự động chạy embedding model, tạo vector và tính toán Cosine Distance.
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Bước 3: Chuẩn hóa dữ liệu trả về theo đúng cấu trúc Interface yêu cầu
        search_outputs = []
        
        # Kiểm tra xem hệ thống có trả về dữ liệu khớp hay không
        if results and results["documents"] and len(results["documents"][0]) > 0:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for doc, meta, dist in zip(documents, metadatas, distances):
                # Chuyển đổi công thức toán học: Cosine Similarity = 1.0 - Cosine Distance
                # Điều này đáp ứng tiêu chuẩn: Đạt độ tương đồng càng cao, score càng lớn.
                similarity_score = 1.0 - dist

                search_outputs.append({
                    "content": doc,
                    "score": round(float(similarity_score), 4), # Làm tròn 4 chữ số thập phân cho gọn sạch
                    "metadata": meta
                })

            # Đảm bảo danh sách kết quả được sắp xếp giảm dần theo điểm số (Sorted by score descending)
            search_outputs = sorted(search_outputs, key=lambda x: x["score"], reverse=True)

        return search_outputs

    except Exception as e:
        print(f"[!] Xảy ra lỗi trong quá trình thực thi Semantic Search: {str(e)}")
        return []
