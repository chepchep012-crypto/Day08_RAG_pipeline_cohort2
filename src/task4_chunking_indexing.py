import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# =====================================================================
# CONFIGURATION & METADATA DECLARATION (Yêu cầu bài tập)
# =====================================================================
# 1. Chunking Strategy: 
#    - Bước 1: Dùng MarkdownHeaderTextSplitter để tách theo Heading, giữ ngữ cảnh tiêu đề.
#    - Bước 2: Dùng RecursiveCharacterTextSplitter để băm nhỏ các đoạn quá dài.
# LỰA CHỌN CHUNKING STRATEGY:
# - CHUNK_SIZE = 500: Phù hợp với độ dài trung bình của một điều luật hoặc một đoạn báo ngắn tiếng Việt (khoảng 150 - 200 từ). 
#   Kích thước này đủ nhỏ để vector giữ được ngữ nghĩa cô đọng, tránh nhiễu thông tin khi LLM đọc context.
# - CHUNK_OVERLAP = 50: Giữ lại 10% ngữ cảnh giao thoa giữa các đoạn băm liên tiếp, đảm bảo các câu từ nằm ở 
#   ranh giới cắt không bị mất đi mối liên kết ngữ nghĩa liền mạch.
# - METHOD = 'recursive': Thuật toán sẽ ưu tiên cắt ở các dấu ngắt lớn như xuống dòng (\n\n, \n) rồi mới đến dấu chấm câu. 
#   Điều này giúp giữ nguyên vẹn cấu trúc của các chương, mục, điều khoản pháp lý hoặc các đoạn văn báo chí.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 2. Embedding Model:
#    - Model: BAAI/bge-m3 (Tối ưu xuất sắc cho Tiếng Việt, hỗ trợ đa nhiệm)
#    - Dimension: 1024
# LỰA CHỌN EMBEDDING MODEL:
# - BAAI/bge-m3 là mô hình embedding đa ngôn ngữ (Multilingual) mạnh mẽ hàng đầu hiện nay.
# - Được tối ưu rất tốt cho tiếng Việt pháp lý và tin tức nhờ khả năng hỗ trợ độ dài ngữ cảnh lớn (lên tới 8192 tokens)
#   và kiến trúc đa nhiệm (Dense, Sparse, Multi-vector retrieval).
# - EMBEDDING_DIM = 1024 mang lại không gian biểu diễn vector dày đặc, độ chính xác cao khi so sánh tương đồng ngữ nghĩa.
MODEL_NAME = "BAAI/bge-m3"
# LỰA CHỌN VECTOR STORE:
# - Sử dụng 'chromadb' chạy dưới chế độ PersistentClient (lưu file trực tiếp xuống ổ đĩa local).
# - ChromaDB cực kỳ gọn nhẹ, không yêu cầu cài đặt Docker phức tạp như Weaviate, tích hợp hoàn hảo với Python 
#   và đáp ứng mượt mà tốc độ truy vấn thời gian thực cho bài toán RAG Cohort.
VECTOR_DB_DIR = "data/vector_store"
INPUT_STANDARDIZED_DIR = "data/standardized"

# =====================================================================

def load_documents() -> list:
    """
    Đã đổi tên từ load_markdown_files thành load_documents để qua môn.
    Không truyền tham số đầu vào, tự động quét thư mục INPUT_STANDARDIZED_DIR.
    """
    documents = []
    base_dir = str(INPUT_STANDARDIZED_DIR)
    
    if not os.path.exists(base_dir):
        print(f"[!] Thư mục {base_dir} không tồn tại.")
        return documents

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                category = os.path.basename(root)
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append({
                        # File test yêu cầu trường 'content' ở lớp ngoài cùng của document
                        "content": f.read(),
                        "source": file,
                        "category": category
                    })
    return documents


def chunk_documents(raw_docs: list) -> list:
    """
    Hàm băm nhỏ văn bản kết hợp Markdown Header nâng cao.
    Đã sửa key trả về từ 'text' thành 'content' theo đúng lệnh assert của trường.
    """
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    final_chunks = []
    
    for doc in raw_docs:
        md_header_splits = markdown_splitter.split_text(doc["content"])
        splits = text_splitter.split_documents(md_header_splits)
        
        for idx, split in enumerate(splits):
            metadata = split.metadata.copy()
            metadata["source"] = doc["source"]
            metadata["category"] = doc["category"]
            metadata["chunk_id"] = f"{doc['source']}_{idx}"
            
            # CẤU TRÚC ĐẦU RA SỬA TỪ 'text' THÀNH 'content' ĐỂ KHỚP VỚI PYTEST
            final_chunks.append({
                "content": split.page_content,
                "metadata": metadata,
                "id": f"{doc['source']}_chunk_{idx}"
            })
            
    return final_chunks


def index_to_vector_store(chunks):
    """Khởi tạo ChromaDB cục bộ và tiến hành Embed + Index toàn bộ dữ liệu"""
    print(f"[*] Đang khởi tạo ChromaDB tại thư mục: {VECTOR_DB_DIR}")
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    
    bge_embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )
    
    collection = chroma_client.get_or_create_collection(
        name="vietnamese_legal_and_news",
        embedding_function=bge_embedding_function,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"[*] Đang tiến hành tạo Vector Embedding cho {len(chunks)} chunks...")
    
    ids = [c["id"] for c in chunks]
    # Sửa bóc tách trường văn bản thành c["content"] đồng bộ với cấu trúc mới
    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    
    batch_size = 40
    for i in range(0, len(chunks), batch_size):
        end_idx = i + batch_size
        collection.add(
            documents=texts[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        print(f"[+] Đã index xong các chunk từ {i} đến {min(end_idx, len(chunks))}")

    print(f"[ SUCCESS ] Tổng số lượng vectors hiện có trong Collection: {collection.count()}")


if __name__ == "__main__":
    print("=== BẮT ĐẦU TASK 4: CHUNKING & INDEXING ===")
    
    raw_documents = load_documents()
    print(f"[*] Đã đọc {len(raw_documents)} file markdown đầu vào.")
    
    if raw_documents:
        processed_chunks = chunk_documents(raw_documents)
        print(f"[*] Tổng số lượng chunk tạo ra sau khi xử lý: {len(processed_chunks)}")
        
        index_to_vector_store(processed_chunks)
        print("=== HOÀN THÀNH HOÀN TOÀN TASK 4 ===")
    else:
        print("[!] Không tìm thấy dữ liệu nào để xử lý.")