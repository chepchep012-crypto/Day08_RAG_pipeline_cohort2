"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
import time
from pageindex import PageIndexClient
from dotenv import load_dotenv

load_dotenv()

# =====================================================================
# CONFIGURATION & INITIALIZATION
# =====================================================================
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")

# Khởi tạo PageIndexClient
pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

# ID tài liệu sau khi upload thành công sẽ được lưu lại để query cứu cánh
# (Trong thực tế bạn có thể lưu ID này vào DB hoặc biến môi trường)
GLOBAL_DOC_ID = None 

# =====================================================================

def upload_and_index_document(file_path: str) -> str:
    if not os.path.exists(file_path):
        print(f"[!] File không tồn tại: {file_path}")
        return None
        
    print(f"[*] Đang upload tài liệu lên PageIndex: {os.path.basename(file_path)}")
    
    try:
        # 1. Dùng 'submit_document' để tải file lên hệ thống
        # Hàm này thường trả về một dictionary hoặc object chứa thông tin document
        response = pi_client.submit_document(file_path=file_path)
        
        # Lấy doc_id từ kết quả trả về (thử lấy theo cả dạng object lẫn dict để tránh lỗi)
        if isinstance(response, dict):
            doc_id = response.get("document_id") or response.get("id")
        else:
            doc_id = getattr(response, "document_id", None) or getattr(response, "id", None)
            
        print(f"[+] Upload thành công! Khởi tạo mã Doc ID: {doc_id}")
        
        # 2. Dùng 'is_retrieval_ready' để kiểm tra xem hệ thống đã build xong cây chưa
        print("[*] Server đang phân tích cấu trúc cây tài liệu (Vui lòng đợi)...")
        while True:
            is_ready = pi_client.is_retrieval_ready(document_id=doc_id)
            
            # Hàm is_retrieval_ready thường trả về True/False trực tiếp
            if is_ready:
                print("[SUCCESS] Cấu trúc tài liệu đã được index hoàn tất!")
                break
            
            time.sleep(5) # Kiểm tra lại sau mỗi 5 giây
            
        return doc_id
    except Exception as e:
        print(f"[!] Lỗi xảy ra trong quá trình Upload/Index: {str(e)}")
        return None


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    global GLOBAL_DOC_ID
    if not GLOBAL_DOC_ID:
        print("[!] Chưa có tài liệu nào được chỉ định để query. Hãy chạy upload trước.")
        return []
        
    print(f"[*] [FALLBACK MODE] Đang kích hoạt PageIndex Vectorless Search cho câu hỏi: '{query}'")
    
    try:
        # 3. Dùng 'get_retrieval' để thực hiện Vectorless Retrieval lấy context liên quan
        # (Nếu get_retrieval đòi hỏi tham số khác, bạn có thể truyền document_id và query)
        search_results = pi_client.get_retrieval(
            document_id=GLOBAL_DOC_ID,
            query=query,
            top_k=top_k  # hoặc tùy biến theo tham số SDK như size/limit nếu có
        )
        
        formatted_outputs = []
        
        # Xử lý bóc tách dữ liệu trả về từ get_retrieval
        if search_results:
            # Thông thường kết quả trả về là một danh sách các đoạn text/pages liên quan
            for idx, node in enumerate(search_results):
                if isinstance(node, dict):
                    content = node.get("content") or node.get("text") or ""
                    metadata = node.get("metadata", {})
                    score = node.get("score", 1.0 - (idx * 0.1))
                else:
                    content = getattr(node, "content", "") or getattr(node, "text", "")
                    metadata = getattr(node, "metadata", {})
                    score = getattr(node, "score", 1.0 - (idx * 0.1))
                
                # Đồng bộ cấu trúc metadata đầu ra theo chuẩn hệ thống
                formatted_metadata = {
                    "source": "PageIndex Vectorless Engine",
                    "doc_id": GLOBAL_DOC_ID,
                    "original_metadata": metadata
                }
                
                formatted_outputs.append({
                    "content": content,
                    "score": round(float(score), 4),
                    "metadata": formatted_metadata
                })
                
        # Sắp xếp giảm dần theo điểm số trước khi trả về
        formatted_outputs = sorted(formatted_outputs, key=lambda x: x["score"], reverse=True)
        return formatted_outputs

    except Exception as e:
        print(f"[!] Lỗi xảy ra khi thực hiện Vectorless Search: {str(e)}")
        return []