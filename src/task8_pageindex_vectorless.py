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
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
# PageIndex SDK xử lý trực tiếp file PDF gốc (OCR + sinh cây cấu trúc tài liệu),
# nên ta upload PDF gốc từ data/landing/ thay vì bản markdown đã chuẩn hoá.
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"

_client = None
_doc_ids: list[str] = []


def _get_client():
    global _client
    if _client is None:
        from pageindex import PageIndexClient
        _client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    return _client


def upload_documents() -> list[str]:
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    # TODO: Implement upload
    #
    # Tham khảo: https://github.com/VectifyAI/PageIndex
    #
    # from pageindex import PageIndex
    #
    # pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    #
    # for md_file in STANDARDIZED_DIR.rglob("*.md"):
    #     content = md_file.read_text(encoding="utf-8")
    #     pi.upload(
    #         content=content,
    #         metadata={"filename": md_file.name, "type": md_file.parent.name}
    #     )
    #     print(f"  ✓ Uploaded: {md_file.name}")
    client = _get_client()

    doc_ids = []
    for pdf_file in sorted(LANDING_DIR.rglob("*.pdf")):
        result = client.submit_document(str(pdf_file))
        doc_id = result["doc_id"]
        doc_ids.append(doc_id)
        print(f"  ✓ Uploaded: {pdf_file.name} → doc_id={doc_id}")

    # Upload bất đồng bộ (OCR + sinh cây cấu trúc) — chờ tới khi sẵn sàng cho retrieval
    for doc_id in doc_ids:
        while not client.is_retrieval_ready(doc_id):
            time.sleep(5)
        print(f"  ✓ Ready for retrieval: {doc_id}")

    global _doc_ids
    _doc_ids = doc_ids
    return doc_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    # TODO: Implement PageIndex query
    #
    # from pageindex import PageIndex
    #
    # pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    # results = pi.query(query=query, top_k=top_k)
    #
    # return [
    #     {
    #         "content": r.text,
    #         "score": r.score,
    #         "metadata": r.metadata,
    #         "source": "pageindex"
    #     }
    #     for r in results
    # ]
    client = _get_client()
    if not _doc_ids:
        upload_documents()

    results = []
    for doc_id in _doc_ids:
        submitted = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = submitted["retrieval_id"]

        # Retrieval chạy bất đồng bộ (LLM duyệt cây cấu trúc tài liệu) — poll tới khi xong
        while True:
            retrieval = client.get_retrieval(retrieval_id)
            if retrieval.get("status") in ("completed", "done", "ready"):
                break
            time.sleep(2)

        nodes = retrieval.get("results") or retrieval.get("nodes") or []
        for rank, node in enumerate(nodes, start=1):
            results.append({
                "content": node.get("content") or node.get("text", ""),
                # PageIndex là vectorless (không có similarity score) —
                # dùng relevance_score nếu API trả về, nếu không thì suy ra từ thứ hạng
                "score": node.get("relevance_score", 1.0 / rank),
                "metadata": {
                    "doc_id": doc_id,
                    "node_id": node.get("node_id"),
                    "title": node.get("title"),
                },
                "source": "pageindex"
            })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
