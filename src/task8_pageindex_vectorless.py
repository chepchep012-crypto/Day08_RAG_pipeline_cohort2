"""
Task 8 — PageIndex Vectorless RAG (fallback local khi không có API key).
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _local_vectorless_search(query: str, top_k: int = 5) -> list[dict]:
    """Fallback: tìm kiếm dựa trên keyword match trong markdown files."""
    if not STANDARDIZED_DIR.exists():
        return []

    query_tokens = set(re.findall(r"\w+", query.lower()))
    scored = []

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]

        for para in paragraphs:
            doc_tokens = set(re.findall(r"\w+", para.lower()))
            overlap = len(query_tokens & doc_tokens)
            if overlap > 0:
                score = overlap / max(len(query_tokens), 1)
                scored.append({
                    "content": para,
                    "score": score,
                    "metadata": {"source": md_file.name, "type": md_file.parent.name},
                    "source": "pageindex",
                })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def upload_documents():
    """Upload documents lên PageIndex (hoặc skip nếu không có API key)."""
    if not PAGEINDEX_API_KEY:
        print("⚠ Không có PAGEINDEX_API_KEY — dùng local fallback")
        return

    try:
        from pageindex import PageIndex

        pi = PageIndex(api_key=PAGEINDEX_API_KEY)
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            pi.upload(
                content=content,
                metadata={"filename": md_file.name, "type": md_file.parent.name},
            )
            print(f"  ✓ Uploaded: {md_file.name}")
    except Exception as e:
        print(f"⚠ PageIndex upload failed: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless retrieval — PageIndex API hoặc local fallback."""
    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndex

            pi = PageIndex(api_key=PAGEINDEX_API_KEY)
            results = pi.query(query=query, top_k=top_k)
            return [
                {
                    "content": r.text,
                    "score": r.score,
                    "metadata": getattr(r, "metadata", {}),
                    "source": "pageindex",
                }
                for r in results
            ]
        except Exception:
            pass

    return _local_vectorless_search(query, top_k)


if __name__ == "__main__":
    results = pageindex_search("hình phạt sử dụng ma túy", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
