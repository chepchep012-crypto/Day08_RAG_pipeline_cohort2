"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai

Ghi chú: Crawl4AI cần cài thêm trình duyệt Playwright (nặng). Ở đây dùng
requests để tải HTML rồi MarkItDown (đã cần cho Task 3) để convert thẳng
response sang markdown — cho kết quả tương đương mà gọn nhẹ hơn.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import requests
from markitdown import MarkItDown

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

_md = MarkItDown()


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    # Ví dụ:
    # "https://vnexpress.net/...",
    # "https://tuoitre.vn/...",
    # "https://thanhnien.vn/...",
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929804.html",
    "https://vnexpress.net/su-nghiep-long-nhat-truoc-khi-bi-bat-vi-lien-quan-ma-tuy-5076081.html",
    "https://thanhnien.vn/ca-si-son-ngoc-minh-vua-bi-bat-vi-lien-quan-den-ma-tuy-la-ai-18526052012481811.htm",
    "https://tienphong.vn/lien-tiep-nghe-si-dung-chat-cam-post1842599.tpo"
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    # from crawl4ai import AsyncWebCrawler
    #
    # async with AsyncWebCrawler() as crawler:
    #     result = await crawler.arun(url=url)
    #     return {
    #         "url": url,
    #         "title": result.metadata.get("title", "Unknown"),
    #         "date_crawled": datetime.now().isoformat(),
    #         "content_markdown": result.markdown,
    #     }

    # Implement bằng requests + MarkItDown (xem ghi chú ở đầu file).
    response = await asyncio.to_thread(requests.get, url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    result = await asyncio.to_thread(_md.convert, response, url=url)

    return {
        "url": url,
        "title": (result.title or "Unknown").strip(),
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": result.text_content,
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:
            print(f"  ✗ Lỗi khi crawl {url}: {exc}")
            continue

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath}")

        await asyncio.sleep(1)  # tránh gửi request quá dồn dập


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
