"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.
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


# Danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://vnexpress.net/toa-an-chua-the-xac-dinh-cuu-pho-chu-tich-phuong-khoi-xuong-dung-ma-tuy-5065326.html",
    "https://vnexpress.net/227-nguoi-bi-truy-to-trong-vu-4-tiep-vien-hang-khong-xach-ma-tuy-5057648.html",
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929804.html",
    "https://vnexpress.net/su-nghiep-long-nhat-truoc-khi-bi-bat-vi-lien-quan-ma-tuy-5076081.html",
    "https://vnexpress.net/ong-lao-ngoi-xe-lan-buon-8-kg-ma-tuy-qua-bien-gioi-4937421.html",
    "https://vnexpress.net/tiktoker-long-tong-bi-bat-tai-san-bay-4995112.html",
    "https://vnexpress.net/rapper-binh-gold-tiep-tuc-duong-tinh-voi-ma-tuy-lai-cuop-taxi-4919259.html"
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về cấu trúc phẳng đáp ứng chính xác bộ test suite.
    """
    response = await asyncio.to_thread(requests.get, url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    result = await asyncio.to_thread(_md.convert, response, url=url)

    # ĐƯA TẤT CẢ RA LỚP NGOÀI CÙNG THEO YÊU CẦU CỦA TEST SUITE
    return {
        "url": url,
        "title": (result.title or "Unknown").strip(),
        "date_crawled": datetime.now().isoformat(),
        "content": result.text_content.strip(),
        "status": "success"
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    print(f"[*] Bắt đầu tiến trình thu thập {len(ARTICLE_URLS)} bài báo...")

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Đang cào dữ liệu: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:
            print(f"  ✗ Lỗi không thể thu thập dữ liệu từ {url}: {exc}")
            continue

        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
        print(f"  ✓ Đã lưu file thành công: {filepath}")

        await asyncio.sleep(1)


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Vui lòng cấu hình danh sách ARTICLE_URLS trước khi thực thi!")
    else:
        asyncio.run(crawl_all())