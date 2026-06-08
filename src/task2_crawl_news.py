"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Bài báo thật từ VnExpress về nghệ sĩ liên quan ma tuý
ARTICLE_URLS = [
    "https://vnexpress.net/nguoi-mau-andrea-aybar-va-ca-si-chi-dan-bi-bat-4814295.html",
    "https://vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929804.html",
    "https://ngoisao.vnexpress.net/chi-dan-va-andrea-aybar-bi-khoi-to-vi-to-chuc-su-dung-ma-tuy-4815983.html",
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
    "https://vnexpress.net/su-nghiep-long-nhat-truoc-khi-bi-bat-vi-lien-quan-ma-tuy-5076081.html",
]


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _slug_from_url(url: str) -> str:
    """Tạo tên file từ URL."""
    slug = url.rstrip("/").split("/")[-1].replace(".html", "")
    return re.sub(r"[^\w\-]", "_", slug)[:80]


def _extract_title(html: str) -> str:
    """Lấy tiêu đề từ HTML."""
    for pattern in (
        r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"',
        r"<title>([^<]+)</title>",
        r'<h1[^>]*class="[^"]*title[^"]*"[^>]*>([^<]+)</h1>',
    ):
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return "Unknown"


def _extract_content(html: str) -> str:
    """Lấy nội dung chính từ HTML VnExpress (fallback khi không có Crawl4AI)."""
    # VnExpress: nội dung nằm trong các thẻ <p> trong vùng article
    article_match = re.search(
        r'<article[^>]*>(.*?)</article>',
        html,
        re.DOTALL | re.IGNORECASE,
    )
    block = article_match.group(1) if article_match else html

    paragraphs = re.findall(
        r"<p[^>]*class=\"[^\"]*(?:Normal|description|fck_detail)[^\"]*\"[^>]*>(.*?)</p>",
        block,
        re.DOTALL | re.IGNORECASE,
    )
    if not paragraphs:
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", block, re.DOTALL)

    lines = []
    for p in paragraphs:
        text = re.sub(r"<[^>]+>", "", p)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 30:
            lines.append(text)

    return "\n\n".join(lines) if lines else ""


def _crawl_with_requests(url: str) -> dict:
    """Fallback crawler dùng requests khi Crawl4AI chưa cài hoặc lỗi."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    html = response.text

    title = _extract_title(html)
    content = _extract_content(html)

    if not content:
        content = re.sub(r"<[^>]+>", " ", html)
        content = re.sub(r"\s+", " ", content).strip()[:5000]

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": f"# {title}\n\n{content}",
    }


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
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            title = "Unknown"
            if result.metadata:
                title = result.metadata.get("title") or result.metadata.get("og:title", "Unknown")
            content = result.markdown or result.cleaned_html or ""
            if len(content.strip()) < 100:
                raise ValueError("Crawl4AI returned empty content")

            return {
                "url": url,
                "title": title,
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": content,
            }
    except Exception:
        return await asyncio.to_thread(_crawl_with_requests, url)


async def crawl_all(urls: list[str] | None = None):
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()
    targets = urls or ARTICLE_URLS

    for i, url in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
            filename = f"article_{i:02d}_{_slug_from_url(url)}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  Saved: {filepath} ({len(article['content_markdown'])} chars)")
        except Exception as e:
            print(f"  Failed: {url} - {e}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("Hay dien ARTICLE_URLS truoc khi chay!")
        print("Goi y: tim bai bao tren VnExpress, Tuoi Tre, Thanh Nien, ...")
    else:
        asyncio.run(crawl_all())
