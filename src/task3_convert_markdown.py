"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path
from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Duyệt tài liệu pháp lý thô chuyển sang định dạng Markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    
    if not legal_dir.exists():
        print(f"[!] Thư mục nguồn không tồn tại: {legal_dir}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    md_converter = MarkItDown()

    for filepath in legal_dir.iterdir():
        if filepath.is_file() and filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            try:
                result = md_converter.convert(str(filepath))
                output_path = output_dir / f"{filepath.stem}.md"
                output_path.write_text(result.text_content, encoding="utf-8")
                print(f"  ✓ Saved: {output_path.name}")
            except Exception as e:
                print(f"[!] Lỗi khi xử lý file văn bản pháp lý {filepath.name}: {str(e)}")


def convert_news_articles():
    """Đọc file JSON cấu trúc phẳng trích xuất chuyển thành file .md."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    
    if not news_dir.exists():
        print(f"[!] Thư mục nguồn không tồn tại: {news_dir}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.is_file() and filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            try:
                raw_data = json.loads(filepath.read_text(encoding="utf-8"))
                output_path = output_dir / f"{filepath.stem}.md"
                
                # Đọc trực tiếp từ lớp ngoài cùng của JSON phẳng mới
                title = raw_data.get("title", "Unknown Title")
                url = raw_data.get("url", "N/A")
                date_crawled = raw_data.get("date_crawled", "N/A")
                content = raw_data.get("content", "")
                
                header = f"# {title}\n\n"
                header += f"**Source:** {url}\n"
                header += f"**Crawled:** {date_crawled}\n\n---\n\n"
                
                full_content = header + content
                output_path.write_text(full_content, encoding="utf-8")
                print(f"  ✓ Saved: {output_path.name}")
            except Exception as e:
                print(f"[!] Lỗi khi chuẩn hóa cấu trúc dữ liệu JSON {filepath.name}: {str(e)}")


def convert_all():
    """Hàm điều phối kích hoạt chu trình chuẩn hóa."""
    print("=" * 60)
    print("Task 3: Convert Data to Markdown (Microsoft MarkItDown)")
    print("=" * 60)

    print("\n--- Processing: Legal Documents ---")
    convert_legal_docs()

    print("\n--- Processing: News Articles ---")
    convert_news_articles()

    print(f"\n[*] Tiến trình hoàn tất. Dữ liệu đầu ra sạch tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()