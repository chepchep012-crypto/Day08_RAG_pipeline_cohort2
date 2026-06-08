"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        return

    md = MarkItDown()

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            output_path = output_dir / f"{filepath.stem}.md"
            try:
                result = md.convert(str(filepath))
                text = result.text_content or ""
            except Exception:
                text = ""

            if len(text.strip()) < 100 and filepath.suffix.lower() == ".docx":
                import zipfile
                import re as _re

                with zipfile.ZipFile(filepath) as zf:
                    xml = zf.read("word/document.xml").decode("utf-8")
                text = _re.sub(r"<[^>]+>", "\n", xml)
                text = _re.sub(r"\n+", "\n", text).strip()

            if len(text.strip()) < 100:
                text = filepath.read_bytes().decode("latin-1", errors="ignore")
                text = "".join(c for c in text if c.isprintable() or c in "\n")

            output_path.write_text(text, encoding="utf-8")
            print(f"  Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        return

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            content = header + data.get("content_markdown", "")
            output_path.write_text(content, encoding="utf-8")
            print(f"  Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\nDone! Output at:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
