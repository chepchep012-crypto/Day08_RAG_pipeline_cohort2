"""Luu / load lich su chat ra file JSON — giu sau khi reload trang."""

import json
from pathlib import Path

CHAT_DIR = Path(__file__).resolve().parent / "data"
CHAT_FILE = CHAT_DIR / "chat_history.json"
MAX_SOURCE_SNIPPET = 500


def _trim_sources(sources: list[dict] | None) -> list[dict]:
    if not sources:
        return []
    trimmed = []
    for src in sources:
        trimmed.append({
            "content": (src.get("content") or "")[:MAX_SOURCE_SNIPPET],
            "score": src.get("score", 0),
            "metadata": src.get("metadata", {}),
            "source": src.get("source", "hybrid"),
        })
    return trimmed


def load_chat_history() -> list[dict]:
    if not CHAT_FILE.exists():
        return []
    try:
        data = json.loads(CHAT_FILE.read_text(encoding="utf-8"))
        messages = data.get("messages", [])
        return messages if isinstance(messages, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_chat_history(messages: list[dict]) -> None:
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    serializable = []
    for msg in messages:
        item = {"role": msg["role"], "content": msg["content"]}
        if msg.get("sources"):
            item["sources"] = _trim_sources(msg["sources"])
        serializable.append(item)
    CHAT_FILE.write_text(
        json.dumps({"messages": serializable}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clear_chat_history() -> None:
    if CHAT_FILE.exists():
        CHAT_FILE.unlink()
