"""Load .env reliably — override=True de ghi de bien moi truong cu."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_DIR / ".env"


def load_project_env() -> Path:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=True, encoding="utf-8")
    return ENV_PATH


def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    return key.strip().strip('"').strip("'")


def get_openai_base_url() -> str | None:
    url = os.getenv("OPENAI_BASE_URL", "").strip().strip('"').strip("'")
    return url or None


def get_openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip().strip('"').strip("'")
