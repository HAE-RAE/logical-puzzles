"""OpenAI client init helpers."""

import os
from pathlib import Path
from typing import Optional


_dotenv_loaded = False


def ensure_dotenv(env_path: Optional[Path] = None) -> None:
    """Load .env once (idempotent). Defaults to project root."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        _dotenv_loaded = True
        return
    if env_path is not None:
        load_dotenv(env_path)
    else:
        load_dotenv()
    _dotenv_loaded = True


def get_openai_client(*, api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Return an OpenAI client. Loads .env if api_key is unset and OPENAI_API_KEY env var is missing."""
    if api_key is None and not os.environ.get("OPENAI_API_KEY"):
        ensure_dotenv()
    from openai import OpenAI
    kwargs = {}
    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url is not None:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)
