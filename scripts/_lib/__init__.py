"""Shared helpers for scripts/.

Centralizes PROJECT_ROOT, jsonl I/O, OpenAI client init, and answer parsing
so individual scripts stop redefining the same boilerplate.
"""

from .paths import PROJECT_ROOT
from .io import load_jsonl, save_jsonl
from .api_client import get_openai_client, ensure_dotenv
from .parsing import THINK_RE, FINAL_RE, parse_thinking_response
from .prompts import THINK_FORMAT_INSTRUCTION
from .logging import setup_logger

__all__ = [
    "PROJECT_ROOT",
    "load_jsonl",
    "save_jsonl",
    "get_openai_client",
    "ensure_dotenv",
    "THINK_RE",
    "FINAL_RE",
    "parse_thinking_response",
    "THINK_FORMAT_INSTRUCTION",
    "setup_logger",
]
