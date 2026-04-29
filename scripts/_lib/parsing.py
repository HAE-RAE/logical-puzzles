"""Reasoning-trace and final-answer parsing helpers."""

import re
from typing import Optional, Tuple


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
FINAL_RE = re.compile(
    r"[Ff]inal\s*[Aa]nswer\s*[:：]\s*(.+?)(?:\n|$)", re.IGNORECASE
)


def parse_thinking_response(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Split a model output into (reasoning, final_answer).

    reasoning = first <think>...</think> block content (None if absent).
    final_answer = first "Final Answer:" capture (None if absent).
    """
    think_match = THINK_RE.search(text)
    reasoning = think_match.group(1).strip() if think_match else None

    final_match = FINAL_RE.search(text)
    final_answer = final_match.group(1).strip() if final_match else None

    return reasoning, final_answer
