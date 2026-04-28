"""JSONL I/O helpers."""

import json
from pathlib import Path
from typing import Iterable, List, Any


def load_jsonl(path: Path | str) -> List[Any]:
    """Read a JSONL file. Skips blank lines."""
    p = Path(path)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl(path: Path | str, items: Iterable[Any], *, ensure_ascii: bool = False) -> None:
    """Write items as JSONL. Creates parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")
