#!/usr/bin/env python3
"""API 모델 스모크 테스트 — model_configs.yaml의 calibration_reference + api_models 전부.

하네스의 실제 코드 경로(evaluation.model.LiteLLMClient)를 그대로 사용해
키 유효성 + 모델 ID + 파라미터 변환(claude-opus-4-8의 adaptive thinking 변환 포함)을
end-to-end 검증한다. 비용 최소화를 위해 짧은 프롬프트 + max_tokens 축소.

사용: python scripts/calltest/calltest_api_models.py
"""
import concurrent.futures as cf
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.model import create_client  # noqa: E402

cfg = yaml.safe_load(open(ROOT / "run/eval/model_configs.yaml"))
models = {}
for group in ("calibration_reference", "api_models"):
    for key, e in cfg.get(group, {}).items():
        models[key] = e


def parse_kwargs(s):
    out = {}
    for pair in s.split(","):
        k, v = pair.split("=", 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        out[k] = v
    return out


def test(key, entry):
    kw = parse_kwargs(entry["gen_kwargs"])
    kw["max_tokens"] = min(kw.get("max_tokens", 2048), 2048)  # 테스트용 축소
    try:
        client = create_client(entry["model"], timeout=180, gen_kwargs=kw)
        text, meta = client.generate(
            [{"role": "user", "content": "What is 7*8? Answer with the number only."}]
        )
        thinking = "yes" if meta.get("thinking_content") else "no"
        return (
            key, "OK",
            f"answer={text.strip()[:40]!r} | tokens={meta.get('tokens')} "
            f"| thinking={thinking} | finish={meta.get('finish_reason')} "
            f"| {meta.get('latency_ms', 0):.0f}ms",
        )
    except Exception as e:
        msg = str(e).split("\n")[0][:220]
        return (key, "FAIL", f"{type(e).__name__}: {msg}")


with cf.ThreadPoolExecutor(max_workers=5) as ex:
    futs = [ex.submit(test, k, e) for k, e in models.items()]
    results = [f.result() for f in futs]

print()
ok = fail = 0
for key, status, detail in results:
    mark = "✅" if status == "OK" else "❌"
    if status == "OK":
        ok += 1
    else:
        fail += 1
    print(f"{mark} {key:26s} {detail}")
print(f"\n{ok} OK / {fail} FAIL")
sys.exit(0 if fail == 0 else 1)
