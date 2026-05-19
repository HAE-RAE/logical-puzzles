"""Google Gemini `generateContent` + requests (최소 예시).

비교 실험:
  python calltest_gemini_requests.py --compare
    → thinkingLevel 생략( includeThoughts만 ) vs thinkingLevel=HIGH 명시
    → 토큰/텍스트가 같은지 한 화면에서 확인 (API가 생략 시 기본 HIGH로 도는지는 이 수치로 추론)
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _lib import PROJECT_ROOT, ensure_dotenv

ensure_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("GEMINI_API_KEY is not set. Please check the .env file.")

MODEL = "gemini-3-flash-preview"
url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
headers = {"Content-Type": "application/json"}
params = {"key": API_KEY}

content = """
수로 안내원 오씨는 이 강에서만 13년을 일한 베테랑으로, 총 길이 332km의 상류 지역에 물품을 운송하는 임무를 맡았다. 그는 오전 9시에 15kg짜리 의료품 키트 19개와 24kg짜리 식수통 9개를 싣고 출발했다. 배는 정수(靜水)에서 시속 23km로 이동 가능하다.
이 강에는 시속 8km의 물살이 있는데, B구역에서는 순류(실효 속력 = 배 속력 + 유속), A구역에서는 역류(실효 속력 = 배 속력 - 유속)이다.
첫 42km는 A구역(제한속도 17km/h), 이후 B구역(제한속도 16km/h)이다. 제한 속도는 유속 반영 후 실효 속력에 적용된다.
안전 중량 기준(300kg) 초과 시, A구역 제한속도는 16% 감소하고 B구역 제한속도는 32% 감소한다.
오전 10시부터 오후 3시까지는 혼잡시간대로, B구역의 제한속도가 추가로 32% 감소한다. 이 감속은 화물 규정 적용 후 제한속도에 추가 적용된다.
90km 지점의 중간 기착지에서 의료품 키트 8개를 하역한다. 하역 후 잔여 화물 무게에 따라 화물 규정이 재적용된다.
연속 110분 이상 운항할 수 없으며, 휴게 지점(매 10km)에서만 쉴 수 있다. 기본 휴식 시간은 57분이다. 단, 누적 피로 규정에 따라 매 휴식마다 37분씩 추가된다(첫째 57분, 둘째 94분, 셋째 131분, …).
이 모든 조건을 준수하여 최종 목적지까지 도착했을 때, 의무 휴식을 포함한 총 소요 시간은 몇 시간 몇 분입니까? 총 소요 시간을 먼저 분(分) 단위 정수로 계산한 후 시간과 분으로 변환하여 답하시오. (예: 총 1450분 → 24시간 10분)
"""

messages = [
    {"role": "user", "content": "안녕"},
]

BASE_GENERATION: Dict[str, Any] = {
    "temperature": 1.0,
    "maxOutputTokens": 65536,
    "topP": 0.95,
    "topK": 64,
}


def build_data(
    thinking_level: Optional[str],
    *,
    include_thoughts: bool = True,
    drop_thinking_config: bool = False,
) -> Dict[str, Any]:
    """thinking_level None + drop_thinking_config False → thinkingConfig에 level 키 없음 (주석한 것과 동일)."""
    gen = {**BASE_GENERATION}
    if drop_thinking_config:
        pass
    elif include_thoughts:
        tc: Dict[str, Any] = {"includeThoughts": True}
        if thinking_level is not None:
            tc["thinkingLevel"] = thinking_level
        gen["thinkingConfig"] = tc
    return {
        "contents": [
            {"role": "user", "parts": [{"text": messages[0]["content"]}]},
        ],
        "generationConfig": gen,
    }


def extract_result(result_json: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    parts = (result_json.get("candidates", [{}])[0].get("content") or {}).get("parts") or []
    thoughts: List[str] = []
    texts: List[str] = []
    for p in parts:
        t = p.get("text") or ""
        if p.get("thought") is True:
            thoughts.append(t)
        else:
            texts.append(t)
    return (
        "".join(thoughts),
        "".join(texts),
        result_json.get("usageMetadata") or {},
    )


def post(data: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(
        url, headers=headers, params=params, json=data, timeout=300
    )
    r.raise_for_status()
    return r.json()


def run_single(label: str, data: Dict[str, Any]) -> None:
    print("🔍 API 요청 정보:")
    print(f"[{label}]")
    print(f"URL: {url}")
    print(f"Data: {json.dumps(data, ensure_ascii=False, indent=2)}")
    print()
    response = requests.post(
        url, headers=headers, params=params, json=data, timeout=300
    )
    print(f"📡 HTTP 상태 코드: {response.status_code}")
    result_json = response.json()
    print("📄 API 응답(전체):")
    print(json.dumps(result_json, ensure_ascii=False, indent=2))
    print()
    if not isinstance(result_json, dict):
        return
    if "error" in result_json:
        print(f"❌ API 에러: {result_json['error']}")
        return
    if response.ok and "candidates" in result_json:
        th, tx, usage = extract_result(result_json)
        print(f"💭 Thought(분리): {th or '(없음)'}")
        print(f"✅ 결과(텍스트): {tx or '(없음)'}")
        print(f"📊 usageMetadata: {usage}")


def run_compare() -> None:
    # A: thinkingLevel 키 없이 includeThoughts만 (= 파일에서 "thinkingLevel" 줄 주석한 것과 동일)
    data_omit = build_data(
        None, include_thoughts=True, drop_thinking_config=False
    )
    # B: thinkingLevel=HIGH 명시
    data_high = build_data("HIGH", include_thoughts=True, drop_thinking_config=False)

    print("=" * 60)
    print("비교: (A) thinkingLevel 생략  vs  (B) thinkingLevel=HIGH")
    print("=" * 60)

    for label, d in [("A_omit", data_omit), ("B_HIGH", data_high)]:
        print(f"\n--- 요청 {label} ---\n{json.dumps(d, ensure_ascii=False, indent=2)[:1200]}...\n")

    results: List[Tuple[str, Dict[str, Any], str, str, Dict]] = []
    for label, d in [("A_omit (level 키 없음)", data_omit), ("B_명시 HIGH", data_high)]:
        try:
            j = post(d)
            th, tx, usage = extract_result(j)
            results.append((label, d, th, tx, usage))
        except Exception as e:
            print(f"❌ {label} 실패: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(e.response.text[:2000])
            return

    print("\n" + "=" * 60)
    print("요약 비교 (응답이 동일에 가깝다면, 생략 시 API 기본이 HIGH에 가깝다고 볼 수 있음 — 반대로 토큰/문구가 다르면 완전 동일하지는 않음)")
    print("=" * 60)
    for label, _d, th, tx, usage in results:
        print(f"\n■ {label}")
        print(f"  usageMetadata: {usage}")
        print(f"  answer 길이: {len(tx)} / thought 길이: {len(th)}")

    _, _, t_a, a_a, u_a = results[0]
    _, _, t_b, a_b, u_b = results[1]

    if a_a.strip() == a_b.strip() and t_a.strip() == t_b.strip():
        print("\n✅ 답·thought 문자열이 동일합니다.")
    else:
        print("\n⚠️ 답 또는 thought 내용이 다릅니다 (스토캐스틱/비결정성일 수도 있음).")
    if u_a == u_b:
        print("✅ usageMetadata 가 동일합니다.")
    else:
        print("⚠️ usageMetadata 가 다릅니다:")
        print("  A:", u_a)
        print("  B:", u_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare",
        action="store_true",
        help="thinkingLevel 생략 vs HIGH 명시 두 요청을 연속 호출해 비교",
    )
    args = parser.parse_args()

    if args.compare:
        run_compare()
    else:
        # 단일 실행: 아래 data만 손보면 됨
        data = {
            "contents": [
                {"role": "user", "parts": [{"text": messages[0]["content"]}]},
            ],
            "generationConfig": {
                "temperature": 1.0,
                "maxOutputTokens": 65536,
                "topP": 0.95,
                "topK": 64,
                "thinkingConfig": {
                    "includeThoughts": True,
                    # "thinkingLevel": "HIGH",  # ← 주석이면 level 키를 안 보냄 (compare와 A 케이스 동일)
                },
            },
        }
        run_single("single", data)
