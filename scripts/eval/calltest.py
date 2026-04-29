"""End-to-end call test with the Korean ferryman puzzle.

Two backends:
  --backend gemini : LiteLLM → gemini/gemini-3-flash-preview (uses GEMINI_API_KEY)
  --backend qwen   : Plain HTTP POST to a vLLM /v1/chat/completions server

The system prompt + user content are identical across backends so you can
compare reasoning/quality with one canonical hard test (target answer 68시간 24분).

Examples:
  python scripts/calltest.py --backend gemini
  python scripts/calltest.py --backend qwen --url https://...ngrok-free.dev --model Qwen/Qwen3-1.7B
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _lib import PROJECT_ROOT, ensure_dotenv


SYSTEM_PROMPT = """당신은 뱃사공 운항 문제를 정확히 해결하는 전문가입니다.

### 규칙
1. 주어진 운항 규정을 모두 고려하여 단계별로 분석하세요.
2. 속도 제한, 의무 휴식, 화물 규정을 모두 적용하여 계산하세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 최종 답 형식
마지막에 $\\boxed{N시간 M분}$ 형식으로 정답을 표시하세요.
"""

USER_CONTENT = """
수로 안내원 오씨는 이 강에서만 9년을 일한 베테랑으로, 총 길이 206km의 상류 지역에 물품을 운송하는 임무를 맡았다. 그는 오전 8시에 12kg짜리 의료품 키트 10개와 15kg짜리 식수통 5개를 싣고 출발했다. 배는 정수(靜水)에서 시속 22km로 이동 가능하다.
이 강에는 시속 7km의 물살이 있는데, B구역에서는 순류(실효 속력 = 배 속력 + 유속), A구역에서는 역류(실효 속력 = 배 속력 - 유속)이다.
첫 23km는 A구역(제한속도 17km/h), 이후 B구역(제한속도 13km/h)이다. 제한 속도는 유속 반영 후 실효 속력에 적용된다.
안전 중량 기준(400kg) 초과 시, A구역 제한속도는 21% 감소하고 B구역 제한속도는 33% 감소한다.
오전 9시부터 오후 2시까지는 혼잡시간대로, 모든 구역의 제한속도가 추가로 41% 감소한다. 이 감속은 화물 규정 적용 후 제한속도에 추가 적용된다.
60km 지점의 중간 기착지에서 식수통 3개를 하역한다. 하역 후 잔여 화물 무게에 따라 화물 규정이 재적용된다.
연속 101분 이상 운항할 수 없으며, 휴게 지점(매 12km)에서만 쉴 수 있다. 기본 휴식 시간은 44분이다. 단, 누적 피로 규정에 따라 매 휴식마다 20분씩 추가된다(첫째 44분, 둘째 64분, 셋째 84분, …).
이 모든 조건을 준수하여 최종 목적지까지 도착했을 때, 의무 휴식을 포함한 총 소요 시간은 몇 시간 몇 분입니까? (분은 소숫점 첫째 자리에서 반올림)\
"""

# hard target answer: 68시간 24분


def call_gemini(args):
    ensure_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is not set. Please check the .env file.")

    import litellm
    from litellm import completion

    litellm.set_verbose = True

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_CONTENT},
    ]
    print(f"[Messages] {len(messages)} messages loaded.")
    print("-" * 50)

    start = time.time()
    try:
        response = completion(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            reasoning_effort="high",
        )
        latency = time.time() - start
        print(f"[Response Time] {latency:.2f}s")

        msg = response.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None)

        print(f"\n[Reasoning]\n{reasoning if reasoning else '(None — may be inlined or unsupported by current LiteLLM version)'}")
        print(f"\n[Answer]\n{content}")

        if hasattr(response, "usage"):
            print(f"\n[Usage] {response.usage}")
    except Exception as e:
        print(f"[Error] {e}")


def call_qwen(args):
    import requests

    if not args.url:
        raise SystemExit("--url is required for --backend qwen")

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_CONTENT},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
    }

    endpoint = f"{args.url.rstrip('/')}/v1/chat/completions"
    print(f"[URL] {endpoint}")
    print(f"[Model] {payload['model']}")
    print(f"[Thinking] {payload['chat_template_kwargs']['enable_thinking']}")
    print(f"[Prompt] {payload['messages']}")
    print("-" * 50)

    start = time.time()
    try:
        resp = requests.post(endpoint, json=payload, timeout=args.timeout)
        latency = time.time() - start
        print(f"[HTTP Status] {resp.status_code}")
        print(f"[Response Time] {latency:.2f}s")

        try:
            result = resp.json()
        except ValueError:
            print("[Body (not JSON)]")
            print(resp.text[:4000])
            raise

        if resp.status_code != 200 or "choices" not in result:
            print("[Body]")
            print(json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, dict) else result)
            raise SystemExit(1)

        choice = result["choices"][0]["message"]
        thinking = (choice.get("reasoning_content") or choice.get("reasoning") or "").strip()
        content = (choice.get("content") or "").strip()

        if thinking:
            print(f"\n[Reasoning]\n{thinking}")
        print(f"\n[Answer]\n{content}")
        if "usage" in result:
            print(f"\n[Usage] {result['usage']}")
    except requests.exceptions.Timeout:
        print(f"[Error] Timeout: {endpoint}")
    except requests.exceptions.ConnectionError:
        print(f"[Error] Connection failed: {endpoint} — check server/ngrok URL.")
    except Exception as e:
        print(f"[Error] {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["gemini", "qwen"], required=True)
    parser.add_argument("--model", default=None,
                        help="Model id. Default: gemini/gemini-3-flash-preview (gemini), Qwen/Qwen3-0.6B (qwen)")
    parser.add_argument("--url", default=None, help="vLLM server base URL (qwen only)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Default: 1.0 (gemini), 0.6 (qwen)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Default: 65539 (gemini), 16384 (qwen)")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None,
                        help="Default: 64 (gemini), 20 (qwen)")
    parser.add_argument("--timeout", type=int, default=600, help="HTTP timeout (qwen only)")
    parser.add_argument("--enable-thinking", action="store_true", default=True,
                        help="vLLM chat_template enable_thinking flag (qwen only)")
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    args = parser.parse_args()

    # Backend-specific defaults
    if args.backend == "gemini":
        args.model = args.model or "gemini/gemini-3-flash-preview"
        args.temperature = 1.0 if args.temperature is None else args.temperature
        args.max_tokens = 65539 if args.max_tokens is None else args.max_tokens
        args.top_k = 64 if args.top_k is None else args.top_k
        call_gemini(args)
    else:
        args.model = args.model or "Qwen/Qwen3-0.6B"
        args.temperature = 0.6 if args.temperature is None else args.temperature
        args.max_tokens = 16384 if args.max_tokens is None else args.max_tokens
        args.top_k = 20 if args.top_k is None else args.top_k
        call_qwen(args)


if __name__ == "__main__":
    main()
