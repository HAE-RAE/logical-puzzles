import json
import requests
import time

URL = "https://tremendously-bureaucratic-alda.ngrok-free.dev"

data = {
    "model": "Qwen/Qwen3-0.6B",
    # "model": "Qwen/Qwen3-1.7B",

    "messages": [
        {"role": "user", "content": "세종대왕 맥북 던짐 사건 알려줘"},
    ],
    "temperature": 0.6,
    "max_tokens": 16384,
    "top_p": 0.95,
    "top_k": 20,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": True},
}

print(f"[URL] {URL}/v1/chat/completions")
print(f"[Model] {data['model']}")
print(f"[Thinking] {data['chat_template_kwargs']['enable_thinking']}")
print(f"[Prompt] {data['messages'][0]['content']}")
print("-" * 50)

start = time.time()
try:
    resp = requests.post(f"{URL}/v1/chat/completions", json=data, timeout=600)
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
    print(f"[Error] Timeout: {URL}")
except requests.exceptions.ConnectionError:
    print(f"[Error] Connection failed: {URL} — Check Colab server/ngrok URL.")
except Exception as e:
    print(f"[Error] {e}")
