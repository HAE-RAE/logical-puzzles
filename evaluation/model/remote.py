import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests as http_requests

from .base import BaseLLMClient

logger = logging.getLogger(__name__)


def _coerce_stream_flag(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("1", "true", "on", "yes")
    return bool(value)


def _openai_sse_parse_line(line: str) -> Optional[Dict]:
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    payload = line[5:].lstrip()
    if payload == "[DONE]":
        return {"__done__": True}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("Skipping malformed SSE JSON line: %s...", line[:200])
        return None


def _openai_sse_accumulate_from_lines(
    lines: Any,
) -> Tuple[str, str, Dict[str, Any]]:
    """OpenAI-style chat completion SSE: accumulate content / reasoning, capture usage."""
    content_parts: List[str] = []
    thinking_parts: List[str] = []
    usage: Dict[str, Any] = {}

    for line in lines:
        if line is None:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.rstrip("\r\n")
        obj = _openai_sse_parse_line(line)
        if obj is None:
            continue
        if obj.get("__done__"):
            break

        u = obj.get("usage")
        if isinstance(u, dict):
            usage = u

        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            c = delta.get("content")
            if c:
                content_parts.append(c)
            r = delta.get("reasoning_content") or delta.get("reasoning")
            if r:
                thinking_parts.append(r)

    content = "".join(content_parts)
    thinking = "".join(thinking_parts)
    return content, thinking, usage


async def _openai_sse_accumulate_from_async_lines(
    lines: Any,
) -> Tuple[str, str, Dict[str, Any]]:
    """동기 `accumulate`와 동일; aiohttp `readline` 등 async iterator용."""
    content_parts: List[str] = []
    thinking_parts: List[str] = []
    usage: Dict[str, Any] = {}

    async for line in lines:
        if line is None:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.rstrip("\r\n")
        obj = _openai_sse_parse_line(line)
        if obj is None:
            continue
        if obj.get("__done__"):
            break

        u = obj.get("usage")
        if isinstance(u, dict):
            usage = u

        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            c = delta.get("content")
            if c:
                content_parts.append(c)
            r = delta.get("reasoning_content") or delta.get("reasoning")
            if r:
                thinking_parts.append(r)

    content = "".join(content_parts)
    thinking = "".join(thinking_parts)
    return content, thinking, usage


# 스크립트 등 외부에서 스트림 파싱 재사용 시 사용 (예: scripts/calltest_qwen.py)
accumulate_openai_chat_sse_lines = _openai_sse_accumulate_from_lines


class RemoteLLMClient(BaseLLMClient):
    """LLM client via remote vLLM server (OpenAI-compatible API)."""

    _OPENAI_PARAMS = {
        "temperature",
        "max_tokens",
        "top_p",
        "presence_penalty",
        "top_k",
        "min_p",
        "repetition_penalty",
    }

    def __init__(
        self,
        model: str,
        timeout: float = 600.0,
        remote_url: str = "",
        gen_kwargs: Optional[Dict] = None,
    ):
        super().__init__(model, timeout, gen_kwargs)
        self.remote_url = remote_url.rstrip("/")
        self._endpoint = f"{self.remote_url}/v1/chat/completions"
        logger.info(f"Remote server mode (OpenAI compatible): {self._endpoint}")

    def _build_payload(self, messages: List[Dict]) -> Dict:
        use_stream = _coerce_stream_flag(self.gen_kwargs.get("stream"), default=True)
        payload: Dict = {
            "model": self.model,
            "messages": messages,
            "stream": use_stream,
        }
        for k in self._OPENAI_PARAMS:
            if k in self.gen_kwargs:
                payload[k] = self.gen_kwargs[k]
        if "enable_thinking" in self.gen_kwargs:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.gen_kwargs["enable_thinking"],
            }
        return payload

    @staticmethod
    def _parse_response(data: Dict, latency_ms: float) -> Tuple[str, Dict]:
        choice = data["choices"][0]["message"]
        content = choice.get("content", "") or ""
        # vLLM reasoning parser는 버전에 따라 `reasoning_content` 또는 `reasoning` 필드로 반환.
        # 둘 다 체크하여 누락 방지.
        thinking = (
            choice.get("reasoning_content")
            or choice.get("reasoning")
            or ""
        )

        if not thinking and "<think>" in content:
            m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if m:
                thinking = m.group(1).strip()
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        tokens = data.get("usage", {}).get("total_tokens", 0)
        return (
            content,
            {
                "latency_ms": latency_ms,
                "tokens": tokens,
                "thinking_content": thinking,
            },
        )

    @staticmethod
    def _finalize_stream_texts(content: str, thinking: str) -> Tuple[str, str]:
        if not thinking and "<think>" in content:
            m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if m:
                thinking = m.group(1).strip()
                content = re.sub(
                    r"<think>.*?</think>",
                    "",
                    content,
                    flags=re.DOTALL,
                ).strip()
        return content, thinking

    def _parse_streaming_requests(
        self, resp: http_requests.Response, start: float
    ) -> Tuple[str, Dict]:
        resp.raise_for_status()
        content, thinking, usage = _openai_sse_accumulate_from_lines(
            resp.iter_lines(decode_unicode=True)
        )
        content, thinking = self._finalize_stream_texts(content, thinking)
        tokens = 0
        if isinstance(usage, dict):
            tokens = usage.get("total_tokens", 0) or 0
        latency_ms = (time.time() - start) * 1000
        return (
            content,
            {
                "latency_ms": latency_ms,
                "tokens": tokens,
                "thinking_content": thinking,
            },
        )

    def generate(self, messages: List[Dict]) -> Tuple[str, Dict]:
        payload = self._build_payload(messages)
        start = time.time()
        try:
            use_stream = bool(payload.get("stream"))
            if use_stream:
                with http_requests.post(
                    self._endpoint,
                    json=payload,
                    timeout=self.timeout,
                    stream=True,
                ) as resp:
                    return self._parse_streaming_requests(resp, start)
            resp = http_requests.post(self._endpoint, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return self._parse_response(resp.json(), (time.time() - start) * 1000)
        except http_requests.exceptions.Timeout:
            error_msg = f"Remote server timeout ({self.timeout}s): {self._endpoint}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except http_requests.exceptions.ConnectionError:
            error_msg = f"Remote server connection failed: {self._endpoint} — check if the URL is valid."
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            error_msg = f"Remote request failed ({latency_ms:.0f}ms): {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    async def _async_generate(
        self, messages: List[Dict], max_retries: int = 3
    ) -> Tuple[str, Dict]:
        import aiohttp

        payload = self._build_payload(messages)
        start = time.time()

        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    use_stream = bool(payload.get("stream"))
                    async with session.post(self._endpoint, json=payload) as resp:
                        if use_stream:

                            async def _line_iter():
                                while True:
                                    raw = await resp.content.readline()
                                    if not raw:
                                        break
                                    yield raw.decode("utf-8", errors="replace")

                            resp.raise_for_status()
                            content, thinking, usage = await _openai_sse_accumulate_from_async_lines(
                                _line_iter()
                            )
                            content, thinking = self._finalize_stream_texts(content, thinking)
                            tokens = 0
                            if isinstance(usage, dict):
                                tokens = usage.get("total_tokens", 0) or 0
                            return (
                                content,
                                {
                                    "latency_ms": (time.time() - start) * 1000,
                                    "tokens": tokens,
                                    "thinking_content": thinking,
                                },
                            )
                        resp.raise_for_status()
                        data = await resp.json()
                        return self._parse_response(data, (time.time() - start) * 1000)
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    error_msg = f"Remote server timeout ({self.timeout}s): {self._endpoint}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
            except aiohttp.ClientConnectorError:
                if attempt == max_retries - 1:
                    error_msg = f"Remote server connection failed: {self._endpoint} — check if the URL is valid."
                    logger.error(error_msg)
                    raise Exception(error_msg)
            except Exception as e:
                if attempt == max_retries - 1:
                    latency_ms = (time.time() - start) * 1000
                    error_msg = f"Remote async request failed ({latency_ms:.0f}ms): {e}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

            wait_time = min(2 ** attempt, 30)
            logger.warning(
                f"Remote async request failed (attempt {attempt + 1}/{max_retries}): "
                f"retrying in {wait_time}s..."
            )
            await asyncio.sleep(wait_time)

    def __repr__(self) -> str:
        return f"RemoteLLMClient(remote_url={self.remote_url}, gen_kwargs={self.gen_kwargs})"
