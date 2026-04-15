import asyncio
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import requests as http_requests

from .base import BaseLLMClient

logger = logging.getLogger(__name__)


class RemoteLLMClient(BaseLLMClient):
    """LLM client via remote vLLM server (OpenAI-compatible API)."""

    _OPENAI_PARAMS = {"temperature", "max_tokens", "top_p", "presence_penalty", "top_k", "min_p"}

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
        payload: Dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
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

    def generate(self, messages: List[Dict]) -> Tuple[str, Dict]:
        payload = self._build_payload(messages)
        start = time.time()
        try:
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
                    async with session.post(self._endpoint, json=payload) as resp:
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
