import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

import litellm
from litellm import completion, acompletion

from .base import BaseLLMClient

logger = logging.getLogger(__name__)

env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

if os.getenv("LITELLM_DEBUG", "false").lower() == "true":
    litellm.set_verbose = True
else:
    litellm.set_verbose = False
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class LiteLLMClient(BaseLLMClient):
    """LLM client via liteLLM (Gemini, OpenAI, etc.)."""

    def __init__(self, model: str, timeout: float = 600.0, gen_kwargs: Optional[Dict] = None):
        super().__init__(model, timeout, gen_kwargs)

    def _prepare_params(self, messages: List[Dict]) -> Dict:
        params: Dict = {
            "model": self.model,
            "messages": messages,
            "timeout": self.timeout,
        }
        if "temperature" in self.gen_kwargs:
            params["temperature"] = self.gen_kwargs["temperature"]
        if "max_tokens" in self.gen_kwargs:
            params["max_tokens"] = self.gen_kwargs["max_tokens"]
        if "top_p" in self.gen_kwargs:
            params["top_p"] = self.gen_kwargs["top_p"]
        if "top_k" in self.gen_kwargs:
            params["top_k"] = self.gen_kwargs["top_k"]
        if "reasoning_effort" in self.gen_kwargs:
            params["reasoning_effort"] = self.gen_kwargs["reasoning_effort"]
        return params

    def _extract_response(self, response, latency_ms: float) -> Tuple[str, Dict]:
        return (
            response.choices[0].message.content,
            {
                "latency_ms": latency_ms,
                "tokens": getattr(response.usage, "total_tokens", 0),
            },
        )

    def _handle_error(self, e: Exception, latency_ms: float, is_async: bool = False):
        prefix = "LLM async generation" if is_async else "LLM generation"
        error_msg = f"{prefix} failed after {latency_ms:.0f}ms: {str(e)}"
        if "api_key" in str(e).lower() or "authentication" in str(e).lower():
            error_msg += "\nHint: Check if your API key is set correctly in .env file"
        logger.error(error_msg)
        raise Exception(error_msg)

    def generate(self, messages: List[Dict]) -> Tuple[str, Dict]:
        start = time.time()
        try:
            params = self._prepare_params(messages)
            response = completion(**params)
            return self._extract_response(response, (time.time() - start) * 1000)
        except Exception as e:
            self._handle_error(e, (time.time() - start) * 1000, is_async=False)

    async def _async_generate(
        self, messages: List[Dict], max_retries: int = 3
    ) -> Tuple[str, Dict]:
        start = time.time()
        for attempt in range(max_retries):
            try:
                params = self._prepare_params(messages)
                response = await acompletion(**params)
                return self._extract_response(response, (time.time() - start) * 1000)
            except Exception as e:
                latency_ms = (time.time() - start) * 1000
                if attempt == max_retries - 1:
                    self._handle_error(e, latency_ms, is_async=True)
                wait_time = min(2 ** attempt, 30)
                logger.warning(
                    f"LLM async generation failed (attempt {attempt + 1}/{max_retries}): "
                    f"{str(e)[:200]}. Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)

    def __repr__(self) -> str:
        return f"LiteLLMClient(model={self.model}, gen_kwargs={self.gen_kwargs})"
