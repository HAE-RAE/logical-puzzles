import logging
import os
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from dotenv import load_dotenv
import litellm
from litellm import completion, acompletion

logger = logging.getLogger(__name__)

env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Loaded environment variables from {env_path}")
else:
    load_dotenv()
    logger.info("Loaded environment variables from default locations")

if os.getenv("LITELLM_DEBUG", "false").lower() == "true":
    litellm.set_verbose = True
else:
    litellm.set_verbose = False
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class UnifiedLLMClient:
    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        reasoning_effort: Optional[str] = None
    ):
        """
        Args:
            model: LiteLLM model identifier
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens
            timeout: Request timeout (seconds)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            reasoning_effort: Reasoning effort level (for Gemini models, low/medium/high)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.top_p = top_p
        self.top_k = top_k
        self.reasoning_effort = reasoning_effort
    
    def _prepare_params(self, messages: List[Dict], **kwargs) -> Dict:
        """Prepare common parameters"""
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "timeout": self.timeout,
        }
        
        # Add top_p parameter
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            params["top_p"] = top_p
        
        # Add top_k parameter
        top_k = kwargs.get("top_k", self.top_k)
        if top_k is not None:
            params["top_k"] = top_k
        
        # Add reasoning_effort parameter (for Gemini models)
        reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        if reasoning_effort is not None:
            # LiteLLM can pass model-specific parameters through extra_body
            if "extra_body" not in params:
                params["extra_body"] = {}
            params["extra_body"]["reasoning_effort"] = reasoning_effort
        
        return params
    
    def _extract_response(self, response, latency_ms: float) -> Tuple[str, Dict]:
        """Common response extraction logic"""
        return (
            response.choices[0].message.content,
            {
                "latency_ms": latency_ms,
                "tokens": getattr(response.usage, "total_tokens", 0)
            }
        )
    
    def _handle_error(self, e: Exception, latency_ms: float, is_async: bool = False):
        """Common error handling"""
        prefix = "LLM async generation" if is_async else "LLM generation"
        error_msg = f"{prefix} failed after {latency_ms:.0f}ms: {str(e)}"
        if "api_key" in str(e).lower() or "authentication" in str(e).lower():
            error_msg += "\nHint: Check if your API key is set correctly in .env file"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def generate(
        self,
        messages: List[Dict],
        **kwargs
    ) -> Tuple[str, Dict]:
        """
        Single generation
        
        Args:
            messages: Message list [{"role": "user", "content": "..."}]
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            (response_text, usage_dict) tuple
            usage_dict includes latency_ms, tokens, etc.
        """
        start = time.time()
        try:
            params = self._prepare_params(messages, **kwargs)
            response = completion(**params)
            return self._extract_response(response, (time.time() - start) * 1000)
        except Exception as e:
            self._handle_error(e, (time.time() - start) * 1000, is_async=False)
    
    async def _async_generate(
        self,
        messages: List[Dict],
        max_retries: int = 3,
        **kwargs
    ) -> Tuple[str, Dict]:
        """
        Async single generation with retry logic (internal use)
        
        Args:
            messages: Message list [{"role": "user", "content": "..."}]
            max_retries: Maximum number of retry attempts (default: 3)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            (response_text, usage_dict) tuple
            usage_dict includes latency_ms, tokens, etc.
        """
        start = time.time()
        for attempt in range(max_retries):
            try:
                params = self._prepare_params(messages, **kwargs)
                response = await acompletion(**params)
                return self._extract_response(response, (time.time() - start) * 1000)
            except Exception as e:
                latency_ms = (time.time() - start) * 1000
                if attempt == max_retries - 1:
                    self._handle_error(e, latency_ms, is_async=True)
                
                wait_time = min(2 ** attempt, 30)
                logger.warning(
                    f"LLM async generation failed (attempt {attempt + 1}/{max_retries}): {str(e)[:200]}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
    
    async def async_batch_generate(
        self,
        messages_list: List[List[Dict]],
        max_concurrent: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Tuple[str, Dict]]:
        """
        Async batch generation (concurrent execution)
        
        Args:
            messages_list: List of message lists
            max_concurrent: Maximum number of concurrent executions
            progress_callback: Progress callback function (completed, total) -> None
            
        Returns:
            List of (response_text, usage_dict) tuples
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        total = len(messages_list)
        completed = 0
        completed_lock = asyncio.Lock()
        
        async def generate_with_semaphore(messages, index):
            nonlocal completed
            try:
                async with semaphore:
                    result = await self._async_generate(messages)
                    async with completed_lock:
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total)
                    return result
            except Exception as e:
                async with completed_lock:
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
                raise
        
        # Execute all requests concurrently
        tasks = [generate_with_semaphore(messages, i) for i, messages in enumerate(messages_list)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Exception handling
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Error in async request {i+1}: {result}")
                processed_results.append(("", {"error": str(result), "tokens": 0}))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def __repr__(self) -> str:
        return f"UnifiedLLMClient(model={self.model}, temperature={self.temperature}, max_tokens={self.max_tokens}, top_p={self.top_p}, top_k={self.top_k}, reasoning_effort={self.reasoning_effort})"
