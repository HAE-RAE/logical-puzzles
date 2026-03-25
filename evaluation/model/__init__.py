from .base import BaseLLMClient
from .litellm import LiteLLMClient
from .remote import RemoteLLMClient


def create_client(
    model: str,
    timeout: float = 600.0,
    remote_url: str | None = None,
    gen_kwargs: dict | None = None,
) -> BaseLLMClient:
    """Create an appropriate LLM client based on model_router."""
    if remote_url:
        return RemoteLLMClient(model, timeout, remote_url, gen_kwargs)
    return LiteLLMClient(model, timeout, gen_kwargs)


__all__ = [
    "BaseLLMClient",
    "LiteLLMClient",
    "RemoteLLMClient",
    "create_client",
]
