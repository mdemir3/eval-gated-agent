"""LLM provider factory, configured only from the environment.

    LLM_PROVIDER   openai | anthropic | ollama        (required)
    LLM_MODEL      model name for that provider        (required, no default)
    LLM_TEMPERATURE / LLM_SEED                         (optional, default 0 / 42)
    OPENAI_API_KEY / ANTHROPIC_API_KEY / OLLAMA_BASE_URL

The judge suite reuses the factory with the JUDGE_ prefix (falling back to LLM_).
"""

from __future__ import annotations

import os
from typing import Any, Protocol, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

PROVIDERS = ("openai", "anthropic", "ollama")
API_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}

T = TypeVar("T", bound=BaseModel)


class LLMConfigError(RuntimeError):
    pass


def _env(prefix: str, name: str) -> str:
    value = os.getenv(f"{prefix}_{name}", "").strip()
    if not value and prefix != "LLM":
        value = os.getenv(f"LLM_{name}", "").strip()
    return value


def resolve_config(prefix: str = "LLM") -> tuple[str, str]:
    """Return (provider, model) or raise LLMConfigError with an actionable message."""
    provider = _env(prefix, "PROVIDER").lower()
    model = _env(prefix, "MODEL")
    if provider not in PROVIDERS:
        raise LLMConfigError(f"{prefix}_PROVIDER must be one of {', '.join(PROVIDERS)} (got {provider or 'nothing'}).")
    if not model:
        raise LLMConfigError(f"{prefix}_MODEL is not set. Pick a model for {provider}; see .env.example.")
    key_env = API_KEY_ENV.get(provider)
    if key_env and not os.getenv(key_env):
        raise LLMConfigError(f"{key_env} is not set but {prefix}_PROVIDER={provider}.")
    return provider, model


def has_api_key(provider: str) -> bool:
    key_env = API_KEY_ENV.get(provider)
    return bool(key_env and os.getenv(key_env))


def get_chat_model(prefix: str = "LLM") -> BaseChatModel:
    provider, model = resolve_config(prefix)
    temperature = float(_env(prefix, "TEMPERATURE") or 0)
    seed = int(_env(prefix, "SEED") or 42)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, seed=seed, timeout=120)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature, max_tokens=1024, timeout=120)
    from langchain_ollama import ChatOllama

    kwargs: dict[str, Any] = {}
    if os.getenv("OLLAMA_BASE_URL"):
        kwargs["base_url"] = os.environ["OLLAMA_BASE_URL"]
    return ChatOllama(model=model, temperature=temperature, seed=seed, num_ctx=4096, **kwargs)


class StructuredLLM(Protocol):
    """What the graph needs from an LLM: a system+user prompt in, a validated object out."""

    calls: int

    async def generate(self, schema: type[T], system: str, user: str) -> T: ...


class LangChainLLM:
    """StructuredLLM backed by any LangChain chat model via `with_structured_output`."""

    # Tool calling is the most portable structured-output path for hosted APIs;
    # Ollama uses grammar-constrained JSON-schema decoding.
    _METHOD = {"openai": "function_calling", "anthropic": "function_calling", "ollama": "json_schema"}

    def __init__(self, chat_model: BaseChatModel, provider: str, model: str) -> None:
        self.chat_model = chat_model
        self.provider = provider
        self.model = model
        self.calls = 0

    @classmethod
    def from_env(cls, prefix: str = "LLM") -> LangChainLLM:
        provider, model = resolve_config(prefix)
        return cls(get_chat_model(prefix), provider, model)

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    async def generate(self, schema: type[T], system: str, user: str) -> T:
        self.calls += 1
        runnable = self.chat_model.with_structured_output(schema, method=self._METHOD[self.provider])
        result = await runnable.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
        if not isinstance(result, schema):
            result = schema.model_validate(result)
        return result
