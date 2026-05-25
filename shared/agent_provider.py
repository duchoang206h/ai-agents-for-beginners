"""Provider selection helpers for Microsoft Agent Framework samples.

The lesson notebooks use a provider-like object with an async ``create_agent``
method. This module keeps that shape while allowing the samples to run against
Azure Foundry, OpenAI, GitHub Models, MiniMax, or another OpenAI-compatible API.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1"
DEFAULT_GITHUB_ENDPOINT = "https://models.inference.ai.azure.com"
DEFAULT_MINIMAX_ENDPOINT = "https://api.minimax.io/v1"
DEFAULT_MINIMAX_MODEL = "MiniMax-M2.7"
_DOTENV_LOADED = False


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved provider configuration for display and troubleshooting."""

    name: str
    model_id: str | None = None
    base_url: str | None = None


class ChatClientProvider:
    """Adapt an Agent Framework chat client to the course provider interface."""

    def __init__(self, chat_client: Any, config: ProviderConfig) -> None:
        self.chat_client = chat_client
        self.config = config

    async def create_agent(self, **kwargs: Any) -> Any:
        """Create an agent from the configured chat client."""
        create_agent = getattr(self.chat_client, "create_agent", None)
        if create_agent is not None:
            agent = create_agent(**kwargs)
        else:
            as_agent = getattr(self.chat_client, "as_agent", None)
            if as_agent is not None:
                agent = as_agent(**kwargs)
            else:
                from agent_framework import Agent

                agent = Agent(client=self.chat_client, **kwargs)

        if inspect.isawaitable(agent):
            return await agent
        return agent


class FoundryChatClientProvider(ChatClientProvider):
    """Azure Foundry provider adapter for current Agent Framework releases."""


def _normalize_provider(provider: str | None) -> str | None:
    if not provider:
        return None

    provider = provider.strip().lower().replace("_", "-")
    aliases = {
        "azure": "azure",
        "azure-ai": "azure",
        "azure-foundry": "azure",
        "foundry": "azure",
        "openai": "openai",
        "github": "github",
        "github-models": "github",
        "minimax": "minimax",
        "custom": "openai-compatible",
        "qwen": "openai-compatible",
        "compatible": "openai-compatible",
        "openai-compatible": "openai-compatible",
    }
    if provider not in aliases:
        raise ValueError(
            "Unsupported AI_AGENT_PROVIDER value "
            f"{provider!r}. Use azure, openai, github, minimax, or openai-compatible."
        )
    return aliases[provider]


def _load_dotenv() -> None:
    global _DOTENV_LOADED

    if _DOTENV_LOADED:
        return

    _DOTENV_LOADED = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    repo_env = Path(__file__).resolve().parent.parent / ".env"
    if repo_env.exists():
        load_dotenv(repo_env, override=False)
    else:
        load_dotenv(override=False)


def _env(name: str, default: str | None = None) -> str | None:
    _load_dotenv()
    value = os.getenv(name)
    if value is None:
        return default

    value = value.strip()
    if value in {"", "...", "<...>", "your-api-key", "your_github_personal_access_token"}:
        return default
    if value.startswith("your-") or value.startswith("<your-"):
        return default
    return value


def _env_url(name: str, default: str | None = None) -> str | None:
    value = _env(name, default)
    if value is None:
        os.environ.pop(name, None)
        return None

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            f"{name} must start with http:// or https://. Current value: {value!r}")
    return value


def detect_provider() -> str:
    """Resolve the provider from AI_AGENT_PROVIDER or available credentials."""
    requested = _normalize_provider(_env("AI_AGENT_PROVIDER"))
    if requested:
        return requested

    if _env_url("OPENAI_BASE_URL"):
        return "openai-compatible"
    if _env("OPENAI_API_KEY"):
        return "openai"
    if _env("MINIMAX_API_KEY"):
        return "minimax"
    if _env("GITHUB_TOKEN"):
        return "github"
    return "azure"


def _create_openai_chat_client(
    *,
    api_key: str | None,
    model_id: str,
    base_url: str | None = None,
) -> Any:
    from agent_framework.openai import OpenAIChatClient

    if not base_url:
        os.environ.pop("OPENAI_BASE_URL", None)

    parameters = inspect.signature(OpenAIChatClient.__init__).parameters
    model_parameter = "model" if "model" in parameters else "model_id"

    kwargs: dict[str, Any] = {model_parameter: model_id}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAIChatClient(**kwargs)


def _openai_model_id(default: str = DEFAULT_OPENAI_MODEL) -> str:
    return (
        _env("OPENAI_CHAT_MODEL")
        or _env("OPENAI_MODEL")
        or _env("OPENAI_MODEL_ID")
        or _env("OPENAI_CHAT_MODEL_ID")
        or default
    )


def _create_openai_provider() -> ChatClientProvider:
    model_id = _openai_model_id()
    client = _create_openai_chat_client(
        api_key=_env("OPENAI_API_KEY"),
        model_id=model_id,
        base_url=DEFAULT_OPENAI_ENDPOINT,
    )
    return ChatClientProvider(
        client,
        ProviderConfig(name="OpenAI", model_id=model_id,
                       base_url=DEFAULT_OPENAI_ENDPOINT),
    )


def _create_openai_compatible_provider() -> ChatClientProvider:
    base_url = _env_url("OPENAI_BASE_URL")
    if not base_url:
        raise ValueError(
            "OPENAI_BASE_URL is required when AI_AGENT_PROVIDER=openai-compatible.")

    model_id = _openai_model_id()
    api_key = _env("OPENAI_API_KEY") or _env(
        "OPENAI_COMPATIBLE_API_KEY") or "not-needed"
    client = _create_openai_chat_client(
        api_key=api_key, model_id=model_id, base_url=base_url)
    return ChatClientProvider(
        client,
        ProviderConfig(name="OpenAI-compatible",
                       model_id=model_id, base_url=base_url),
    )


def _create_github_provider() -> ChatClientProvider:
    token = _env("GITHUB_TOKEN")
    if not token:
        raise ValueError(
            "GITHUB_TOKEN is required when AI_AGENT_PROVIDER=github.")

    endpoint = _env_url("GITHUB_ENDPOINT", DEFAULT_GITHUB_ENDPOINT)
    model_id = _env("GITHUB_MODEL_ID", DEFAULT_OPENAI_MODEL)
    client = _create_openai_chat_client(
        api_key=token, model_id=model_id, base_url=endpoint)
    return ChatClientProvider(client, ProviderConfig(name="GitHub Models", model_id=model_id, base_url=endpoint))


def _create_minimax_provider() -> ChatClientProvider:
    api_key = _env("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError(
            "MINIMAX_API_KEY is required when AI_AGENT_PROVIDER=minimax.")

    endpoint = _env_url("MINIMAX_BASE_URL", DEFAULT_MINIMAX_ENDPOINT)
    model_id = _env("MINIMAX_MODEL_ID", DEFAULT_MINIMAX_MODEL)
    client = _create_openai_chat_client(
        api_key=api_key, model_id=model_id, base_url=endpoint)
    return ChatClientProvider(client, ProviderConfig(name="MiniMax", model_id=model_id, base_url=endpoint))


def _create_legacy_azure_provider() -> Any:
    from agent_framework.azure import AzureAIProjectAgentProvider
    from azure.identity import AzureCliCredential

    provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())
    try:
        provider.config = ProviderConfig(
            name="Azure AI Foundry",
            model_id=_env("AZURE_AI_MODEL_DEPLOYMENT_NAME"),
            base_url=_env("AZURE_AI_PROJECT_ENDPOINT"),
        )
    except AttributeError:
        pass
    return provider


def _create_foundry_provider() -> FoundryChatClientProvider:
    from agent_framework.foundry import FoundryChatClient
    from azure.identity import AzureCliCredential

    endpoint = _env("AZURE_AI_PROJECT_ENDPOINT") or _env(
        "FOUNDRY_PROJECT_ENDPOINT")
    if not endpoint:
        raise ValueError(
            "AZURE_AI_PROJECT_ENDPOINT or FOUNDRY_PROJECT_ENDPOINT is required for Azure Foundry.")

    model_id = _env("AZURE_AI_MODEL_DEPLOYMENT_NAME") or _env("FOUNDRY_MODEL")
    if not model_id:
        raise ValueError(
            "AZURE_AI_MODEL_DEPLOYMENT_NAME or FOUNDRY_MODEL is required for Azure Foundry.")

    client = FoundryChatClient(
        project_endpoint=endpoint, model=model_id, credential=AzureCliCredential())
    return FoundryChatClientProvider(
        client,
        ProviderConfig(name="Azure AI Foundry",
                       model_id=model_id, base_url=endpoint),
    )


def _create_azure_provider() -> Any:
    try:
        return _create_legacy_azure_provider()
    except ImportError:
        return _create_foundry_provider()


def create_provider(provider: str | None = None) -> Any:
    """Create the configured Agent Framework provider.

    Provider order when ``AI_AGENT_PROVIDER`` is not set:
    OpenAI-compatible endpoint, OpenAI, MiniMax, GitHub Models, then Azure.
    """
    selected = _normalize_provider(provider) or detect_provider()

    if selected == "openai":
        return _create_openai_provider()
    if selected == "openai-compatible":
        return _create_openai_compatible_provider()
    if selected == "github":
        return _create_github_provider()
    if selected == "minimax":
        return _create_minimax_provider()
    if selected == "azure":
        return _create_azure_provider()

    raise AssertionError(f"Unhandled provider: {selected}")


def describe_provider(provider: Any) -> str:
    """Return a concise provider description for notebook output."""
    config = getattr(provider, "config", None)
    if not config:
        return provider.__class__.__name__

    parts = [config.name]
    if config.model_id:
        parts.append(f"model={config.model_id}")
    if config.base_url:
        parts.append(f"endpoint={config.base_url}")
    return " | ".join(parts)
