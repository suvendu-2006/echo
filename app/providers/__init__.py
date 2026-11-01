"""Echo provider seam package (Phase 1) — all the brains behind one interface."""

from app.providers.base import (
    Capability,
    Message,
    MockProvider,
    ProviderAdapter,
    ProviderConfigError,
    ProviderError,
    ProviderUnavailable,
    Request,
    StreamEvent,
    ToolCall,
    ToolSpec,
    Usage,
)
from app.providers.failover import ProviderChain
from app.providers.ollama import (
    DEFAULT_BASE_URL,
    list_ollama_models,
    ollama_compat_provider,
    ollama_running,
)
from app.providers.openai_compat import OpenAICompatProvider
from app.providers.probe import TextProtocolAdapter, probe_capabilities

__all__ = [
    # base contract
    "Capability",
    "Message",
    "MockProvider",
    "ProviderAdapter",
    "ProviderConfigError",
    "ProviderError",
    "ProviderUnavailable",
    "Request",
    "StreamEvent",
    "ToolCall",
    "ToolSpec",
    "Usage",
    # adapters & services (PLAN 1.2–1.5)
    "DEFAULT_BASE_URL",
    "OpenAICompatProvider",
    "ProviderChain",
    "TextProtocolAdapter",
    "list_ollama_models",
    "ollama_compat_provider",
    "ollama_running",
    "probe_capabilities",
]
