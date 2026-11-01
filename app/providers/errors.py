"""The failure taxonomy (Atticus's failure_why, adapted) — one human
message per failure class instead of a single generic shrug.

Provider failures arrive as exception strings and HTTP bodies from the
failover chain. explain() classifies them into a specific, actionable
sentence the client CAN act on. Secrets and URLs are never echoed back.
"""

from __future__ import annotations

DEFAULT_MESSAGE = "the provider failed - try again or switch model"


def explain(text: str) -> str:
    """Classify one provider failure string into a human message."""
    if not text:
        return DEFAULT_MESSAGE
    lowered = str(text).lower()

    if "model_not_found" in lowered or ("does not exist" in lowered and "model" in lowered):
        return "that model no longer exists on this provider - pick another in Settings > Providers"
    if "401" in lowered or "invalid_api_key" in lowered or "unauthorized" in lowered:
        return "your API key was rejected - check it in Settings > Providers"
    if "429" in lowered or "rate limit" in lowered or "quota" in lowered:
        return "rate limit reached - wait a moment or let the failover try another brain"
    if "context" in lowered and ("length" in lowered or "too long" in lowered or "token" in lowered):
        return "this conversation outgrew the model's window - start a new chat or pick a bigger brain"
    if "insufficient" in lowered and "credit" in lowered:
        return "this provider says the account is out of credits - top up or switch brain"
    if "503" in lowered or "overloaded" in lowered:
        return "the provider is overloaded - try again shortly"
    if "cannot reach" in lowered or "connection" in lowered or "unreachable" in lowered or "timed out" in lowered or "timeout" in lowered:
        return "the provider is unreachable - check your connection or pick another brain"
    if "no usable provider" in lowered:
        return "no brain configured - add a provider in Settings > Providers"
    return DEFAULT_MESSAGE
