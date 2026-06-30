"""Named provider profiles -> LangChain chat models (model-agnostic core).

A *profile* is a named, env-driven bundle of (provider kind, model, api key,
base url, extra kwargs). Any number of profiles can coexist (e.g. ``zai``,
``anthropic``, ``local``); one is the *active* profile and one optionally the
*cheap* profile (used by the model router for low-stakes subtasks).

Provider kinds map to LangChain chat-model classes. Each provider package is
imported lazily so the harness runs out-of-the-box with only ``langchain-openai``
(the OpenAI-compatible path covers Z.ai, OpenAI, OpenRouter, Together, Groq,
Fireworks, local vLLM, Ollama's OpenAI shim, ...). Missing optional packages
raise a clear, actionable error.

This replaces the original Z.ai-only ``ChatOpenAI`` wiring while preserving
backwards compatibility with the legacy ``ZAI_*`` environment variables.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

# ---------------------------------------------------------------------------
# Provider kinds -> (package, classname). Resolved lazily in build_chat_model.
# ---------------------------------------------------------------------------
# "openai" covers every OpenAI-compatible endpoint (Z.ai, OpenAI, OpenRouter,
# Together, Groq, Fireworks, local vLLM, Ollama OpenAI mode, ...).
_KIND_TABLE: dict[str, tuple[str, str]] = {
    "openai": ("langchain_openai", "ChatOpenAI"),
    "openai_compatible": ("langchain_openai", "ChatOpenAI"),
    "azure_openai": ("langchain_openai", "AzureChatOpenAI"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic"),
    "ollama": ("langchain_ollama", "ChatOllama"),
    "google": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
    "bedrock": ("langchain_aws", "ChatBedrockConverse"),
}

# Hint text shown when a provider package is missing, so users know exactly
# what to install to enable a profile.
_INSTALL_HINT: dict[str, str] = {
    "anthropic": "pip install langchain-anthropic",
    "ollama": "pip install langchain-ollama",
    "google": "pip install langchain-google-genai",
    "bedrock": "pip install langchain-aws",
    "azure_openai": "pip install langchain-openai",
}

# Default provider kind inferred from a profile *name* when MODEL_PROVIDER_<N>
# is not set. Names that are well-known OpenAI-compatible endpoints get base
# URLs too, so a one-line ``MODEL_<N>=<model>`` is enough to use them.
# NOTE: the ``zai`` default uses the *coding* endpoint (/api/coding/paas/v4/)
# required by the GLM-4.x/5.x coding models — not the generic /api/paas/v4/.
_NAME_DEFAULTS: dict[str, dict[str, str]] = {
    "zai": {"kind": "openai", "base_url": "https://api.z.ai/api/coding/paas/v4/"},
    "openai": {"kind": "openai"},
    "openrouter": {"kind": "openai", "base_url": "https://openrouter.ai/api/v1"},
    "together": {"kind": "openai", "base_url": "https://api.together.xyz/v1"},
    "groq": {"kind": "openai", "base_url": "https://api.groq.com/openai/v1"},
    "fireworks": {
        "kind": "openai",
        "base_url": "https://api.fireworks.ai/inference/v1",
    },
    "deepseek": {"kind": "openai", "base_url": "https://api.deepseek.com"},
    "mistral": {"kind": "openai", "base_url": "https://api.mistral.ai/v1"},
    "anthropic": {"kind": "anthropic"},
    "ollama": {"kind": "ollama", "base_url": "http://localhost:11434"},
    "local": {"kind": "openai", "base_url": "http://localhost:8000/v1"},
    "google": {"kind": "google"},
    "bedrock": {"kind": "bedrock"},
    "azure": {"kind": "azure_openai"},
}


@dataclass
class ProviderProfile:
    """A resolved, named provider configuration."""

    name: str
    kind: str
    model: str
    api_key: str = ""
    base_url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        """Short human label for logs / reports (e.g. ``zai:glm-5.2``)."""
        return f"{self.name}:{self.model}"

    def safe_label(self) -> str:
        """Like ``label()`` but never leaks the API key into repr/logs."""
        return self.label()


def _env(*names: str, default: str = "") -> str:
    """First non-empty env var from ``names`` (case-insensitive on the suffix)."""
    for n in names:
        v = os.environ.get(n)
        if v and v.strip():
            return v.strip()
    return default


def _resolve_api_key(profile_name: str) -> str:
    """Resolve a profile's API key from explicit value or env-var indirection."""
    direct = _env(f"API_KEY_{profile_name}", f"PROFILE_{profile_name}_API_KEY")
    if direct:
        return direct
    env_name = _env(
        f"API_KEY_ENV_{profile_name}", f"PROFILE_{profile_name}_API_KEY_ENV"
    )
    if env_name:
        return os.environ.get(env_name, "").strip()
    return ""


def load_profile(name: str) -> ProviderProfile | None:
    """Resolve a named profile from environment variables.

    Returns ``None`` when the profile has no model configured. The well-known
    ``zai`` profile is also seeded from the legacy ``ZAI_*`` variables so
    existing deployments keep working unchanged.

    Precedence (highest first) for each field:
      1. explicit profile vars (``MODEL_<name>``, ``BASE_URL_<name>``, ...)
      2. legacy ``ZAI_*`` vars (only for the ``zai`` profile)
      3. the name's built-in default (``_NAME_DEFAULTS``)
    """
    name = name.strip()
    defaults = _NAME_DEFAULTS.get(name, {"kind": "openai"})
    is_zai = name == "zai"

    kind = _env(
        f"MODEL_PROVIDER_{name}",
        f"PROFILE_{name}_PROVIDER",
        # zai legacy: ZAI_MODEL_PROVIDER isn't a thing, but keep the default.
        default=defaults["kind"],
    )
    # For each field we try: explicit profile var -> legacy ZAI_* (zai only)
    # -> built-in name default. _env() returns the first non-empty value, so
    # listing the legacy name before `default=` gives it priority over the
    # name default while still letting an explicit BASE_URL_zai override it.
    legacy_model = "ZAI_MODEL" if is_zai else None
    legacy_url = "ZAI_ENDPOINT" if is_zai else None
    model = _env(
        f"MODEL_{name}",
        f"PROFILE_{name}_MODEL",
        *([legacy_model] if legacy_model else []),
        default="glm-5.2" if is_zai else "",
    )
    base_url = _env(
        f"BASE_URL_{name}",
        f"PROFILE_{name}_BASE_URL",
        *([legacy_url] if legacy_url else []),
        default=defaults.get("base_url", ""),
    )
    # API key: explicit API_KEY_<name> (or indirection) -> legacy ZAI_API_KEY.
    api_key = _resolve_api_key(name)
    if not api_key and is_zai:
        api_key = _env("ZAI_API_KEY")

    extra_raw = _env(f"EXTRA_{name}", f"PROFILE_{name}_EXTRA")
    extra: dict[str, Any] = {}
    if extra_raw:
        try:
            extra = json.loads(extra_raw)
            if not isinstance(extra, dict):
                extra = {}
        except Exception:
            extra = {}

    if not model and not extra:
        # A profile with no model and no extras is "not configured".
        return None

    return ProviderProfile(
        name=name,
        kind=kind,
        model=model,
        api_key=api_key,
        base_url=base_url,
        extra=extra,
    )
    model = _env(f"MODEL_{name}", f"PROFILE_{name}_MODEL")
    base_url = _env(
        f"BASE_URL_{name}",
        f"PROFILE_{name}_BASE_URL",
        default=defaults.get("base_url", ""),
    )

    # Legacy ZAI_* compatibility: seed the `zai` profile from the old vars.
    if name == "zai" and not model:
        model = _env("ZAI_MODEL", default="glm-5.2")
    if name == "zai" and not base_url:
        base_url = _env("ZAI_ENDPOINT")
    api_key = _resolve_api_key(name)
    if name == "zai" and not api_key:
        api_key = _env("ZAI_API_KEY")

    extra_raw = _env(f"EXTRA_{name}", f"PROFILE_{name}_EXTRA")
    extra: dict[str, Any] = {}
    if extra_raw:
        try:
            extra = json.loads(extra_raw)
            if not isinstance(extra, dict):
                extra = {}
        except Exception:
            extra = {}

    if not model and not extra:
        # A profile with no model and no extras is "not configured".
        return None

    return ProviderProfile(
        name=name,
        kind=kind,
        model=model,
        api_key=api_key,
        base_url=base_url,
        extra=extra,
    )


def available_profiles() -> list[str]:
    """Names of profiles declared via ``MODEL_PROFILES`` (plus legacy ``zai``)."""
    declared = _env("MODEL_PROFILES", default="zai")
    names = [n.strip() for n in declared.split(",") if n.strip()]
    # Always include the legacy zai profile if old vars are present and not listed.
    if "zai" not in names and (_env("ZAI_API_KEY") or _env("ZAI_MODEL")):
        names.append("zai")
    return names


def _build_chat_model(profile: ProviderProfile, *, temperature: float, timeout: float):
    """Construct the LangChain chat model for ``profile`` (lazy provider import).

    Raises a clear error naming the missing package when an optional provider
    is requested but not installed.
    """
    kind = profile.kind
    if kind not in _KIND_TABLE:
        raise ValueError(
            f"Unknown provider kind {kind!r} for profile {profile.name!r}. "
            f"Known kinds: {', '.join(sorted(_KIND_TABLE))}."
        )
    package, classname = _KIND_TABLE[kind]
    try:
        mod = __import__(package, fromlist=[classname])
        ChatCls = getattr(mod, classname)
    except ImportError as exc:
        hint = _INSTALL_HINT.get(kind, f"pip install {package}")
        raise RuntimeError(
            f"Provider package {package!r} is not installed (needed for profile "
            f"{profile.name!r}, kind {kind!r}). Install it with: {hint}"
        ) from exc

    kwargs: dict[str, Any] = {"model": profile.model, "temperature": temperature}

    if kind in ("openai", "openai_compatible", "azure_openai"):
        # OpenAI-compatible path. base_url selects the endpoint (Z.ai, OpenRouter,
        # Together, Groq, Fireworks, local vLLM, Ollama shim, ...).
        kwargs["api_key"] = profile.api_key or "not-required"
        if profile.base_url and kind != "azure_openai":
            kwargs["base_url"] = profile.base_url
        kwargs["timeout"] = timeout
        kwargs["max_retries"] = 0  # we run our own backoff in config._invoke_with_retry
        # Use certifi CA bundle for SSL on Windows (httpx doesn't use system store).
        try:
            import certifi
            import httpx
            kwargs["http_client"] = httpx.Client(verify=certifi.where())
            kwargs["http_async_client"] = httpx.AsyncClient(verify=certifi.where())
        except Exception:
            pass
    elif kind == "anthropic":
        if profile.api_key:
            kwargs["api_key"] = profile.api_key
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        kwargs["timeout"] = timeout
        kwargs["max_retries"] = 0
    elif kind == "ollama":
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        kwargs["timeout"] = int(timeout)
    elif kind == "google":
        if profile.api_key:
            kwargs["api_key"] = profile.api_key
        kwargs["timeout"] = timeout
    elif kind == "bedrock":
        kwargs["model_id"] = profile.model
        kwargs.pop("model")
        kwargs.pop("temperature", None)
    else:  # defensive — shouldn't reach
        if profile.api_key:
            kwargs["api_key"] = profile.api_key
        if profile.base_url:
            kwargs["base_url"] = profile.base_url

    kwargs.update(profile.extra)
    return ChatCls(**kwargs)


@lru_cache(maxsize=16)
def _cached(profile_key: str, temperature: float, timeout: float):
    """Cache constructed clients by (profile name, temperature, timeout)."""
    profile = load_profile(profile_key)
    if profile is None:
        raise RuntimeError(
            f"Provider profile {profile_key!r} is not configured. Set "
            f"MODEL_{profile_key}=<model> (and API_KEY_{profile_key}/BASE_URL_{profile_key} "
            f"as needed), or list it in MODEL_PROFILES."
        )
    return _build_chat_model(profile, temperature=temperature, timeout=timeout)


def get_chat_model(
    profile_name: str, *, temperature: float = 0.3, timeout: float = 180.0
):
    """Return a (cached) LangChain chat model for a named profile."""
    return _cached(profile_name.strip(), temperature, timeout)


def clear_cache() -> None:
    """Drop cached clients (used by tests and config reloads)."""
    _cached.cache_clear()


def build_for(profile_name: str | None, fallback: str) -> tuple[str, ProviderProfile]:
    """Resolve an effective profile name + object, with a fallback when unset."""
    name = (profile_name or "").strip() or fallback
    profile = load_profile(name)
    if profile is None:
        # Fall back to the provided fallback profile rather than hard-failing,
        # so a missing cheap profile silently uses the main one.
        profile = load_profile(fallback)
    if profile is None:
        raise RuntimeError(
            f"No provider profile configured (tried {name!r} and {fallback!r}). "
            "Set MODEL_PROFILES and MODEL_<name>, or the legacy ZAI_* variables."
        )
    return name, profile


# ---------------------------------------------------------------------------
# Cheap-task routing hook (kept here so providers.py owns all model selection).
# ---------------------------------------------------------------------------
def is_cheap_task(task_type: str) -> bool:
    cheap = {t.strip() for t in _env("CHEAP_TASKS", default="").split(",") if t.strip()}
    cheap |= {
        "summarization",
        "compression",
        "classification",
        "validation",
        "query_rewrite",
    }
    return task_type in cheap
