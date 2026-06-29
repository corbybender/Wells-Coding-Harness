"""Configuration: loads credentials/settings from the environment and exposes an LLM client.

Provider/model selection is delegated to :mod:`coding_harness.providers`, which
supports multiple named provider profiles (Z.ai, OpenAI, Anthropic, Ollama,
...) selected via env vars. This module re-exports the legacy ``ZAI_*`` names
for backwards compatibility and keeps the tuning knobs (retries, budgets,
summarization, workspace, safety policy) that the rest of the harness reads.
"""

import os
import time

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage


def _configure_ca_bundle() -> None:
    """Point OpenSSL at a usable CA bundle.

    uv's standalone CPython builds ship their own OpenSSL which, on some Linux
    systems, has no default CA file (``ssl.get_default_verify_paths().cafile``
    is None), causing ``CERTIFICATE_VERIFY_FAILED`` on every HTTPS call. We fix
    this by exporting ``SSL_CERT_FILE``/``SSL_CERT_DIR`` to the first bundle we
    can find (system bundle, then certifi's). A user-provided value wins.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return

    candidates = [
        "/etc/ssl/certs/ca-certificates.crt",  # Debian/Ubuntu/WSL
        "/etc/pki/tls/certs/ca-bundle.crt",  # RHEL/Fedora
        "/etc/ssl/cert.pem",  # Alpine/macOS
        "/etc/ssl/certs",  # capath fallback
    ]
    for path in candidates:
        if os.path.exists(path):
            os.environ["SSL_CERT_FILE"] = path
            os.environ.setdefault("REQUESTS_CA_BUNDLE", path)
            os.environ.setdefault("CURL_CA_BUNDLE", path)
            return

    try:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())
    except Exception:
        pass


_configure_ca_bundle()
load_dotenv()

from coding_harness import providers  # noqa: E402 (must run after load_dotenv)
from coding_harness.tokens import TokenBudget  # noqa: E402

# ---------------------------------------------------------------------------
# Backwards-compatible legacy ZAI_* names.
# These are now *seeds* for the built-in ``zai`` provider profile. New code
# should read ACTIVE_PROFILE / CHEAP_PROFILE and call get_llm_for_task.
# ---------------------------------------------------------------------------
ZAI_API_KEY: str = os.getenv("ZAI_API_KEY", "").strip()
ZAI_ENDPOINT: str = os.getenv("ZAI_ENDPOINT", "https://api.z.ai/api/paas/v4/").strip()
ZAI_MODEL: str = os.getenv("ZAI_MODEL", "glm-5.2").strip()
ZAI_MODEL_CHEAP: str = os.getenv("ZAI_MODEL_CHEAP", "").strip()

# ---------------------------------------------------------------------------
# Model selection (new). MODEL_PROFILES lists available profiles (default
# ``zai``); MODEL_PROFILE selects the active one; MODEL_PROFILE_CHEAP selects
# the low-stakes model (defaults to the active profile when unset/missing).
# ---------------------------------------------------------------------------
MODEL_PROFILES: str = os.getenv("MODEL_PROFILES", "zai").strip() or "zai"
ACTIVE_PROFILE: str = os.getenv("MODEL_PROFILE", "").strip() or "zai"
CHEAP_PROFILE: str = os.getenv("MODEL_PROFILE_CHEAP", "").strip()
# Shown in logs/reports; falls back to the active profile's resolved model.
ACTIVE_MODEL_LABEL: str = os.getenv("ACTIVE_MODEL_LABEL", "").strip()

MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))

# Retry tuning for transient network / rate-limit blips.
LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "180"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "5"))
LLM_BACKOFF_BASE: float = float(os.getenv("LLM_BACKOFF_BASE", "2.0"))

# --- Token optimization configuration -------------------------------------
BUDGET = TokenBudget(
    max_input_tokens=int(os.getenv("TOKEN_BUDGET_MAX_INPUT", "24000")),
    reserved_output_tokens=int(os.getenv("TOKEN_BUDGET_RESERVED_OUTPUT", "4000")),
)
SMALL_BUDGET = TokenBudget(
    max_input_tokens=int(os.getenv("TOKEN_BUDGET_SMALL_INPUT", "8000")),
    reserved_output_tokens=int(os.getenv("TOKEN_BUDGET_RESERVED_OUTPUT", "4000")),
)
# Replace verbatim plan/architecture with a summary on loop iterations when the
# durable context exceeds this many (estimated) tokens. Set 0 to disable.
SUMMARIZE_ON_LOOP: bool = os.getenv("SUMMARIZE_ON_LOOP", "1") not in ("0", "false", "")
SUMMARIZE_THRESHOLD: int = int(os.getenv("SUMMARIZE_THRESHOLD", "1500"))

# Task types routed to the cheaper model (Phase 5: model router).
CHEAP_TASKS = {
    "summarization",
    "compression",
    "classification",
    "validation",
    "query_rewrite",
}

# --- Agentic execution configuration (Layer 1/2) -------------------------
# Workspace root: tools are confined to this directory (prevents path escapes).
WORKSPACE_ROOT: str = os.getenv("WORKSPACE_ROOT", os.getcwd()).strip() or os.getcwd()

# Safety policy for writes/shell. One of: auto | approve | dryrun.
#   auto    - execute immediately, confined to WORKSPACE_ROOT
#   approve - require an approval callback (caller-provided); dry-run otherwise
#   dryrun  - never execute, just describe what would happen
HARNESS_SAFETY: str = os.getenv("HARNESS_SAFETY", "auto").strip().lower() or "auto"

# Max tool-call steps in a single executor run before the loop is forced to stop.
MAX_TOOL_STEPS: int = int(os.getenv("MAX_TOOL_STEPS", "25"))

# Max seconds for a single shell command run by the harness.
SHELL_TIMEOUT: float = float(os.getenv("SHELL_TIMEOUT", "120"))

# Plan mode: when true, the coder plans edits but does not apply them
# (produces a diff / step list only). Useful for review-first workflows.
PLAN_MODE: bool = os.getenv("PLAN_MODE", "0") not in ("0", "false", "no", "")

# Stream output to the console during generation if the model supports it.
STREAM_OUTPUT: bool = os.getenv("STREAM_OUTPUT", "1") not in ("0", "false", "no", "")

# Auto-build/update the structural repo index before each harness run (if available).
# Set to 0 to disable automatic indexing.
INDEX_AUTO_UPDATE: bool = os.getenv("INDEX_AUTO_UPDATE", "1") not in ("0", "false", "no", "")

# Commands blocked from run_command regardless of safety policy (regex patterns,
# ``|``-separated — each piece is compiled independently).
BLOCKED_COMMANDS: tuple[str, ...] = tuple(
    s.strip()
    for s in os.getenv(
        "BLOCKED_COMMANDS",
        r"rm\s+-rf\s+/|mkfs|dd\s+if=|:\(\)\s*\{|shutdown|reboot",
    ).split("|")
    if s.strip()
)


def active_profile_name() -> str:
    """Name of the currently active provider profile."""
    return ACTIVE_PROFILE


def cheap_profile_name() -> str:
    """Name of the cheap provider profile (falls back to active when unset/missing)."""
    if CHEAP_PROFILE and CHEAP_PROFILE in MODEL_PROFILES.split(","):
        return CHEAP_PROFILE
    return ACTIVE_PROFILE


def model_name_for_task(task_type: str) -> str:
    """Pick the model *label* for a task type.

    Cheap subtasks use the cheap profile's model when configured, else the main
    profile's model. Returns the human label (e.g. ``zai:glm-5.2``).
    """
    name = cheap_profile_name() if task_type in CHEAP_TASKS else ACTIVE_PROFILE
    profile = providers.load_profile(name)
    if profile is None:
        profile = providers.load_profile(ACTIVE_PROFILE)
    if profile is None:
        # Ultimate fallback: legacy ZAI_MODEL value so old configs keep working.
        return ACTIVE_MODEL_LABEL or ZAI_MODEL or "glm-5.2"
    return profile.label()


def get_llm(temperature: float = 0.3):
    """Cached chat-model client for the active provider profile."""
    return providers.get_chat_model(
        ACTIVE_PROFILE, temperature=temperature, timeout=LLM_TIMEOUT
    )


def get_llm_for_task(task_type: str, temperature: float = 0.3):
    """Cached chat-model client selected by the model router for ``task_type``."""
    name = cheap_profile_name() if task_type in CHEAP_TASKS else ACTIVE_PROFILE
    return providers.get_chat_model(name, temperature=temperature, timeout=LLM_TIMEOUT)


def _is_transient(err: Exception) -> bool:
    """True for errors worth retrying (timeouts, connection issues, 429, 5xx)."""
    try:
        import openai

        if isinstance(
            err,
            (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.InternalServerError,
            ),
        ):
            return True
    except Exception:
        pass
    return False


def _invoke_with_retry(llm, messages):
    """Invoke ``llm`` with ``messages``, retrying transient failures.

    OpenAI-level retries are disabled on the client (max_retries=0); we run this
    backoff loop so progress is logged and transient TLS/429 errors are survived.
    Raises the last error if every retry fails.
    """
    last_err: Exception | None = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            return llm.invoke(messages)
        except Exception as err:
            last_err = err
            if not _is_transient(err) or attempt == LLM_MAX_RETRIES:
                break
            backoff = min(LLM_BACKOFF_BASE**attempt, 30.0)
            print(
                f"[llm] transient {type(err).__name__} on attempt {attempt}/"
                f"{LLM_MAX_RETRIES}; retrying in {backoff:.1f}s ..."
            )
            time.sleep(backoff)
    assert last_err is not None
    raise last_err


def ask_llm(prompt: str, temperature: float = 0.3) -> str:
    """Legacy convenience wrapper: one human message -> response text.

    New agent code uses :func:`coding_harness.runtime.run_step` instead, which
    accounts for tokens. This is kept for ad-hoc / external use.
    """
    try:
        resp = _invoke_with_retry(get_llm(temperature), [HumanMessage(content=prompt)])
        return (resp.content or "").strip()
    except Exception as err:
        msg = f"[LLM call failed after {LLM_MAX_RETRIES} attempts: {type(err).__name__}: {str(err)[:200]}]"
        print(f"[llm] giving up: {msg}")
        return msg
