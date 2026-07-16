"""Configuration: loads credentials/settings from the environment and exposes an LLM client.

Provider/model selection is delegated to :mod:`wells.providers`, which
supports multiple named provider profiles (Z.ai, OpenAI, Anthropic, Ollama,
...) selected via env vars. This module re-exports the legacy ``ZAI_*`` names
for backwards compatibility and keeps the tuning knobs (retries, budgets,
summarization, workspace, safety policy) that the rest of the harness reads.
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage


def _configure_ca_bundle() -> None:
    """Inject system CA certs into Python's SSL layer.

    uv's bundled Python doesn't use the Windows certificate store by default.
    truststore patches ssl.SSLContext to use the OS native trust store (Windows
    Cert Store / macOS Keychain / Linux system bundle), which always has the
    right issuer chain for commercial APIs like Z.ai.
    """
    # Try truststore first: patches ssl.SSLContext to use OS native cert store.
    try:
        import truststore
        truststore.inject_into_ssl()
        return
    except Exception:
        pass

    # Already set — respect it (e.g. corporate proxy cert bundle).
    if os.environ.get("SSL_CERT_FILE"):
        return

    # On Linux/macOS try system bundles next.
    candidates = [
        "/etc/ssl/certs/ca-certificates.crt",  # Debian/Ubuntu/WSL
        "/etc/pki/tls/certs/ca-bundle.crt",  # RHEL/Fedora
        "/etc/ssl/cert.pem",  # Alpine/macOS
    ]
    for path in candidates:
        if os.path.exists(path):
            os.environ["SSL_CERT_FILE"] = path
            os.environ.setdefault("REQUESTS_CA_BUNDLE", path)
            os.environ.setdefault("CURL_CA_BUNDLE", path)
            return

    # Last resort: certifi's bundled CA certs.
    try:
        import certifi
        bundle = certifi.where()
        os.environ["SSL_CERT_FILE"] = bundle
        os.environ["REQUESTS_CA_BUNDLE"] = bundle
        os.environ["CURL_CA_BUNDLE"] = bundle
    except Exception:
        pass


_configure_ca_bundle()
# Load .env from the Wells package directory (where it's installed).
# When wells is installed at Q:\payload\nopayload\Wells,
# config.py is at Wells/src/wells/config.py,
# so go up 3 levels to reach Wells/.env
_wells_root = Path(__file__).parent.parent.parent
_env_path = _wells_root / ".env"
load_dotenv(_env_path)

from wells import providers  # noqa: E402 (must run after load_dotenv)
from wells.tokens import TokenBudget  # noqa: E402

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

# Limit knobs: 0 means NO LIMIT everywhere. The practical backstops when
# running unlimited are MAX_RUN_TOKENS, Escape (cooperative cancel), and the
# executor's stuck-loop detector.
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "0"))

# Retry tuning for transient network / rate-limit blips.
LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "180"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "5"))
LLM_BACKOFF_BASE: float = float(os.getenv("LLM_BACKOFF_BASE", "2.0"))

# Ollama (and other locally-served models) commonly load with a small default
# context window (Ollama: 4096 tokens) regardless of what the model actually
# supports — Wells' own system prompt (principles + rules + skills + tool
# catalog) can approach or exceed that on its own, causing silent mid-run
# truncation that reads as the model "losing the thread". Ollama's native API
# accepts a per-request context-size override that reloads the model at that
# size; the OpenAI-compatible endpoint (what profiles normally talk to) has
# no equivalent, so Wells can fire one native warm-up call per (endpoint,
# model) before first use.
#
# OFF by default (0) — measured live against a real 7B model on real Apple
# Silicon hardware, the reload this triggers took ~294s (not the sub-minute
# assumed when this was first wired up). That's a one-time cost per process,
# not per round, but most `wells "<task>"` invocations are a single fresh
# process — defaulting this on would silently tax every quick task with a
# ~5 minute wait up front, which is worse than the truncation risk it fixes.
# Opt in (e.g. OLLAMA_NUM_CTX=16384) for long/complex runs on a local model
# where truncation is the bigger risk, ideally warmed once before a work
# session rather than inline on a single task's critical path.
OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "0"))

# Keep the local Ollama model loaded in memory between requests. Ollama's
# server default unloads an idle model after ~5 minutes; the next request then
# silently re-pays the load (measured live: ~294s for a 7B model reloading at
# a larger context size, multi-second even at defaults). "-1" = keep loaded
# forever; any Ollama duration string ("30m") also works; empty disables.
# Piggybacks on the same native warm-up call as OLLAMA_NUM_CTX, so it costs
# nothing extra — a keep_alive ping against a loaded model is milliseconds.
OLLAMA_KEEP_ALIVE: str = os.getenv("OLLAMA_KEEP_ALIVE", "-1").strip()

# --- Token optimization configuration -------------------------------------
# These are the enforced context-trim ceiling for the executor's safety-drop
# pipeline (executor._effective_ctx_budget), not just informational — a
# request that would exceed max_input_tokens gets oldest rounds dropped
# before it's sent. BUDGET is the default; SMALL_BUDGET is auto-selected
# instead for a profile that looks like local Ollama.
BUDGET = TokenBudget(
    max_input_tokens=int(os.getenv("TOKEN_BUDGET_MAX_INPUT", "24000")),
    reserved_output_tokens=int(os.getenv("TOKEN_BUDGET_RESERVED_OUTPUT", "4000")),
)
# Sized to fit Ollama's own native default context (4096, regardless of what
# the model architecturally supports — see OLLAMA_NUM_CTX above) with a
# safety margin for chat-template overhead, since most local profiles never
# opt into raising it (that costs a multi-minute reload). When OLLAMA_NUM_CTX
# IS set, use that instead — we've explicitly raised the real ceiling, so the
# trim budget should track it rather than stay pinned to the crippled default.
SMALL_BUDGET = TokenBudget(
    max_input_tokens=int(os.getenv("TOKEN_BUDGET_SMALL_INPUT", str(OLLAMA_NUM_CTX or 3584))),
    reserved_output_tokens=int(os.getenv("TOKEN_BUDGET_SMALL_RESERVED_OUTPUT", "512")),
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
# A WORKSPACE_ROOT that doesn't exist (an .env copied from another machine, a
# deleted project dir) would make EVERY shell command fail with WinError 267 /
# FileNotFoundError — subprocess cwd must exist. Fall back to the current
# directory and record the bad value so the UI/doctor can surface it.
WORKSPACE_ROOT: str = os.getenv("WORKSPACE_ROOT", os.getcwd()).strip() or os.getcwd()
WORKSPACE_ROOT_INVALID: str = ""
if not os.path.isdir(WORKSPACE_ROOT):
    WORKSPACE_ROOT_INVALID = WORKSPACE_ROOT
    WORKSPACE_ROOT = os.getcwd()

# Safety policy for writes/shell. One of: auto | approve | dryrun.
#   auto    - execute immediately, confined to WORKSPACE_ROOT
#   approve - require an approval callback (caller-provided); dry-run otherwise
#   dryrun  - never execute, just describe what would happen
HARNESS_SAFETY: str = os.getenv("HARNESS_SAFETY", "auto").strip().lower() or "auto"

# Max tool-call steps in a single executor run (0 = no limit).
MAX_TOOL_STEPS: int = int(os.getenv("MAX_TOOL_STEPS", "0"))

# How many times the executor will coach a stalled model (produced text but
# zero tool calls, before it has taken any real action this run) back onto
# the tool-calling protocol before giving up and treating it as a genuine
# final answer. Guards weak/local models that ignore the protocol on the
# first try without risking an infinite loop for models that legitimately
# have nothing to do (e.g. answering a question, not a coding task).
STALL_NUDGE_MAX: int = int(os.getenv("STALL_NUDGE_MAX", "2"))

# Per-agent step caps (0 = no limit). Applied to the agentic planner, the
# tester/reviewer verification loops, and spawned subagents.
PLANNER_MAX_STEPS: int = int(os.getenv("PLANNER_MAX_STEPS", "0"))
TESTER_MAX_STEPS: int = int(os.getenv("TESTER_MAX_STEPS", "0"))
REVIEWER_MAX_STEPS: int = int(os.getenv("REVIEWER_MAX_STEPS", "0"))
SUBAGENT_MAX_STEPS: int = int(os.getenv("SUBAGENT_MAX_STEPS", "0"))

# Rules engine: deterministic enforcement of .wells/rules.yaml at the tool
# boundary (block/confirm/warn/liability). RULES_AUTODISCHARGE runs a bounded
# follow-up agent pass to close open liabilities (e.g. terminate a rented GPU)
# when a run tries to finish with one open.
RULES_ENFORCE: bool = os.getenv("RULES_ENFORCE", "1") not in ("0", "false", "no", "")
RULES_AUTODISCHARGE: bool = os.getenv("RULES_AUTODISCHARGE", "1") not in ("0", "false", "no", "")

# Auto-commit (opt-in): after each successful auto-mode run that changed the
# working tree, create a git commit with an LLM-generated Conventional Commits
# message and a Wells authorship trailer. /undo still works (checkpoint ref).
AUTO_COMMIT: bool = os.getenv("AUTO_COMMIT", "0") not in ("0", "false", "no", "")

# Route the tester/reviewer verification agents to the cheap model profile
# when one is configured (they are judgment-light relative to the coder).
CHEAP_VERIFY: bool = os.getenv("CHEAP_VERIFY", "1") not in ("0", "false", "no", "")

# Model cascade: when the executor is about to hard-stop a run as stuck
# (identical-call loop, invented tool names, task-drift thrashing) and this
# names a configured profile different from the one running, it switches to
# that profile ONCE and lets the run continue instead of stopping — a user on
# a 7B local model with an occasional cloud profile gets local-model costs
# with frontier-grade unsticking, paid only at the exact moment the small
# model has demonstrably failed. Empty (default) = off; a second stuck
# condition after escalating stops the run for real.
ESCALATION_PROFILE: str = os.getenv("ESCALATION_PROFILE", "").strip()

# Stepwise coding: run each numbered step of the planner's plan as its own
# short executor run with a fresh context (carrying only the goal, the list
# of completed steps, and the current step). Context never grows past one
# step's needs — the structural answer to small-context models drifting or
# truncating mid-run: a model can't lose the thread across 20 rounds when no
# run lasts 20 rounds. "auto" (default) = only when the active profile looks
# like local Ollama; "1" = always; "0" = never.
WELLS_STEPWISE: str = (os.getenv("WELLS_STEPWISE", "auto").strip().lower() or "auto")

# Self-heal: after every write/edit, run the fastest available checker for
# that file type (ruff/py_compile, node --check, json parse) and inject any
# failure into the agent's next observation. Set 0 to disable.
SELF_CHECK: bool = os.getenv("SELF_CHECK", "1") not in ("0", "false", "no", "")

# Per-run token budget: hard cap on input+output tokens across one run
# (all agents combined — the ledger is reset at run start). 0 disables the cap.
# A warning is printed when a run crosses 80% of the budget.
MAX_RUN_TOKENS: int = int(os.getenv("MAX_RUN_TOKENS", "0"))

# Max seconds for a single shell command run by the harness.
SHELL_TIMEOUT: float = float(os.getenv("SHELL_TIMEOUT", "120"))

# Plan mode: when true, the coder plans edits but does not apply them
# (produces a diff / step list only). Useful for review-first workflows.
PLAN_MODE: bool = os.getenv("PLAN_MODE", "0") not in ("0", "false", "no", "")

# Stream output to the console during generation if the model supports it.
STREAM_OUTPUT: bool = os.getenv("STREAM_OUTPUT", "1") not in ("0", "false", "no", "")

# Repetition kill-switch: for local (compact-prompt) profiles, stream the
# generation internally and abort it the moment the output degenerates into
# verbatim self-repetition ("I will now fix it. I will now fix it. ..." —
# a classic small-model failure). Closing the stream makes Ollama stop
# generating, so a doomed reply costs seconds instead of running to the
# token limit or the request timeout. Set 0 to disable.
STREAM_GUARD: bool = os.getenv("WELLS_STREAM_GUARD", "1") not in ("0", "false", "no", "")

# Intent routing: when the heuristic can't decide auto vs orchestrate, ask the
# cheap model (adds a full round-trip before ANY work starts). Default off —
# ambiguous requests route to auto, which handles everything; /orchestrate
# forces the full pipeline explicitly.
INTENT_LLM_FALLBACK: bool = os.getenv("INTENT_LLM_FALLBACK", "0") not in ("0", "false", "no", "")

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

# Agent skills: discoverable SKILL.md know-how loaded on demand via load_skill.
# Default on; set WELLS_SKILLS=0 to disable. Extra skill roots may be added via
# WELLS_SKILLS_PATHS (path-separator list).
WELLS_SKILLS: bool = os.getenv("WELLS_SKILLS", "1") not in ("0", "false", "no", "")

# CodeAct: a sandboxed run_code tool for in-context computation. Default on.
WELLS_CODEACT: bool = os.getenv("WELLS_CODEACT", "1") not in ("0", "false", "no", "")

# When one model reply contains several tool calls, run the leading run of
# read-only calls (read_file, grep, glob, ...) concurrently instead of one by
# one. Results land in the transcript in call order; mutating calls and
# everything after the first one stay strictly sequential, so ordering
# semantics are unchanged. Set 0 to disable.
PARALLEL_READS: bool = os.getenv("WELLS_PARALLEL_READS", "1") not in ("0", "false", "no", "")

# When the model re-issues an identical read-only call whose result is still
# verbatim in its recent context (within the last WELLS_KEEP_ROUNDS rounds,
# with no write/shell mutation since), skip the dispatch and return a pointer
# to the earlier step instead of re-paying the full output in tokens. Also
# makes re-read loops visible to the model immediately. Set 0 to disable.
DEDUPE_READS: bool = os.getenv("WELLS_DEDUPE_READS", "1") not in ("0", "false", "no", "")

# Structured outputs: for local Ollama profiles, constrain the model's reply
# to a tool-call JSON schema at the token-sampling level (Ollama "format" /
# OpenAI-compat "response_format: json_schema"). The sampler literally cannot
# emit malformed tool-call JSON — the whole class of unescaped-quote /
# prose-mixed-with-JSON parse failures observed live with 7B models dies at
# the source instead of being papered over after generation. The text parsers
# remain as fallback for providers without schema support. Set 0 to disable
# (e.g. an old Ollama server that rejects response_format; the executor also
# auto-falls-back once per run if the server errors on it).
STRUCTURED_OUTPUTS: bool = os.getenv("WELLS_STRUCTURED", "1") not in ("0", "false", "no", "")

# Background agents: bg_start / bg_status / bg_collect for concurrent fan-out
# (async counterpart to the blocking parallel_research). Default on.
WELLS_BG_AGENTS: bool = os.getenv("WELLS_BG_AGENTS", "1") not in ("0", "false", "no", "")


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
    from wells.control import CONTROL
    from wells.logger import log_error, log_path
    last_err: Exception | None = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            return llm.invoke(messages)
        except Exception as err:
            last_err = err
            log_error(f"LLM invoke attempt {attempt}/{LLM_MAX_RETRIES}: {type(err).__name__}: {err}", err)
            if not _is_transient(err) or attempt == LLM_MAX_RETRIES:
                break
            backoff = min(LLM_BACKOFF_BASE**attempt, 30.0)
            print(
                f"[llm] transient {type(err).__name__} on attempt {attempt}/"
                f"{LLM_MAX_RETRIES}; retrying in {backoff:.1f}s ..."
            )
            # Sleep in small slices so a cancel lands within ~250ms instead
            # of blocking for the full backoff (up to 30s per attempt).
            waited = 0.0
            while waited < backoff and not CONTROL.cancelled():
                step = min(0.25, backoff - waited)
                time.sleep(step)
                waited += step
            if CONTROL.cancelled():
                break
    assert last_err is not None
    print(f"[llm] all retries failed. Full error logged to: {log_path()}")
    raise last_err


def ask_llm(prompt: str, temperature: float = 0.3) -> str:
    """Legacy convenience wrapper: one human message -> response text.

    New agent code uses :func:`wells.runtime.run_step` instead, which
    accounts for tokens. This is kept for ad-hoc / external use.
    """
    try:
        resp = _invoke_with_retry(get_llm(temperature), [HumanMessage(content=prompt)])
        return (resp.content or "").strip()
    except Exception as err:
        msg = f"[LLM call failed after {LLM_MAX_RETRIES} attempts: {type(err).__name__}: {str(err)[:200]}]"
        print(f"[llm] giving up: {msg}")
        return msg
