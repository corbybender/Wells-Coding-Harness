"""Interactive settings menu + .env persistence for the Wells harness.

Lets a user quickly inspect and change every harness setting from a single
interactive CLI menu (``coding-harness config`` or the top-level menu when
running with no args). Changes are written back to ``.env`` (comments and
ordering preserved) and also applied to the live process so they take effect
immediately on the next run.

Design goals:
  * Fast and EASY — numbered menu, type the number, type the new value, done.
  * Non-destructive to .env — only the touched keys change; comments/blank lines
    and unrelated keys are preserved verbatim.
  * Live + persistent — updated values are written to os.environ AND .env, so
    a subsequent ``coding-harness <goal>`` in the same shell picks them up too.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Settings schema
# ---------------------------------------------------------------------------
# Each setting is (env var, label, help, category, default, parser, formatter).
# Grouped by category so the menu can present them cleanly.


@dataclass
class Setting:
    key: str  # env var name
    label: str  # short label in the menu
    category: str  # grouping
    help: str  # one-line explanation
    default: str  # default value (string form)
    choices: tuple[str, ...] = ()  # enumerated allowed values, if any
    secret: bool = False  # mask the value when displaying


# Ordered list — also defines menu order.
SETTINGS: list[Setting] = [
    # --- Providers ---------------------------------------------------------
    Setting(
        "MODEL_PROFILE",
        "Active provider profile",
        "Providers",
        "Which named profile to use for the main reasoning/coding model.",
        "zai",
    ),
    Setting(
        "MODEL_PROFILE_CHEAP",
        "Cheap provider profile",
        "Providers",
        "Profile for low-stakes subtasks (summarize/compress). Blank = same as active.",
        "",
    ),
    Setting(
        "MODEL_PROFILES",
        "Available profiles",
        "Providers",
        "Comma-separated list of profile names that are configured.",
        "zai",
    ),
    # --- Active profile (zai) quick-edit shortcuts -------------------------
    Setting(
        "MODEL_zai",
        "Model (zai)",
        "Profile: zai",
        "Model id for the built-in 'zai' profile.",
        "glm-5.2",
    ),
    Setting(
        "API_KEY_zai",
        "API key (zai)",
        "Profile: zai",
        "API key for the 'zai' profile. Also seeded by ZAI_API_KEY.",
        "",
        secret=True,
    ),
    Setting(
        "BASE_URL_zai",
        "Base URL (zai)",
        "Profile: zai",
        "OpenAI-compatible base URL for the 'zai' profile.",
        "https://api.z.ai/api/paas/v4/",
    ),
    # --- Loop / run behaviour ---------------------------------------------
    Setting(
        "MAX_ITERATIONS",
        "Max coder<->reviewer iterations",
        "Run",
        "Cap on coder/tester/reviewer loop iterations (0 = no limit).",
        "0",
    ),
    Setting(
        "MAX_TOOL_STEPS",
        "Max tool steps per executor run",
        "Run",
        "Cap on tool-calling steps per executor run (0 = no limit).",
        "0",
    ),
    Setting(
        "PLAN_MODE",
        "Plan mode",
        "Run",
        "If on, the coder plans edits without applying them (review-first).",
        "0",
        choices=("0", "1"),
    ),
    Setting(
        "HARNESS_SAFETY",
        "Safety policy",
        "Run",
        "How writes/shell are handled: auto (run), approve (ask), dryrun (simulate).",
        "auto",
        choices=("auto", "approve", "dryrun"),
    ),
    Setting(
        "WORKSPACE_ROOT",
        "Workspace root",
        "Run",
        "Directory tools are confined to (path escapes are blocked).",
        "",
    ),
    Setting(
        "SHELL_TIMEOUT",
        "Shell command timeout (s)",
        "Run",
        "Max seconds for a single shell command.",
        "120",
    ),
    # --- Token optimization ------------------------------------------------
    Setting(
        "TOKEN_BUDGET_MAX_INPUT",
        "Max input tokens / call",
        "Tokens",
        "Input budget per call; above this, low-priority context is trimmed.",
        "24000",
    ),
    Setting(
        "TOKEN_BUDGET_SMALL_INPUT",
        "Small budget (summarize)",
        "Tokens",
        "Input budget for low-stakes calls (summarizer/compressor).",
        "8000",
    ),
    Setting(
        "TOKEN_BUDGET_RESERVED_OUTPUT",
        "Reserved output tokens",
        "Tokens",
        "Output tokens reserved so the model can always answer.",
        "4000",
    ),
    Setting(
        "SUMMARIZE_ON_LOOP",
        "Summarize on loop",
        "Tokens",
        "Replace durable context with a summary on loop iterations.",
        "1",
        choices=("0", "1"),
    ),
    Setting(
        "SUMMARIZE_THRESHOLD",
        "Summarize threshold (tokens)",
        "Tokens",
        "Summarize durable context once it exceeds this many tokens.",
        "1500",
    ),
    # --- LLM tuning --------------------------------------------------------
    Setting(
        "LLM_TIMEOUT",
        "LLM call timeout (s)",
        "LLM",
        "Per-call timeout for chat-model requests.",
        "180",
    ),
    Setting(
        "LLM_MAX_RETRIES",
        "LLM max retries",
        "LLM",
        "Retry attempts for transient errors (timeouts, 429, 5xx).",
        "5",
    ),
    Setting(
        "LLM_BACKOFF_BASE",
        "LLM backoff base",
        "LLM",
        "Exponential backoff base (seconds) between retries.",
        "2.0",
    ),
]

SETTINGS_BY_KEY: dict[str, Setting] = {s.key: s for s in SETTINGS}


# ---------------------------------------------------------------------------
# .env read / merge / write (comment-preserving)
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def _parse_env_file(path: Path) -> list[tuple[str, str, str]]:
    """Return ``[(raw_line, key, value)]``. For comment/blank lines key/value are ''."""
    if not path.exists():
        return []
    out: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _LINE_RE.match(line)
        if m:
            out.append((line, m.group(1), _strip_quotes(m.group(2))))
        else:
            out.append((line, "", ""))
    return out


def _strip_quotes(v: str) -> str:
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def _format_value(v: str) -> str:
    """Quote a value if it contains spaces or special chars when writing to .env."""
    if v == "":
        return ""
    if re.search(r"[\s#\"'`=]", v):
        return '"' + v.replace('"', '\\"') + '"'
    return v


def update_env_file(path: Path, changes: dict[str, str]) -> tuple[bool, list[str]]:
    """Apply ``changes`` to ``path``. Returns (any_written, changed_keys).

    Preserves comments, blank lines and unrelated keys. New keys are appended
    under a ``# Wells harness`` marker. Existing keys are updated in place
    (keeping their original line position and indentation style).
    """
    lines = _parse_env_file(path)
    keys_seen: set[str] = set()
    changed: list[str] = []
    out: list[str] = []

    for raw, key, _ in lines:
        if key in changes:
            new_val = _format_value(changes[key])
            # Preserve leading whitespace before KEY= if any.
            indent = re.match(r"^\s*", raw).group(0)
            out.append(f"{indent}{key}={new_val}")
            keys_seen.add(key)
            changed.append(key)
        else:
            out.append(raw)

    # Append new keys that didn't already exist.
    new_keys = [k for k in changes if k not in keys_seen]
    if new_keys:
        if out and out[-1].strip() != "":
            out.append("")
        out.append("# --- Added by Wells settings menu ---")
        for k in new_keys:
            out.append(f"{k}={_format_value(changes[k])}")
            changed.append(k)

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return (bool(changed), changed)


# ---------------------------------------------------------------------------
# Live value read/apply
# ---------------------------------------------------------------------------


def current_value(setting: Setting) -> str:
    """Live value: env var if set, else the configured default.

    For the 'zai' profile, also checks legacy ZAI_* variable names for backwards compatibility.
    """
    v = os.environ.get(setting.key)
    if v is not None and v.strip() != "":
        return v

    # Fallback to legacy ZAI_* names for the zai profile
    if "zai" in setting.key.lower():
        legacy_map = {
            "API_KEY_zai": "ZAI_API_KEY",
            "BASE_URL_zai": "ZAI_ENDPOINT",
            "MODEL_zai": "ZAI_MODEL",
        }
        legacy_key = legacy_map.get(setting.key)
        if legacy_key:
            v = os.environ.get(legacy_key)
            if v is not None and v.strip() != "":
                return v

    return setting.default


def apply_changes(changes: dict[str, str]) -> None:
    """Push changes into os.environ (live) so the rest of the run sees them."""
    for k, v in changes.items():
        os.environ[k] = v


def mask(v: str) -> str:
    if not v:
        return "(unset)"
    if len(v) <= 6:
        return "*" * len(v)
    return v[:3] + "*" * (len(v) - 6) + v[-3:]


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------


def _grouped() -> dict[str, list[Setting]]:
    groups: dict[str, list[Setting]] = {}
    for s in SETTINGS:
        groups.setdefault(s.category, []).append(s)
    return groups


def _print_header(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}\n {title}\n{bar}")


def show_overview() -> None:
    """Print a one-screen overview of all current settings (grouped)."""
    _print_header("Wells harness — current settings")
    for cat, items in _grouped().items():
        print(f"\n[{cat}]")
        for s in items:
            v = current_value(s)
            shown = mask(v) if s.secret else (v or "(default)")
            print(f"  {s.key:<28} {shown}")
    print("\n(Edit any of these from the menu. All are also plain .env vars.)")


def _prompt(label: str, default: str = "") -> str:
    """Prompt for a single value with an editable default."""
    suffix = f" [{default}]" if default != "" else ""
    try:
        raw = input(f"  {label}{suffix}: ").strip()
    except EOFError:
        raw = ""
    return raw or default


def _edit_setting(s: Setting) -> str | None:
    """Prompt for a new value for ``s``; return the new value or None to cancel."""
    cur = current_value(s)
    shown = mask(cur) if s.secret else (cur or "(default)")
    print(f"\n  {s.label}  ({s.key})")
    print(f"  {s.help}")
    print(f"  Current: {shown}")
    if s.choices:
        print(f"  Choices: {', '.join(s.choices)}")
        choice = _prompt("New value (blank to keep)", "").strip()
        if not choice:
            return None
        if choice not in s.choices:
            print(f"  ! Must be one of: {', '.join(s.choices)}")
            return None
        return choice
    new = _prompt("New value (blank to keep)", "")
    if new == cur:
        return None
    return new


def _edit_provider_profile() -> dict[str, str] | None:
    """Shortcut flow: pick the active profile, then edit its model/key/base_url.

    Returns changes to apply (or None if nothing changed).
    """
    from coding_harness import config, providers

    names = [n for n in config.MODEL_PROFILES.split(",") if n.strip()]
    print("\n  Available profiles:", ", ".join(names))
    print(f"  Active profile: {config.ACTIVE_PROFILE}")
    chosen = _prompt("Switch active profile to (blank to keep)", config.ACTIVE_PROFILE)
    changes: dict[str, str] = {}
    if chosen and chosen != config.ACTIVE_PROFILE:
        if chosen not in names:
            add = _prompt(
                f"Profile {chosen!r} is new. Add it to MODEL_PROFILES? [Y/n]", "y"
            )
            if add.lower() not in ("n", "no"):
                names.append(chosen)
                changes["MODEL_PROFILES"] = ",".join(names)
        changes["MODEL_PROFILE"] = chosen

    # Edit the chosen (or current) profile's fields.
    target = chosen or config.ACTIVE_PROFILE
    prof = providers.load_profile(target)
    cur_model = prof.model if prof else ""
    cur_url = prof.base_url if prof else ""
    new_model = _prompt(f"Model for {target}", cur_model)
    if new_model and new_model != cur_model:
        changes[f"MODEL_{target}"] = new_model
    new_key = _prompt(f"API key for {target} (blank to keep)")
    if new_key:
        changes[f"API_KEY_{target}"] = new_key
    new_url = _prompt(f"Base URL for {target}", cur_url)
    if new_url and new_url != cur_url:
        changes[f"BASE_URL_{target}"] = new_url
    return changes or None


def _add_provider_profile() -> dict[str, str] | None:
    """Flow to add a brand-new profile from scratch."""
    print("\n  Add a new provider profile")
    print("  Common names: zai, openai, openrouter, anthropic, ollama, local, ...")
    name = _prompt("Profile name").strip()
    if not name:
        return None
    model = _prompt("Model id")
    if not model:
        print("  ! A model id is required.")
        return None
    key = _prompt("API key (blank if none/local)")
    url = _prompt("Base URL (blank for provider default)")
    changes: dict[str, str] = {f"MODEL_{name}": model}
    if key:
        changes[f"API_KEY_{name}"] = key
    if url:
        changes[f"BASE_URL_{name}"] = url
    from coding_harness import config

    names = [n for n in config.MODEL_PROFILES.split(",") if n.strip()]
    if name not in names:
        names.append(name)
        changes["MODEL_PROFILES"] = ",".join(names)
    activate = _prompt(f"Make {name} the active profile? [Y/n]", "y")
    if activate.lower() not in ("n", "no"):
        changes["MODEL_PROFILE"] = name
    return changes


# Top-level menu actions: (key, label, handler).
# Handlers return a changes dict to persist/apply, or None.
Action = tuple[str, str, Callable[[], "dict[str, str] | None"]]


def _wrap_setting_handler(s: Setting) -> Callable[[], "dict[str, str] | None"]:
    """Wrap a Setting editor so it returns a {key: value} dict (or None)."""

    def _h() -> "dict[str, str] | None":
        new = _edit_setting(s)
        return {s.key: new} if new is not None else None

    return _h


def _menu_actions() -> list[Action]:
    actions: list[Action] = [
        ("p", "Switch / edit provider profile", _edit_provider_profile),
        ("+", "Add a new provider profile", _add_provider_profile),
    ]
    for s in SETTINGS:
        actions.append((s.key, f"{s.label}  ({s.category})", _wrap_setting_handler(s)))
    return actions


def interactive_menu(env_path: Path | None = None, *, loop: bool = True) -> bool:
    """Run the interactive settings menu. Returns True if anything changed.

    ``env_path`` defaults to ``.env`` in the current directory.
    """
    env_path = env_path or Path(".env")
    changed = False

    while True:
        show_overview()
        actions = _menu_actions()
        print("\n" + "-" * 64)
        print("What do you want to change?")
        print("  p) Switch / edit provider profile (fast path)")
        print("  +) Add a new provider profile")
        print("  --- or type an ENV VAR name to edit it directly ---")
        print("  s) Save & exit     q) Quit without saving     w) Write .env now")
        try:
            choice = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice in ("q", "quit", "exit"):
            break
        if choice in ("s", "save"):
            break
        if choice in ("w", "write"):
            if changed:
                update_env_file(env_path, _pending)
                print(f"  Wrote changes to {env_path}.")
                _pending.clear()
            else:
                print("  Nothing to write.")
            continue

        handler = None
        for key, _label, fn in actions:
            if choice == key.lower():
                handler = fn
                break
        if handler is None:
            # Allow typing any env var name directly too.
            if choice.upper() in SETTINGS_BY_KEY:
                handler = _wrap_setting_handler(SETTINGS_BY_KEY[choice.upper()])
            else:
                print(f"  ! Unknown choice: {choice!r}")
                continue

        result = handler()
        if result:
            _pending.update(result)
            apply_changes(result)
            changed = True
            print(f"  Applied: {', '.join(sorted(result))}")

        if not loop:
            break

    if changed and _pending:
        update_env_file(env_path, _pending)
        print(f"\nSaved {len(_pending)} change(s) to {env_path}.")
        _pending.clear()
    return changed


# Changes staged during the current menu session, flushed on save/write.
_pending: dict[str, str] = {}


def reset_pending() -> None:
    """Clear staged menu changes (used by tests)."""
    _pending.clear()


def parse_argv_settings(argv: list[str]) -> dict[str, str]:
    """Parse ``KEY=VALUE`` style overrides from argv (e.g. ``MAX_ITERATIONS=5``).

    Returns the parsed overrides; also applies them to os.environ live.
    """
    changes: dict[str, str] = {}
    for arg in argv:
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", arg)
        if m:
            changes[m.group(1)] = m.group(2)
    if changes:
        apply_changes(changes)
    return changes
