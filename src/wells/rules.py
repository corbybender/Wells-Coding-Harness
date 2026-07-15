"""Rules engine: deterministic enforcement of operating rules.

The problem this solves: prompted rules are probabilistic — every model
eventually forgets or ignores a wall of rules at prompt top. So rules live in
three enforcement tiers, strongest first:

1. **Tool-boundary enforcement** (this module): rules from
   ``.wells/rules.yaml`` are checked against every tool call *before* it
   executes. ``block`` refuses, ``confirm`` routes through the approval gate,
   ``warn`` injects the rule verbatim into the model's next observation, and
   ``liability`` registers a stateful obligation (e.g. "rented GPU must be
   terminated") that the run **cannot silently close** while open.
2. **Moment-of-relevance injection**: when a rule fires, its text lands in the
   tool observation the model reads next — one rule, at the exact moment it
   applies, in the freshest context position.
3. **Prompt + audit**: the workspace ``RULES.md`` is injected into every
   system prompt, and the reviewer audits compliance.

Liabilities are persisted to ``~/.wells/liabilities.json`` so a crash or
restart never loses track of a running paid resource.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

_GLOBAL_RULES = Path.home() / ".wells" / "rules.yaml"
_LIABILITY_FILE = Path.home() / ".wells" / "liabilities.json"

# Default machine rules, used when neither the workspace nor the global file
# exists, and written to ~/.wells/rules.yaml on first run so every workspace
# gets the money-protecting rules out of the box. Kept in sync with the Wells
# repo's own .wells/rules.yaml.
DEFAULT_RULES_YAML = r"""
rules:
  - id: gpu-rental-teardown
    severity: liability
    open: '\b(vastai|vast\.ai)\s+(create|launch)|\brunpod(ctl)?\s+(create|start|resume)|\brunpod\.(create_pod|resume_pod)|\blambda\s+cloud.*(launch|start)|\bgcloud\s+compute\s+instances\s+create|\baws\s+ec2\s+run-instances|\baz\s+vm\s+create'
    close: '\b(vastai|vast\.ai)\s+(destroy|stop)|\brunpod(ctl)?\s+(remove|stop|terminate)|\brunpod\.(terminate_pod|stop_pod)|\blambda\s+cloud.*(terminate|delete)|\bgcloud\s+compute\s+instances\s+(delete|stop)|\baws\s+ec2\s+(terminate|stop)-instances|\baz\s+vm\s+(delete|deallocate)'
    message: >-
      RULE R1: You just started a PAID cloud resource. It is now a tracked
      liability. You MUST terminate it (and verify termination with a status
      check) before this task can be considered complete. State your
      termination plan now.

  - id: interactive-destroy
    severity: warn
    trigger: { tool: run_command, pattern: '\bvastai\s+destroy\s+instance\s+\d+\s*$' }
    message: >-
      RULE R13: 'vastai destroy' without --yes blocks on an interactive prompt
      and will hang in automation. Re-run with --yes.

  - id: no-bulk-rsync
    severity: confirm
    trigger: { tool: run_command, pattern: '\b(rsync|scp)\b(?=.*\b(\w+@[\w.\-]+|ssh)\b).*(\s\.\s|\s\./\s|\*\s|\s~?/?(home|Projects|workspace)/?\s)' }
    message: >-
      RULE R12: This looks like a bulk copy of a whole directory tree to a
      remote host. Copy only the files the job needs; exclude archives, logs,
      caches, and gitignored artifacts. Confirm this transfer is intentional.

  - id: no-force-push
    severity: confirm
    trigger: { tool: run_command, pattern: '\bgit\s+push\b.*(\s--force\b|\s-f\b)' }
    message: >-
      Force-push rewrites remote history. Confirm this is intentional.

  - id: no-hard-reset-clean
    severity: confirm
    trigger: { tool: run_command, pattern: '\bgit\s+(reset\s+--hard|clean\s+-[a-z]*f)' }
    message: >-
      This git command permanently discards uncommitted work. Confirm.

  - id: auth-preflight
    severity: warn
    trigger: { tool: run_command, pattern: '\b(nohup|setsid|screen|tmux)\b.*\b(train|upload|deploy|sync)\b|\bhuggingface-cli\s+upload|\bhf\s+upload' }
    message: >-
      RULE R7: This starts a long-running job that likely depends on external
      auth. Verify the token/credentials with ONE live authenticated call
      FIRST (e.g. whoami). Cached tokens expire; a 401 one minute in wastes
      the whole run.

  - id: no-logstring-monitors
    severity: warn
    trigger: { tool: write_file, pattern: 'grep\s+-q?\s*["\x27](FAILED|ERROR|DONE|COMPLETE)' }
    message: >-
      RULE R8: This looks like a log-string monitor. Monitors must check
      PROCESS LIVENESS (kill -0 / pgrep) — tracebacks, OOM kills, and silent
      deaths never print your sentinel string.
"""


def ensure_global_template() -> None:
    """Write ~/.wells/rules.yaml with the defaults if it doesn't exist."""
    try:
        if not _GLOBAL_RULES.exists():
            _GLOBAL_RULES.parent.mkdir(parents=True, exist_ok=True)
            _GLOBAL_RULES.write_text(DEFAULT_RULES_YAML.lstrip(), encoding="utf-8")
    except Exception:
        pass

_LOCK = threading.Lock()
_ENGINES: dict[str, "RulesEngine"] = {}


@dataclass
class Rule:
    id: str
    severity: str  # block | confirm | warn | liability
    message: str
    tool: str = ""          # trigger tool name ('' = any)
    pattern: str = ""       # trigger regex on the args text
    open: str = ""          # liability-open regex (run_command)
    close: str = ""         # liability-close regex (run_command)

    def __post_init__(self) -> None:
        self._pattern_re = re.compile(self.pattern, re.IGNORECASE) if self.pattern else None
        self._open_re = re.compile(self.open, re.IGNORECASE) if self.open else None
        self._close_re = re.compile(self.close, re.IGNORECASE) if self.close else None


@dataclass
class Decision:
    """Outcome of checking one tool call against the rules."""

    allow: bool = True
    confirm: bool = False      # route through the approval gate
    rule: Rule | None = None
    notes: list[str] = field(default_factory=list)  # inject into obs_text
    # Liability transitions matched by this call — applied only AFTER the
    # command actually succeeds (a failed `vastai create` starts nothing).
    liability_open: Rule | None = None
    liability_close: Rule | None = None
    _detail: str = ""


def _args_text(tool: str, args: dict) -> str:
    if tool in ("run_command", "shell", "bash"):
        return str(args.get("command") or args.get("cmd") or "")
    try:
        return json.dumps(args, default=str)
    except Exception:
        return str(args)


def _parse_rules(raw: dict) -> list[Rule]:
    out: list[Rule] = []
    for item in (raw or {}).get("rules") or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        trig = item.get("trigger") or {}
        try:
            out.append(Rule(
                id=str(item["id"]),
                severity=str(item.get("severity", "warn")).lower(),
                message=str(item.get("message", "")).strip(),
                tool=str(trig.get("tool", "")),
                pattern=str(trig.get("pattern", "")),
                open=str(item.get("open", "")),
                close=str(item.get("close", "")),
            ))
        except re.error:
            # A broken regex must not take the harness down; skip the rule.
            continue
    return out


class RulesEngine:
    """Per-workspace rule set + persistent liability ledger."""

    def __init__(self, workspace: str) -> None:
        self.workspace = str(workspace)
        self.rules: list[Rule] = []
        self.load()

    # -- loading -------------------------------------------------------------

    def load(self) -> None:
        import yaml

        merged: dict[str, Rule] = {}
        for path in (_GLOBAL_RULES, Path(self.workspace) / ".wells" / "rules.yaml"):
            try:
                if path.exists():
                    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                    for r in _parse_rules(raw):
                        merged[r.id] = r  # workspace overrides global by id
            except Exception:
                continue
        if not merged:
            # No rule files anywhere — fall back to the embedded defaults so
            # the money-protecting rules are always on.
            try:
                for r in _parse_rules(yaml.safe_load(DEFAULT_RULES_YAML) or {}):
                    merged[r.id] = r
            except Exception:
                pass
        self.rules = list(merged.values())

    # -- liabilities (persisted) ----------------------------------------------

    _liab_cache: tuple[float, list] = (0.0, [])

    def _read_liabilities(self) -> list[dict]:
        # mtime-cached: the TUI status bar polls this several times a second.
        try:
            mtime = _LIABILITY_FILE.stat().st_mtime
        except OSError:
            return []
        if RulesEngine._liab_cache[0] == mtime:
            return RulesEngine._liab_cache[1]
        try:
            data = json.loads(_LIABILITY_FILE.read_text(encoding="utf-8"))
            data = data if isinstance(data, list) else []
        except Exception:
            data = []
        RulesEngine._liab_cache = (mtime, data)
        return data

    def _write_liabilities(self, items: list[dict]) -> None:
        try:
            _LIABILITY_FILE.parent.mkdir(parents=True, exist_ok=True)
            _LIABILITY_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")
            RulesEngine._liab_cache = (0.0, [])  # invalidate poll cache
        except Exception:
            pass

    def open_liabilities(self) -> list[dict]:
        """Open liabilities for this workspace."""
        return [
            l for l in self._read_liabilities()
            if l.get("workspace") == self.workspace
        ]

    def _open_liability(self, rule: Rule, detail: str) -> None:
        with _LOCK:
            items = self._read_liabilities()
            items.append({
                "rule_id": rule.id,
                "workspace": self.workspace,
                "detail": detail[:200],
                "opened_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            self._write_liabilities(items)

    def _close_liability(self, rule: Rule) -> int:
        """Discharge all open liabilities for a rule; returns count closed."""
        with _LOCK:
            items = self._read_liabilities()
            keep = [
                l for l in items
                if not (l.get("rule_id") == rule.id
                        and l.get("workspace") == self.workspace)
            ]
            closed = len(items) - len(keep)
            if closed:
                self._write_liabilities(keep)
            return closed

    def discharge(self, rule_id: str) -> int:
        """Manually acknowledge/close liabilities for a rule id (user action)."""
        with _LOCK:
            items = self._read_liabilities()
            keep = [
                l for l in items
                if not (l.get("rule_id") == rule_id
                        and l.get("workspace") == self.workspace)
            ]
            closed = len(items) - len(keep)
            if closed:
                self._write_liabilities(keep)
            return closed

    # -- the gate --------------------------------------------------------------

    def check(self, tool: str, args: dict) -> Decision:
        """Check one tool call. Called by the executor before dispatch.

        Trigger rules (block/confirm/warn) resolve here; liability
        transitions are only *matched* here and applied by
        :meth:`apply_liability` after the command succeeds.
        """
        text = _args_text(tool, args)
        d = Decision(_detail=text)
        for r in self.rules:
            # Liability open/close (commands only) — matched, applied later.
            if r.severity == "liability" and tool in ("run_command", "shell", "bash"):
                if r._close_re and r._close_re.search(text):
                    d.liability_close = r
                elif r._open_re and r._open_re.search(text):
                    d.liability_open = r
                continue

            # Trigger rules.
            if r._pattern_re is None:
                continue
            if r.tool and r.tool != tool:
                continue
            if not r._pattern_re.search(text):
                continue
            if r.severity == "block":
                d.allow = False
                d.rule = r
                d.notes.append(f"[RULES {r.id} — BLOCKED: {r.message}]")
                return d
            if r.severity == "confirm":
                d.confirm = True
                d.rule = d.rule or r
                d.notes.append(f"[RULES {r.id}: {r.message}]")
            else:  # warn
                d.notes.append(f"[RULES {r.id}: {r.message}]")
        return d

    def apply_liability(self, d: Decision, *, ok: bool, simulated: bool) -> list[str]:
        """Apply matched liability transitions after the command ran.

        Only successful, non-simulated commands change the ledger. Returns
        notes to inject into the model's observation.
        """
        notes: list[str] = []
        if not ok or simulated:
            if d.liability_open is not None:
                notes.append(
                    f"[RULES {d.liability_open.id}: the resource-start command "
                    f"did not succeed, so no liability was registered — but if "
                    f"any resource DID partially start, verify and clean up.]"
                )
            return notes
        if d.liability_close is not None:
            n = self._close_liability(d.liability_close)
            if n:
                notes.append(
                    f"[RULES: liability '{d.liability_close.id}' discharged "
                    f"({n} closed). Verify the resource is actually terminated "
                    f"with a status check before moving on.]"
                )
        if d.liability_open is not None:
            self._open_liability(d.liability_open, d._detail)
            notes.append(f"[RULES {d.liability_open.id}: {d.liability_open.message}]")
        return notes

    # -- prompt + status blocks --------------------------------------------------

    def prompt_block(self, *, compact: bool = False) -> str:
        """OPERATING RULES block for the system prompt (RULES.md + liabilities).

        ``compact`` swaps the full RULES.md dump (routinely 1000+ tokens —
        the single largest piece of the system prompt, measured at 1710
        tokens/41% of the total for this repo) for a short pointer. Safe to
        do: the machine-checkable rules are enforced by ``check()`` against
        ``.wells/rules.yaml`` regardless of what the model can see in its own
        prompt — a violation still gets caught and explained at the moment
        it's attempted, via the tool-call observation, not just via this
        block. Used for local models with a small context window, where the
        full prose dump can push the system prompt past what the model can
        even see, silently truncating the task itself out of view.
        """
        parts: list[str] = []
        rules_md = Path(self.workspace) / "RULES.md"
        if rules_md.exists():
            try:
                if compact:
                    parts.append(
                        "OPERATING RULES: this workspace has mandatory rules "
                        "(RULES.md) enforced automatically by the harness at every "
                        "tool call — a violation will be blocked or flagged with "
                        "the specific rule cited when you attempt it. You don't "
                        "need to memorize them; just heed the feedback if one fires."
                    )
                else:
                    parts.append(
                        "OPERATING RULES (mandatory — violations have cost real money; "
                        "the harness enforces the machine-checkable ones and audits the rest):\n"
                        + rules_md.read_text(encoding="utf-8", errors="replace")[:9000]
                    )
            except Exception:
                pass
        open_l = self.open_liabilities()
        if open_l:
            lines = "\n".join(
                f"  - [{l['rule_id']}] opened {l['opened_at']}: {l['detail'][:90]}"
                for l in open_l
            )
            parts.append(
                f"⚠ OPEN LIABILITIES ({len(open_l)}) — these obligations are "
                f"UNDISCHARGED and MUST be resolved (terminate/close + verify) "
                f"before any task is complete:\n{lines}"
            )
        return ("\n\n".join(parts) + "\n") if parts else ""

    def liability_summary(self) -> str:
        """One-line summary for warnings/status ('' when none open)."""
        open_l = self.open_liabilities()
        if not open_l:
            return ""
        return "; ".join(
            f"{l['rule_id']} ({l['detail'][:60]})" for l in open_l
        )


def engine_for(workspace: str) -> RulesEngine:
    """Shared per-workspace engine (rules cached; liabilities always re-read)."""
    ws = str(workspace)
    with _LOCK:
        eng = _ENGINES.get(ws)
        if eng is None:
            eng = RulesEngine(ws)
            _ENGINES[ws] = eng
        return eng


def reload_all() -> None:
    with _LOCK:
        for eng in _ENGINES.values():
            eng.load()
