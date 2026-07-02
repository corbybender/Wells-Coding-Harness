"""Session persistence for Wells.

Sessions are stored as JSON files in ~/.wells/sessions/.
Each file captures the goal, outcome, files changed, and token usage from one
harness run, enabling history browsing and context-carrying resume.

Storage: ~/.wells/sessions/YYYYMMDD-HHMMSS-xxxxxx.json
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

SESSIONS_DIR = Path.home() / ".wells" / "sessions"

# Session IDs look like: 20260701-143022-a3b4c5
_SESSION_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


def is_session_id(s: str) -> bool:
    return bool(_SESSION_ID_RE.match(s.strip()))


def new_session_id() -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}-{short}"


def _ensure_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def save_session(session_id: str, data: dict) -> Path:
    _ensure_dir()
    path = SESSIONS_DIR / f"{session_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def load_session(session_id: str) -> dict | None:
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_sessions(workspace: str | None = None, limit: int = 50) -> list[dict]:
    """Return sessions sorted newest-first, optionally filtered by workspace."""
    _ensure_dir()
    sessions = []
    for p in sorted(
        SESSIONS_DIR.glob("*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    ):
        try:
            with open(p, encoding="utf-8") as f:
                s = json.load(f)
            if workspace and s.get("workspace") != workspace:
                continue
            sessions.append(s)
        except Exception:
            continue
    return sessions[:limit]


def delete_session(session_id: str) -> bool:
    path = SESSIONS_DIR / f"{session_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def clear_sessions(workspace: str | None = None) -> int:
    """Delete all sessions, or only those matching workspace. Returns count deleted."""
    _ensure_dir()
    count = 0
    for p in list(SESSIONS_DIR.glob("*.json")):
        if workspace:
            try:
                with open(p, encoding="utf-8") as f:
                    s = json.load(f)
                if s.get("workspace") != workspace:
                    continue
            except Exception:
                pass
        p.unlink()
        count += 1
    return count


def format_age(created_at: str) -> str:
    """Return human-readable age: 'just now', '14m ago', '3h ago', '2d ago', '2026-06-28'."""
    try:
        dt = datetime.fromisoformat(created_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        secs = (datetime.now(tz=timezone.utc) - dt).total_seconds()
        if secs < 90:
            return "just now"
        if secs < 3600:
            return f"{int(secs / 60)}m ago"
        if secs < 86400:
            return f"{int(secs / 3600)}h ago"
        if secs < 86400 * 7:
            return f"{int(secs / 86400)}d ago"
        if secs < 86400 * 30:
            return f"{int(secs / 86400 / 7)}w ago"
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "?"


def session_from_final_state(
    session_id: str,
    goal: str,
    final_state: dict,
    workspace: str,
    tokens_in: int,
    tokens_out: int,
    duration_seconds: int,
    resumed_from: str | None = None,
    in_progress: bool = False,
) -> dict:
    """Build a session dict from a harness run.

    With ``in_progress=True`` the session is a mid-run checkpoint: saved after
    every graph node so a crash/kill loses at most one node's worth of work
    and ``/resume`` can pick up from the last checkpoint.
    """
    files = _files_from_git(workspace, final_state.get("git_summary", ""))
    if in_progress:
        status = "IN_PROGRESS"
    elif final_state.get("review_complete"):
        status = "COMPLETE"
    else:
        status = "INCOMPLETE"
    return {
        "id": session_id,
        "created_at": datetime.now().isoformat(),
        "workspace": workspace,
        "goal": goal,
        "status": status,
        "iterations": final_state.get("iteration", 0),
        "files_modified": files,
        "git_summary": final_state.get("git_summary", ""),
        "pr_url": final_state.get("pr_url", ""),
        "summary": (
            final_state.get("implementation_steps")
            or final_state.get("development_plan")
            or ""
        )[:1000].strip(),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "duration_seconds": duration_seconds,
        "resumed_from": resumed_from,
    }


def _files_from_git(workspace: str, git_summary: str) -> list[str]:
    """Try to get modified file list from the most recent git commit."""
    if not git_summary:
        return []
    try:
        import subprocess
        r = subprocess.run(
            ["git", "log", "--name-only", "--pretty=format:", "-1"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return [f.strip() for f in r.stdout.strip().splitlines() if f.strip()]
    except Exception:
        pass
    return []


def build_resume_context(session: dict) -> str:
    """Build the context block prepended to a resumed run's goal."""
    lines = [
        "=== CONTEXT FROM PREVIOUS SESSION ===",
        f"Session : {session.get('id', '?')}",
        f"Date    : {session.get('created_at', '?')[:19]}",
        f"Status  : {session.get('status', '?')}",
        f"Goal    : {session.get('goal', '?')}",
    ]
    files = session.get("files_modified") or []
    if files:
        lines.append(f"Modified: {', '.join(files[:20])}")
    git = session.get("git_summary", "")
    if git:
        lines.append(f"Git     : {git}")
    summary = (session.get("summary") or "").strip()
    if summary:
        lines.append(f"\nWhat was done:\n{summary[:600]}")
    lines.append("=" * 38)
    return "\n".join(lines)
