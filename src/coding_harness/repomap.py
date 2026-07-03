"""Compressed repo map injected into planner/coder prompts.

The single biggest quality lever for plan-first agents (aider's key trick):
start the model with a map of where things live — directory tree plus the key
symbols per file — so it plans from knowledge instead of spending tool steps
on discovery.

Sources, best-first:
  * wells-index (when available): real symbol names/kinds per file.
  * Filesystem walk fallback: tree only.

The map is cached per workspace with a short TTL; building it costs one
filesystem walk plus fast per-file index queries.
"""

from __future__ import annotations

import time
from pathlib import Path

_SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "node_modules", "target", "dist", "build",
    ".wells_index", ".idea", ".vscode", "wheels", ".pytest_cache", ".mypy_cache",
}
_SOURCE_EXTS = {
    ".py", ".rs", ".go", ".ts", ".tsx", ".js", ".jsx", ".java", ".cs", ".rb",
    ".php", ".c", ".h", ".cpp", ".hpp", ".swift", ".kt", ".scala", ".sql",
}
_MAX_FILES = 250          # stop walking after this many source files
_MAX_SYMBOL_FILES = 80    # query symbols for at most this many files
_MAX_SYMBOLS_PER_FILE = 12
_TTL_SECONDS = 300

_CACHE: dict[str, tuple[float, str]] = {}


def _source_files(root: Path) -> list[Path]:
    out: list[Path] = []
    stack = [root]
    while stack and len(out) < _MAX_FILES:
        d = stack.pop()
        try:
            entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except Exception:
            continue
        for e in entries:
            if e.is_dir():
                if e.name not in _SKIP_DIRS and not e.name.startswith("."):
                    stack.append(e)
            elif e.suffix.lower() in _SOURCE_EXTS:
                out.append(e)
                if len(out) >= _MAX_FILES:
                    break
    return sorted(out)


def _symbols_for(workspace: str, rel_path: str) -> list[str]:
    """Key symbol names for one file via wells-index; [] when unavailable."""
    try:
        from coding_harness import index_tools

        if not index_tools.INDEXER_AVAILABLE:
            return []
        from coding_harness.tools import ToolContext

        engine = index_tools._get_engine(ToolContext(workspace=workspace))
        if engine is None:
            return []
        # The index stores OS-native separators; try native first, then posix.
        import os
        results = engine.list_in_file(rel_path.replace("/", os.sep)) or []
        if not results and os.sep != "/":
            results = engine.list_in_file(rel_path) or []
        # Prefer top-level definitions: classes and functions first.
        keyed = sorted(
            results,
            key=lambda r: (0 if r.get("kind") in ("class", "struct", "trait") else 1,
                           r.get("start_line", 0)),
        )
        names = []
        for r in keyed[:_MAX_SYMBOLS_PER_FILE]:
            kind = (r.get("kind") or "")[:1]  # c/f/m…
            names.append(f"{r.get('name', '?')}({kind})" if kind else r.get("name", "?"))
        return names
    except Exception:
        return []


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "when", "add",
    "fix", "make", "use", "all", "new", "our", "you", "not", "are", "can",
    "file", "files", "code", "should", "please",
}


def _goal_keywords(goal: str) -> set[str]:
    import re
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", (goal or "").lower())
    return {w for w in words if w not in _STOPWORDS}


def _score(rel: str, syms: list[str], keywords: set[str]) -> float:
    """Relevance score: goal-keyword mentions dominate, then symbol richness."""
    score = 0.0
    rel_l = rel.lower()
    for kw in keywords:
        if kw in rel_l:
            score += 6.0
        score += 4.0 * sum(1 for s in syms if kw in s.lower())
    score += min(len(syms), 15) * 0.5          # symbol-rich files matter more
    score -= rel.count("/") * 1.0              # prefer shallow paths
    if "test" in rel_l and not any("test" in k for k in keywords):
        score -= 3.0                           # tests are rarely the target
    return score


def build_repo_map(workspace: str, *, goal: str = "", max_chars: int = 6000) -> str:
    """Return the compressed repo map, most-relevant files first.

    With a ``goal``, files are ranked by goal-keyword mentions (path + symbol
    names) so the truncation cuts the *least* relevant files — on big repos the
    map stays useful instead of alphabetical-prefix trivia. Cached per
    (workspace, goal keywords) with a short TTL.
    """
    keywords = _goal_keywords(goal)
    cache_key = f"{workspace}|{'.'.join(sorted(keywords))}"
    cached = _CACHE.get(cache_key)
    if cached and time.time() - cached[0] < _TTL_SECONDS:
        return cached[1]

    root = Path(workspace)
    files = _source_files(root)
    if not files:
        _CACHE[cache_key] = (time.time(), "")
        return ""

    entries: list[tuple[float, str, str]] = []  # (score, rel, entry_text)
    for i, f in enumerate(files):
        try:
            rel = f.relative_to(root).as_posix()
        except ValueError:
            continue
        syms = _symbols_for(workspace, rel) if i < _MAX_SYMBOL_FILES else []
        entry = f"{rel}: {', '.join(syms)}" if syms else rel
        entries.append((_score(rel, syms, keywords), rel, entry))

    # Most relevant first; ties resolve alphabetically for stable output.
    entries.sort(key=lambda t: (-t[0], t[1]))

    lines: list[str] = []
    total = 0
    for i, (_, _, entry) in enumerate(entries):
        total += len(entry) + 1
        if total > max_chars:
            lines.append(f"… ({len(entries) - i} more files)")
            break
        lines.append(entry)

    repo_map = "\n".join(lines)
    _CACHE[cache_key] = (time.time(), repo_map)
    return repo_map


def repo_map_block(workspace: str, goal: str = "") -> str:
    """Prompt-ready block, or empty string when no map could be built."""
    m = build_repo_map(workspace, goal=goal)
    if not m:
        return ""
    return (
        "\nREPO MAP (most relevant source files first → key symbols; "
        "(c)=class, (f)=function — use find_symbol for exact locations):\n"
        + m + "\n"
    )


def invalidate(workspace: str | None = None) -> None:
    if workspace is None:
        _CACHE.clear()
    else:
        _CACHE.pop(workspace, None)
