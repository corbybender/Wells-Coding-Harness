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


def build_repo_map(workspace: str, *, max_chars: int = 6000) -> str:
    """Return the compressed repo map for ``workspace`` (cached, TTL 5 min)."""
    cached = _CACHE.get(workspace)
    if cached and time.time() - cached[0] < _TTL_SECONDS:
        return cached[1]

    root = Path(workspace)
    files = _source_files(root)
    if not files:
        _CACHE[workspace] = (time.time(), "")
        return ""

    lines: list[str] = []
    total = 0
    for i, f in enumerate(files):
        try:
            rel = f.relative_to(root).as_posix()
        except ValueError:
            continue
        entry = rel
        if i < _MAX_SYMBOL_FILES:
            syms = _symbols_for(workspace, rel)
            if syms:
                entry = f"{rel}: {', '.join(syms)}"
        total += len(entry) + 1
        if total > max_chars:
            lines.append(f"… ({len(files) - i} more files)")
            break
        lines.append(entry)

    repo_map = "\n".join(lines)
    _CACHE[workspace] = (time.time(), repo_map)
    return repo_map


def repo_map_block(workspace: str) -> str:
    """Prompt-ready block, or empty string when no map could be built."""
    m = build_repo_map(workspace)
    if not m:
        return ""
    return (
        "\nREPO MAP (source files → key symbols; (c)=class, (f)=function — "
        "use find_symbol for exact locations):\n" + m + "\n"
    )


def invalidate(workspace: str | None = None) -> None:
    if workspace is None:
        _CACHE.clear()
    else:
        _CACHE.pop(workspace, None)
