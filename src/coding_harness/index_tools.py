"""Tools for structural repository indexing via the wells-index Rust engine.

When wells-index is available, these tools provide symbol-level code retrieval:
- find_symbol: locate definitions by name
- find_references: locate all references to a symbol
- find_callers: locate all call sites for a function
- search_symbols: prefix/substring search
- list_symbols: enumerate symbols in a file

When wells-index is NOT installed, INDEX_TOOLS is an empty list and the harness
gracefully falls back to grep/glob for code search.
"""

def _try_load_indexer():
    """Try to load indexer; if not available, try to build it."""
    try:
        from wells_index import IndexEngine
        return True, IndexEngine
    except ImportError:
        # Try to build indexer on demand
        try:
            from coding_harness import setup as _setup_module
            if _setup_module._ensure_indexer_built():
                from wells_index import IndexEngine
                return True, IndexEngine
        except Exception:
            pass
    return False, None


INDEXER_AVAILABLE, IndexEngine = _try_load_indexer()


from typing import Any, Dict, Optional
from coding_harness.tools import ToolContext, ToolDef, ToolResult

# Module-level cache of engines per workspace
_engine_cache: Dict[str, Optional[Any]] = {}


def _get_engine(ctx: ToolContext) -> Optional[Any]:
    """Return or create cached IndexEngine for the given workspace."""
    if not INDEXER_AVAILABLE:
        return None

    workspace = str(ctx.workspace)
    if workspace not in _engine_cache:
        try:
            _engine_cache[workspace] = IndexEngine(workspace)
        except Exception:
            return None

    return _engine_cache[workspace]


def _clear_cache() -> None:
    """Clear the engine cache (used after index updates)."""
    _engine_cache.clear()


# ---------------------------------------------------------------------------
# Index status: presence + age checks (for auto-detection on startup)
# ---------------------------------------------------------------------------

def index_status(workspace: str) -> dict:
    """Check whether an index exists for ``workspace`` and how stale it is.

    Returns a dict with:
      - ``available`` (bool): whether the wells-index engine is installed
      - ``exists`` (bool): whether an index has been built for this workspace
      - ``total_symbols`` (int): symbol count (0 if no index)
      - ``total_files`` (int): file count (0 if no index)
      - ``last_indexed_at`` (str|None): timestamp of last index build
      - ``age_hours`` (float|None): hours since last index build
      - ``error`` (str|None): error message if the check failed
    """
    result: dict = {
        "available": INDEXER_AVAILABLE,
        "exists": False,
        "total_symbols": 0,
        "total_files": 0,
        "last_indexed_at": None,
        "age_hours": None,
        "error": None,
    }
    if not INDEXER_AVAILABLE:
        return result
    try:
        engine = IndexEngine(str(workspace))
        stats = engine.stats()
        result["exists"] = (stats.get("total_symbols", 0) > 0
                            or stats.get("total_files", 0) > 0)
        result["total_symbols"] = stats.get("total_symbols", 0)
        result["total_files"] = stats.get("total_files", 0)
        last = stats.get("last_indexed_at")
        result["last_indexed_at"] = last
        if last:
            result["age_hours"] = _index_age_hours(last)
    except Exception as e:
        result["error"] = str(e)
    return result


def _index_age_hours(last_indexed_at) -> float | None:
    """Parse a ``last_indexed_at`` value (timestamp/datetime/str) to hours ago."""
    import datetime as _dt
    try:
        if isinstance(last_indexed_at, (int, float)):
            then = _dt.datetime.fromtimestamp(float(last_indexed_at), tz=_dt.timezone.utc)
        elif isinstance(last_indexed_at, _dt.datetime):
            then = last_indexed_at
            if then.tzinfo is None:
                then = then.replace(tzinfo=_dt.timezone.utc)
        else:
            # ISO string or other; try fromisoformat, fall back to timestamp.
            s = str(last_indexed_at).strip()
            try:
                then = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            except ValueError:
                then = _dt.datetime.fromtimestamp(float(s), tz=_dt.timezone.utc)
        now = _dt.datetime.now(tz=_dt.timezone.utc)
        return max(0.0, (now - then).total_seconds() / 3600.0)
    except Exception:
        return None


def ensure_index(workspace: str, *, max_age_hours: float = 24.0,
                 auto_build: bool = True, prompt_fn=None) -> str:
    """Ensure the repo index exists and is fresh; build/update if needed.

    This is the single entry point for auto-indexing. It:
      1. Checks index presence. If missing and ``auto_build`` is True, builds it
         immediately. If ``auto_build`` is False, calls ``prompt_fn`` to ask the
         user (default: prints a message and skips).
      2. Checks index age. If older than ``max_age_hours`` and ``auto_build`` is
         True, updates silently. If not auto, calls ``prompt_fn``.
      3. Returns a short status string describing what happened.

    ``prompt_fn(message)`` should return True to proceed, False to skip. When
    ``prompt_fn`` is None a non-interactive default is used (auto-build if
    ``auto_build``, else skip with a message).
    """
    if not INDEXER_AVAILABLE:
        return "index-unavailable"

    status = index_status(workspace)

    if status.get("error"):
        return f"index-check-failed: {status['error']}"

    # Case 1: no index at all.
    if not status["exists"]:
        if auto_build:
            return _do_index(workspace, verb="Building")
        if prompt_fn and prompt_fn(
            "No repository index found. Build one now? (recommended) [Y/n] "
        ):
            return _do_index(workspace, verb="Building")
        return "index-skipped-missing"

    # Case 2: index exists but may be stale.
    age = status.get("age_hours")
    if age is not None and age > max_age_hours:
        if auto_build:
            return _do_index(workspace, verb=f"Updating (index {age:.0f}h old)")
        if prompt_fn and prompt_fn(
            f"Repository index is {age:.0f} hours old. Update it? [Y/n] "
        ):
            return _do_index(workspace, verb="Updating")
        return f"index-skipped-stale ({age:.0f}h old)"

    return f"index-ready ({status['total_symbols']:,} symbols, {status['total_files']:,} files)"


def _do_index(workspace: str, *, verb: str) -> str:
    """Run the index build/update and return a status string."""
    print(f"[index] {verb} repository index ...")
    try:
        ctx = ToolContext(workspace=str(workspace))
        result = index_workspace(ctx)
        if result.ok:
            return f"index-built: {result.output.strip()}"
        return f"index-failed: {result.error or result.output}"
    except Exception as e:
        return f"index-failed: {e}"


def index_workspace(ctx: ToolContext) -> ToolResult:
    """Build or incrementally update the structural index for the workspace.

    Returns a summary of indexed files and extracted symbols.
    Runs transparently in the background; only re-parses changed files.
    """
    if not INDEXER_AVAILABLE:
        return ToolResult(
            ok=False,
            output="Index engine not available. Install wells-index: pip install wells-index",
        )

    try:
        engine = IndexEngine(str(ctx.workspace))
        stats = engine.index()

        # Clear cache to ensure fresh state
        _clear_cache()

        output = f"""Indexed repository:
- Files processed: {stats['files_indexed']}
- Symbols extracted: {stats['symbols_extracted']}
- Edges extracted: {stats['edges_extracted']}
- Duration: {stats['duration_ms']}ms
"""
        return ToolResult(ok=True, output=output)
    except Exception as e:
        return ToolResult(ok=False, output=f"Indexing failed: {e}")


def find_symbol(ctx: ToolContext, name: str) -> ToolResult:
    """Find definition location(s) for a symbol by exact name.

    Returns all files and line numbers where the symbol is defined.
    Much faster and more precise than grep for symbol lookups.
    """
    if not INDEXER_AVAILABLE:
        return ToolResult(ok=False, output="Index engine not available")

    try:
        engine = _get_engine(ctx)
        if engine is None:
            return ToolResult(ok=False, output="Failed to initialize index engine")

        results = engine.find_symbol(name)
        if not results:
            return ToolResult(ok=True, output=f"No definitions found for '{name}'")

        lines = [f"Definitions of '{name}':"]
        for r in results:
            lines.append(
                f"  {r['file_path']}:{r['start_line']}-{r['end_line']} ({r['kind']})"
            )

        return ToolResult(ok=True, output="\n".join(lines))
    except Exception as e:
        return ToolResult(ok=False, output=f"Query failed: {e}")


def find_references(ctx: ToolContext, symbol: str) -> ToolResult:
    """Find all files and lines that reference or call a symbol.

    Includes direct references, function calls, and inheritance relationships.
    """
    if not INDEXER_AVAILABLE:
        return ToolResult(ok=False, output="Index engine not available")

    try:
        engine = _get_engine(ctx)
        if engine is None:
            return ToolResult(ok=False, output="Failed to initialize index engine")

        results = engine.find_references(symbol)
        if not results:
            return ToolResult(ok=True, output=f"No references found for '{symbol}'")

        lines = [f"References to '{symbol}':"]
        for r in results:
            lines.append(
                f"  {r['file_path']}:{r['start_line']} ({r['kind']})"
            )

        return ToolResult(ok=True, output="\n".join(lines))
    except Exception as e:
        return ToolResult(ok=False, output=f"Query failed: {e}")


def find_callers(ctx: ToolContext, symbol: str) -> ToolResult:
    """Find all functions or methods that call a given function.

    Useful for understanding how a function is used across the codebase.
    """
    if not INDEXER_AVAILABLE:
        return ToolResult(ok=False, output="Index engine not available")

    try:
        engine = _get_engine(ctx)
        if engine is None:
            return ToolResult(ok=False, output="Failed to initialize index engine")

        results = engine.find_callers(symbol)
        if not results:
            return ToolResult(ok=True, output=f"No callers found for '{symbol}'")

        lines = [f"Callers of '{symbol}':"]
        for r in results:
            lines.append(
                f"  {r['file_path']}:{r['start_line']} in {r['name']}"
            )

        return ToolResult(ok=True, output="\n".join(lines))
    except Exception as e:
        return ToolResult(ok=False, output=f"Query failed: {e}")


def search_symbols(ctx: ToolContext, query: str, limit: int = 20) -> ToolResult:
    """Search for symbols by prefix or substring.

    Returns up to `limit` matching symbol names and their locations.
    """
    if not INDEXER_AVAILABLE:
        return ToolResult(ok=False, output="Index engine not available")

    try:
        engine = _get_engine(ctx)
        if engine is None:
            return ToolResult(ok=False, output="Failed to initialize index engine")

        results = engine.search_symbols(query, limit)
        if not results:
            return ToolResult(ok=True, output=f"No symbols matching '{query}'")

        lines = [f"Symbols matching '{query}':"]
        for r in results:
            lines.append(
                f"  {r['name']} ({r['kind']}) at {r['file_path']}:{r['start_line']}"
            )

        return ToolResult(ok=True, output="\n".join(lines))
    except Exception as e:
        return ToolResult(ok=False, output=f"Search failed: {e}")


def list_symbols(ctx: ToolContext, path: str = "") -> ToolResult:
    """List all symbols defined in a file, or show summary statistics.

    If path is empty, returns total symbol count by kind across the repo.
    If path is provided, returns all symbols in that file.
    """
    if not INDEXER_AVAILABLE:
        return ToolResult(ok=False, output="Index engine not available")

    try:
        engine = _get_engine(ctx)
        if engine is None:
            return ToolResult(ok=False, output="Failed to initialize index engine")

        if path:
            results = engine.list_in_file(path)
            if not results:
                return ToolResult(ok=True, output=f"No symbols found in '{path}'")

            lines = [f"Symbols in {path}:"]
            for r in results:
                lines.append(
                    f"  {r['name']} ({r['kind']}) at line {r['start_line']}"
                )
            return ToolResult(ok=True, output="\n".join(lines))
        else:
            # Return repo-wide statistics
            stats = engine.stats()
            output = f"""Index Statistics:
- Total files: {stats['total_files']}
- Total symbols: {stats['total_symbols']}
- Total edges: {stats['total_edges']}
- Last indexed: {stats['last_indexed_at']}
"""
            return ToolResult(ok=True, output=output)
    except Exception as e:
        return ToolResult(ok=False, output=f"Query failed: {e}")


# Tool definitions — only populated if indexer is available
INDEX_TOOLS: list[ToolDef] = []

if INDEXER_AVAILABLE:
    INDEX_TOOLS = [
        ToolDef(
            name="find_symbol",
            description="PREFERRED over grep/read: instantly locate where a function, class, or variable is defined. Returns file path and line number. Use this first whenever you need to find any named symbol.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The exact symbol name to find (class, function, method, variable, module)",
                    }
                },
                "required": ["name"],
            },
            handler=lambda ctx, name: find_symbol(ctx, name),
            mutating=False,
        ),
        ToolDef(
            name="find_references",
            description="Find all files and lines that reference, call, or inherit from a symbol",
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The symbol name to find references for",
                    }
                },
                "required": ["symbol"],
            },
            handler=lambda ctx, symbol: find_references(ctx, symbol),
            mutating=False,
        ),
        ToolDef(
            name="find_callers",
            description="Find all functions and methods that call a given function",
            input_schema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The function/method name to find callers for",
                    }
                },
                "required": ["symbol"],
            },
            handler=lambda ctx, symbol: find_callers(ctx, symbol),
            mutating=False,
        ),
        ToolDef(
            name="search_symbols",
            description="PREFERRED over grep: search for symbols by name prefix or substring across the whole repo. Use when you don't know the exact name. Returns file:line for each match.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The prefix or substring to search for",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default 20)",
                        "default": 20,
                    },
                },
                "required": ["query"],
            },
            handler=lambda ctx, query, limit=20: search_symbols(ctx, query, limit),
            mutating=False,
        ),
        ToolDef(
            name="list_symbols",
            description="List all symbols in a file, or show repository-wide symbol statistics",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace. If empty, show repo stats.",
                    }
                },
            },
            handler=lambda ctx, path="": list_symbols(ctx, path),
            mutating=False,
        ),
    ]
