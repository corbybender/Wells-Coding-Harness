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

try:
    from wells_index import IndexEngine
    INDEXER_AVAILABLE = True
except ImportError:
    INDEXER_AVAILABLE = False
    IndexEngine = None  # type: ignore


from typing import Any, Dict, Optional
from coding_harness.tools import ToolContext, ToolDef, ToolResult
import os

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
        except Exception as e:
            return None

    return _engine_cache[workspace]


def _clear_cache() -> None:
    """Clear the engine cache (used after index updates)."""
    _engine_cache.clear()


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
            description="Find definition of a symbol by exact name (faster and more precise than grep for symbol lookups)",
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
            description="Prefix/substring search across all indexed symbol names",
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
