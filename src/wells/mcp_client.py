"""MCP client: expose external MCP servers' tools inside the executor.

Wells already *is* an MCP server (:mod:`mcp_server`); this is the other
direction — connecting out to MCP servers (databases, docs, ticket systems,
memory banks) and registering their tools so the agent can call them like
any built-in tool.

Two transports are supported:

  * **stdio** — local subprocess. Spec: ``{"command": "uvx",
    "args": ["mcp-server-fetch"]}``. Existing; works everywhere.
  * **HTTP/SSE** — remote server. Spec: ``{"url": "https://...",
    "transport": "http" | "sse", "headers": {...}}``. ``transport`` is
    optional (defaults to ``"http"`` — the newer MCP streamable-http spec;
    set ``"sse"`` explicitly for servers that only speak the legacy SSE
    protocol). ``headers`` is optional and typically carries auth.

Configuration (first match wins):
  * ``MCP_SERVERS`` env var — JSON: ``{"fetch": {"command": "uvx",
    "args": ["mcp-server-fetch"]}, "remote": {"url": "https://...",
    "headers": {"Authorization": "Bearer ..."}}, ...}``
  * ``~/.wells/mcp.json`` — same JSON shape.

Design:
  * The MCP SDK is async; the executor is sync. A dedicated background thread
    owns an asyncio loop, sessions live on that loop, and tool calls hop over
    with ``run_coroutine_threadsafe``.
  * Tools are registered as ``mcp_<server>_<tool>``. Every call passes the
    safety gate (approve/dryrun policies apply — external tools can have side
    effects the workspace confinement can't see).
  * Connection failures are logged and non-fatal: Wells runs fine without.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import threading
from pathlib import Path

_CONFIG_FILE = Path.home() / ".wells" / "mcp.json"
_CONNECT_TIMEOUT = 25
_CALL_TIMEOUT = 120

_BRIDGE: "_Bridge | None" = None
_LOCK = threading.Lock()


_TEMPLATE = {
    "_readme": [
        "Wells MCP client configuration. Servers listed at the top level are",
        "connected on startup and their tools appear to the agent as",
        "mcp_<server>_<tool>. Keys starting with '_' are ignored — move an",
        "entry from _examples to the top level (and fill in paths/keys) to",
        "enable it. The MCP_SERVERS environment variable overrides this file.",
    ],
    "_examples": {
        "fetch": {
            "command": "uvx",
            "args": ["mcp-server-fetch"],
        },
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem",
                     "C:/path/to/allowed/dir"],
        },
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"},
        },
        "postgres": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-postgres",
                     "postgresql://user:pass@localhost/db"],
        },
        "sqlite": {
            "command": "uvx",
            "args": ["mcp-server-sqlite", "--db-path", "C:/path/to/data.db"],
        },
        "memory": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"],
        },
        # Remote (HTTP / SSE) servers — no local subprocess; connect over the
        # network. `transport` defaults to "http" (the newer MCP streamable
        # spec); set it to "sse" explicitly for servers that only speak the
        # legacy SSE protocol. `headers` is optional and typically carries auth.
        "remote-http": {
            "url": "https://mcp.example.com/mcp",
            "headers": {"Authorization": "Bearer YOUR_TOKEN"},
        },
        "remote-sse": {
            "url": "https://mcp-legacy.example.com/sse",
            "transport": "sse",
        },
    },
}


def ensure_template() -> None:
    """Create ~/.wells/mcp.json with documented samples if it doesn't exist."""
    try:
        if _CONFIG_FILE.exists():
            return
        _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_FILE.write_text(
            json.dumps(_TEMPLATE, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        pass


def load_config() -> dict:
    """Read the MCP server config (env first, then ~/.wells/mcp.json).

    Keys starting with ``_`` (docs/samples in the template) and non-dict
    values are ignored.
    """
    raw = os.environ.get("MCP_SERVERS", "").strip()
    if not raw and _CONFIG_FILE.exists():
        try:
            raw = _CONFIG_FILE.read_text(encoding="utf-8")
        except Exception:
            raw = ""
    if not raw:
        return {}
    try:
        cfg = json.loads(raw)
        if not isinstance(cfg, dict):
            return {}
        return {
            k: v for k, v in cfg.items()
            if not k.startswith("_") and isinstance(v, dict)
        }
    except Exception:
        return {}


def _transport_kind(spec: dict) -> str:
    """Classify a server spec: 'stdio' | 'http' | 'sse'.

    - ``{"command": ...}`` → stdio
    - ``{"url": ..., "transport": "sse"|"http"}`` → that transport
    - ``{"url": ...}`` with no explicit transport → 'http' (the newer MCP
      streamable-http spec; SSE is the legacy variant)
    """
    if spec.get("url"):
        t = (spec.get("transport") or "http").strip().lower()
        return t if t in ("http", "sse") else "http"
    return "stdio"


def _validate_spec(spec: dict) -> tuple[bool, str]:
    """Check that a spec is well-formed. Returns (ok, reason)."""
    if not isinstance(spec, dict):
        return False, "spec must be a dict"
    if spec.get("url"):
        if not isinstance(spec["url"], str) or not spec["url"].startswith(("http://", "https://")):
            return False, f"url must be a http(s) URL (got {spec['url']!r})"
        t = (spec.get("transport") or "http").strip().lower()
        if t not in ("", "http", "sse"):
            return False, f"transport must be 'http' or 'sse' (got {t!r})"
        return True, ""
    if spec.get("command"):
        if not isinstance(spec["command"], str):
            return False, "command must be a string"
        return True, ""
    return False, "spec needs either 'command' (stdio) or 'url' (http/sse)"


class _Bridge:
    """Background thread that owns the asyncio loop all MCP sessions live on."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.sessions: dict[str, tuple] = {}  # name -> (session, AsyncExitStack)
        self._thread = threading.Thread(
            target=self._run, name="wells-mcp", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, coro, timeout: float):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def _open(self, name: str, spec: dict):
        from contextlib import AsyncExitStack

        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        stack = AsyncExitStack()
        kind = _transport_kind(spec)
        if kind == "stdio":
            from mcp import StdioServerParameters

            params = StdioServerParameters(
                command=spec["command"],
                args=list(spec.get("args") or []),
                env={**os.environ, **(spec.get("env") or {})},
                cwd=spec.get("cwd"),
            )
            read, write = await stack.enter_async_context(stdio_client(params))
        elif kind == "sse":
            from mcp.client.sse import sse_client

            read, write = await stack.enter_async_context(
                sse_client(
                    spec["url"],
                    headers=spec.get("headers") or None,
                )
            )
        else:  # http (streamable-http)
            from mcp.client.streamable_http import streamablehttp_client

            ctx = streamablehttp_client(
                spec["url"],
                headers=spec.get("headers") or None,
            )
            # streamablehttp_client yields (read, write, get_session_id) —
            # we don't need the third element for tool calls.
            read, write, *_ = await stack.enter_async_context(ctx)

        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self.sessions[name] = (session, stack)
        return session

    def open_session(self, name: str, spec: dict):
        return self.call(self._open(name, spec), _CONNECT_TIMEOUT)

    def shutdown(self) -> None:
        for name, (_, stack) in list(self.sessions.items()):
            try:
                self.call(stack.aclose(), 10)
            except Exception:
                pass
        self.sessions.clear()
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass


def _get_bridge() -> _Bridge:
    global _BRIDGE
    with _LOCK:
        if _BRIDGE is None:
            _BRIDGE = _Bridge()
            atexit.register(_BRIDGE.shutdown)
        return _BRIDGE


def _wrap_tool(bridge: _Bridge, session, server: str, mcp_tool):
    """Build a Wells ToolDef that proxies one MCP tool call."""
    from wells import safety
    from wells.tools import ToolDef, ToolResult

    remote_name = mcp_tool.name
    local_name = f"mcp_{server}_{remote_name}"[:64]
    schema = getattr(mcp_tool, "inputSchema", None) or {
        "type": "object", "properties": {},
    }

    def handler(ctx, **kwargs) -> ToolResult:
        # External tools can have side effects confinement can't see — always
        # pass the safety gate so approve/dryrun policies apply.
        decision = safety.gate(
            f"mcp:{server}.{remote_name}",
            json.dumps(kwargs)[:160],
            safety=ctx.safety,
            approver=ctx.approver,
        )
        if not decision.allowed:
            return ToolResult(True, decision.reason, simulated=decision.simulated)
        try:
            res = bridge.call(session.call_tool(remote_name, kwargs), _CALL_TIMEOUT)
        except Exception as e:
            return ToolResult(False, "", f"MCP call failed: {type(e).__name__}: {e}")
        parts = []
        for c in getattr(res, "content", None) or []:
            text = getattr(c, "text", None)
            parts.append(text if text is not None else str(c))
        body = "\n".join(parts) or "(no content)"
        is_err = bool(getattr(res, "isError", False))
        return ToolResult(not is_err, body, body[:200] if is_err else "")

    desc = (getattr(mcp_tool, "description", "") or remote_name).strip()
    return ToolDef(
        name=local_name,
        description=f"[MCP:{server}] {desc[:300]}",
        input_schema=schema,
        handler=handler,
        mutating=False,  # gate is enforced inside the handler instead
    )


# Tool names registered per connected server (for disconnect/unregister).
_REGISTERED: dict[str, list[str]] = {}


def connect_server(name: str, spec: dict) -> tuple[bool, str, list[str]]:
    """Connect one server and register its tools.

    Returns (ok, message, registered_tool_names).

    Accepts either stdio servers (``{"command": ...,
    "args": [...]}``) or remote HTTP/SSE servers (``{"url": "...",
    "transport": "http"|"sse", "headers": {...}}``). See
    :func:`_validate_spec` for the exact shape.
    """
    ok, reason = _validate_spec(spec)
    if not ok:
        return False, reason, []

    from wells import tools as tools_mod

    bridge = _get_bridge()
    if name in bridge.sessions:
        return True, "already connected", list(_REGISTERED.get(name, []))
    try:
        session = bridge.open_session(name, spec)
        listing = bridge.call(session.list_tools(), _CONNECT_TIMEOUT)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", []
    defs = [_wrap_tool(bridge, session, name, t) for t in listing.tools]
    tools_mod.register_external(defs)
    names = [d.name for d in defs]
    _REGISTERED[name] = names
    kind = _transport_kind(spec)
    suffix = "" if kind == "stdio" else f" ({kind})"
    return True, f"connected{suffix} ({len(names)} tool(s))", names


def disconnect_server(name: str) -> bool:
    """Close a server's session and unregister its tools. True if it was live."""
    from wells import tools as tools_mod

    bridge = _BRIDGE
    was_live = False
    if bridge is not None and name in bridge.sessions:
        _, stack = bridge.sessions.pop(name)
        try:
            bridge.call(stack.aclose(), 10)
        except Exception:
            pass
        was_live = True
    tools_mod.unregister_external(_REGISTERED.pop(name, []))
    return was_live


def connected() -> dict[str, list[str]]:
    """Currently connected servers -> their registered tool names."""
    return {k: list(v) for k, v in _REGISTERED.items()}


def register_mcp_tools() -> list[str]:
    """Connect configured servers and register their tools. Returns tool names.

    Safe to call multiple times (already-registered names are skipped) and
    safe to call with no config (returns []). Failures are per-server and
    non-fatal.
    """
    cfg = load_config()
    if not cfg:
        return []

    registered: list[str] = []
    for name, spec in cfg.items():
        ok, msg, names = connect_server(name, spec)
        if not ok:
            print(f"[mcp] could not connect '{name}': {msg}")
        registered.extend(names)
    return registered


# ---------------------------------------------------------------------------
# File-config CRUD (backs the /mcp command)
#
# Layout of ~/.wells/mcp.json: active servers at the top level, disabled ones
# under "_disabled", starter samples under "_examples". The MCP_SERVERS env
# var, when set, overrides the file entirely (load_config reads env first).
# ---------------------------------------------------------------------------


def env_override_active() -> bool:
    return bool(os.environ.get("MCP_SERVERS", "").strip())


def read_file_config() -> dict:
    """Full raw content of mcp.json (creating the template if missing)."""
    ensure_template()
    try:
        data = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_file_config(data: dict) -> None:
    _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def add_server(name: str, spec: dict) -> None:
    data = read_file_config()
    data[name] = spec
    data.get("_disabled", {}).pop(name, None)
    write_file_config(data)


def remove_server(name: str) -> dict | None:
    """Remove a server (active or disabled). Returns the removed spec."""
    data = read_file_config()
    spec = data.pop(name, None)
    if spec is None:
        spec = (data.get("_disabled") or {}).pop(name, None)
    if spec is not None:
        write_file_config(data)
    return spec


def set_enabled(name: str, enabled: bool) -> tuple[bool, str]:
    """Move a server between the active top level and _disabled/_examples."""
    data = read_file_config()
    disabled = data.setdefault("_disabled", {})
    examples = data.get("_examples") or {}

    if enabled:
        spec = disabled.pop(name, None) or examples.get(name)
        if spec is None:
            return False, f"'{name}' not found in _disabled or _examples"
        data[name] = spec
        write_file_config(data)
        return True, "enabled"

    spec = data.pop(name, None)
    if spec is None:
        return False, f"'{name}' is not an active server"
    disabled[name] = spec
    write_file_config(data)
    return True, "disabled"
