"""MCP client: expose external MCP servers' tools inside the executor.

Wells already *is* an MCP server (:mod:`mcp_server`); this is the other
direction — connecting out to stdio MCP servers (databases, docs, ticket
systems, memory banks) and registering their tools so the agent can call them
like any built-in tool.

Configuration (first match wins):
  * ``MCP_SERVERS`` env var — JSON: ``{"fetch": {"command": "uvx",
    "args": ["mcp-server-fetch"]}, ...}``
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


def load_config() -> dict:
    """Read the MCP server config (env first, then ~/.wells/mcp.json)."""
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
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


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

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=spec["command"],
            args=list(spec.get("args") or []),
            env={**os.environ, **(spec.get("env") or {})},
            cwd=spec.get("cwd"),
        )
        stack = AsyncExitStack()
        read, write = await stack.enter_async_context(stdio_client(params))
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
    from coding_harness import safety
    from coding_harness.tools import ToolDef, ToolResult

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


def register_mcp_tools() -> list[str]:
    """Connect configured servers and register their tools. Returns tool names.

    Safe to call multiple times (already-registered names are skipped) and
    safe to call with no config (returns []). Failures are per-server and
    non-fatal.
    """
    cfg = load_config()
    if not cfg:
        return []

    from coding_harness import tools as tools_mod

    bridge = _get_bridge()
    registered: list[str] = []
    for name, spec in cfg.items():
        if not isinstance(spec, dict) or not spec.get("command"):
            print(f"[mcp] skipping '{name}': no command configured")
            continue
        if name in bridge.sessions:
            continue  # already connected
        try:
            session = bridge.open_session(name, spec)
            listing = bridge.call(session.list_tools(), _CONNECT_TIMEOUT)
        except Exception as e:
            print(f"[mcp] could not connect '{name}': {type(e).__name__}: {e}")
            continue
        defs = [_wrap_tool(bridge, session, name, t) for t in listing.tools]
        tools_mod.register_external(defs)
        registered.extend(d.name for d in defs)
    return registered
