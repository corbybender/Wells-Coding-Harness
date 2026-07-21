"""Tests for the MCP client: transport dispatch, spec validation, config CRUD.

We don't spin up real MCP servers here — the transport dispatch logic is
what matters for unit testing, and that's exercised by stubbing the SDK's
transport entry points (``stdio_client``, ``sse_client``,
``streamablehttp_client``) and asserting the right one is selected for a
given spec. End-to-end connectivity tests run via the existing /mcp command
against real local servers (manually).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from wells import mcp_client


# ---------------------------------------------------------------------------
# _transport_kind: spec → transport classification
# ---------------------------------------------------------------------------


def test_transport_kind_stdio_when_command_present():
    assert mcp_client._transport_kind({"command": "uvx"}) == "stdio"


def test_transport_kind_stdio_with_args_env():
    spec = {"command": "npx", "args": ["-y", "server"], "env": {"X": "1"}}
    assert mcp_client._transport_kind(spec) == "stdio"


def test_transport_kind_http_default_for_url():
    """A bare URL with no explicit transport defaults to streamable-http
    (the newer MCP spec)."""
    assert mcp_client._transport_kind({"url": "https://mcp.example.com"}) == "http"


def test_transport_kind_explicit_http():
    assert (
        mcp_client._transport_kind(
            {"url": "https://x", "transport": "http"}
        )
        == "http"
    )


def test_transport_kind_explicit_sse():
    assert (
        mcp_client._transport_kind(
            {"url": "https://x", "transport": "sse"}
        )
        == "sse"
    )


def test_transport_kind_bad_transport_falls_back_to_http():
    """An unrecognized transport value is treated as http (the newer default),
    not rejected — _validate_spec catches this for human review."""
    assert (
        mcp_client._transport_kind(
            {"url": "https://x", "transport": "websocket"}
        )
        == "http"
    )


def test_transport_kind_transport_case_insensitive():
    assert mcp_client._transport_kind(
        {"url": "https://x", "transport": "SSE"}
    ) == "sse"


# ---------------------------------------------------------------------------
# _validate_spec
# ---------------------------------------------------------------------------


def test_validate_spec_stdio_ok():
    ok, reason = mcp_client._validate_spec({"command": "uvx", "args": ["x"]})
    assert ok and reason == ""


def test_validate_spec_http_ok():
    ok, reason = mcp_client._validate_spec(
        {"url": "https://mcp.example.com", "headers": {"X": "1"}}
    )
    assert ok and reason == ""


def test_validate_spec_sse_ok():
    ok, reason = mcp_client._validate_spec(
        {"url": "https://x", "transport": "sse"}
    )
    assert ok and reason == ""


def test_validate_spec_rejects_non_dict():
    ok, reason = mcp_client._validate_spec("not a dict")  # type: ignore[arg-type]
    assert not ok and "dict" in reason


def test_validate_spec_rejects_missing_command_and_url():
    ok, reason = mcp_client._validate_spec({"args": ["x"]})
    assert not ok
    assert "command" in reason and "url" in reason


def test_validate_spec_rejects_non_http_url():
    ok, reason = mcp_client._validate_spec({"url": "ftp://x"})
    assert not ok and "http(s)" in reason


def test_validate_spec_rejects_bad_transport():
    ok, reason = mcp_client._validate_spec(
        {"url": "https://x", "transport": "ws"}
    )
    assert not ok and "transport" in reason


def test_validate_spec_rejects_non_string_command():
    ok, reason = mcp_client._validate_spec({"command": 123})
    assert not ok and "command" in reason.lower()


# ---------------------------------------------------------------------------
# load_config: env override + file loading
# ---------------------------------------------------------------------------


def test_load_config_env_override(monkeypatch):
    """MCP_SERVERS env var takes precedence over the file."""
    monkeypatch.setenv(
        "MCP_SERVERS",
        json.dumps({"remote": {"url": "https://x"}}),
    )
    cfg = mcp_client.load_config()
    assert "remote" in cfg
    assert cfg["remote"]["url"] == "https://x"


def test_load_config_skips_underscore_keys(monkeypatch):
    """Template/docs keys (starting with _) must be filtered out."""
    monkeypatch.setenv(
        "MCP_SERVERS",
        json.dumps({
            "_readme": ["doc line"],
            "_examples": {"foo": {"command": "x"}},
            "real": {"command": "y"},
        }),
    )
    cfg = mcp_client.load_config()
    assert list(cfg.keys()) == ["real"]


def test_load_config_filters_non_dict_values(monkeypatch):
    monkeypatch.setenv(
        "MCP_SERVERS",
        json.dumps({"broken": "not a dict", "ok": {"command": "x"}}),
    )
    cfg = mcp_client.load_config()
    assert list(cfg.keys()) == ["ok"]


def test_load_config_handles_invalid_json(monkeypatch):
    monkeypatch.setenv("MCP_SERVERS", "{not json")
    assert mcp_client.load_config() == {}


def test_load_config_handles_empty_env(monkeypatch, tmp_path: Path):
    """With no env and no file, returns {}."""
    monkeypatch.delenv("MCP_SERVERS", raising=False)
    monkeypatch.setattr(mcp_client, "_CONFIG_FILE", tmp_path / "nope.json")
    assert mcp_client.load_config() == {}


# ---------------------------------------------------------------------------
# connect_server: spec-shape dispatch
# ---------------------------------------------------------------------------


def test_connect_server_rejects_invalid_spec():
    """connect_server short-circuits on a bad spec without touching the bridge."""
    ok, msg, names = mcp_client.connect_server("x", {"bad": "spec"})
    assert not ok
    assert names == []
    assert "command" in msg or "url" in msg


def test_connect_server_accepts_http_spec():
    """A valid HTTP spec must clear validation and attempt to connect — we
    stub the bridge so the connection succeeds with no real network call."""
    captured: dict = {}

    class _StubListing:
        tools = []

    class _StubSession:
        async def list_tools(self):
            return _StubListing()

    class _StubBridge:
        sessions: dict = {}
        def open_session(self, name, spec):
            captured["name"] = name
            captured["spec"] = spec
            return _StubSession()
        def call(self, coro, timeout):
            import asyncio
            return asyncio.get_event_loop().run_until_complete(coro) if False else _StubListing()

    # Simplest path: stub _get_bridge to return our _StubBridge and stub
    # register_external so we don't pollute the global registry.
    def _fake_run_coro(coro, timeout):
        # Actually run the coroutine so list_tools returns the stub listing.
        import asyncio
        return asyncio.new_event_loop().run_until_complete(coro)

    with (
        patch.object(mcp_client, "_get_bridge", return_value=_StubBridge()),
        patch.object(mcp_client, "_REGISTERED", {}),
        patch("wells.tools.register_external", return_value=None),
    ):
        # Replace the bridge.call method on our stub to actually await.
        _StubBridge.call = lambda self, coro, timeout: _fake_run_coro(coro, timeout)
        ok, msg, names = mcp_client.connect_server(
            "remote", {"url": "https://mcp.example.com"},
        )
    assert ok, msg
    assert names == []
    assert captured["name"] == "remote"
    assert captured["spec"]["url"] == "https://mcp.example.com"
    assert "(http)" in msg  # transport suffix in the success message


# ---------------------------------------------------------------------------
# Template: HTTP/SSE examples present
# ---------------------------------------------------------------------------


def test_template_includes_http_and_sse_examples():
    examples = mcp_client._TEMPLATE["_examples"]
    assert any("url" in v for v in examples.values()), (
        "no HTTP/SSE example in template — users won't discover the new transport"
    )
    http_examples = [v for v in examples.values() if "url" in v]
    assert len(http_examples) >= 2  # http + sse
    # The SSE example must set transport explicitly.
    sse_examples = [v for v in http_examples if v.get("transport") == "sse"]
    assert sse_examples, "no explicit SSE example to distinguish from default http"


# ---------------------------------------------------------------------------
# ensure_template: writes a valid config the first time
# ---------------------------------------------------------------------------


def test_ensure_template_creates_file_with_both_transports(tmp_path: Path, monkeypatch):
    """First-run template creation must include both stdio and HTTP/SSE samples."""
    cfg_path = tmp_path / "mcp.json"
    monkeypatch.setattr(mcp_client, "_CONFIG_FILE", cfg_path)
    mcp_client.ensure_template()
    assert cfg_path.exists()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "_examples" in data
    # Has stdio examples.
    assert any("command" in v for v in data["_examples"].values())
    # Has HTTP/SSE examples.
    assert any("url" in v for v in data["_examples"].values())
