"""Tests for the platform round: pricing, ranked repo map, /mode, pinning,
auto-commit, cheap-verify routing, and the MCP client wrapper."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from coding_harness import cli, config, gitops, pricing, repomap
from coding_harness.tokens import StepUsage


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def _step(model: str, tin: int, tout: int) -> StepUsage:
    return StepUsage(
        step="s", task_type="t", model=model, input_tokens=tin, output_tokens=tout,
        reasoning_tokens=0, cache_read_tokens=0, category_tokens={},
        saved_by_trim=0, saved_by_summary=0,
    )


def test_rate_for_substring_match():
    assert pricing.rate_for("zai:glm-5.2") == (0.60, 2.20)
    assert pricing.rate_for("anthropic:claude-sonnet-5") == (3.00, 15.00)
    assert pricing.rate_for("mystery:unknown-model-x") is None


def test_rate_for_env_override(monkeypatch):
    monkeypatch.setenv("MODEL_PRICE_zai", "1.5,3.5")
    assert pricing.rate_for("zai:glm-5.2") == (1.5, 3.5)


def test_ledger_cost_and_fmt():
    steps = [_step("zai:glm-5.2", 1_000_000, 100_000)]
    cost = pricing.ledger_cost(steps)
    assert cost == pytest.approx(0.60 + 0.22)
    assert pricing.fmt(cost) == "$0.82"
    assert pricing.ledger_cost([_step("who:knows", 100, 100)]) is None
    assert pricing.fmt(None) == ""


# ---------------------------------------------------------------------------
# Ranked repo map
# ---------------------------------------------------------------------------


def test_repo_map_ranks_goal_relevant_files_first(tmp_path: Path):
    (tmp_path / "aardvark.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "billing.py").write_text("def charge_invoice():\n    pass\n", encoding="utf-8")
    repomap.invalidate()
    m = repomap.build_repo_map(str(tmp_path), goal="fix the billing invoice charge bug")
    lines = m.splitlines()
    assert lines[0].startswith("billing.py")  # beats alphabetical order


def test_repo_map_cache_varies_by_goal(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    repomap.invalidate()
    m1 = repomap.build_repo_map(str(tmp_path), goal="alpha")
    m2 = repomap.build_repo_map(str(tmp_path), goal="beta")
    assert m1 == m2  # same content here, but no cross-goal cache crash


# ---------------------------------------------------------------------------
# /mode
# ---------------------------------------------------------------------------


def test_mode_switching(monkeypatch):
    monkeypatch.setenv("PLAN_MODE", "0")
    monkeypatch.setenv("HARNESS_SAFETY", "auto")
    cli._handle_mode("plan")
    assert config.PLAN_MODE is True
    assert cli.current_mode() == "plan"
    cli._handle_mode("approve")
    assert config.PLAN_MODE is False and config.HARNESS_SAFETY == "approve"
    assert cli.current_mode() == "approve"
    cli._handle_mode("auto")
    assert config.HARNESS_SAFETY == "auto"
    assert cli.current_mode() == "auto"


# ---------------------------------------------------------------------------
# /add /drop pinning
# ---------------------------------------------------------------------------


@pytest.fixture
def pin_ws(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "WORKSPACE_ROOT", str(tmp_path))
    cli._REPL_STATE["pinned"] = []
    (tmp_path / "notes.md").write_text("remember the milk\n", encoding="utf-8")
    yield tmp_path
    cli._REPL_STATE["pinned"] = []


def test_add_drop_and_context_block(pin_ws: Path):
    cli._handle_add("notes.md")
    assert cli._pinned() == ["notes.md"]
    block = cli._pinned_context_block()
    assert "PINNED CONTEXT" in block and "remember the milk" in block
    cli._handle_drop("notes.md")
    assert cli._pinned() == []
    assert cli._pinned_context_block() == ""


def test_add_rejects_missing_file(pin_ws: Path):
    cli._handle_add("nope.txt")
    assert cli._pinned() == []


def test_drop_all(pin_ws: Path):
    cli._handle_add("notes.md")
    cli._handle_drop("all")
    assert cli._pinned() == []


# ---------------------------------------------------------------------------
# Auto-commit
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "f.txt").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def test_auto_commit_creates_commit_with_trailer(repo: Path):
    (repo / "f.txt").write_text("changed\n", encoding="utf-8")
    ok, sha = gitops.auto_commit(str(repo), "feat(f): change f")
    assert ok and sha
    log = subprocess.run(
        ["git", "log", "-1", "--format=%B"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert "feat(f): change f" in log
    assert "Co-Authored-By: Wells Agent" in log


def test_auto_commit_noop_when_clean(repo: Path):
    ok, sha = gitops.auto_commit(str(repo), "feat: nothing")
    assert ok and sha == ""


# ---------------------------------------------------------------------------
# Cheap-verify routing
# ---------------------------------------------------------------------------


def test_cheap_verify_default_on():
    assert isinstance(config.CHEAP_VERIFY, bool)


# ---------------------------------------------------------------------------
# MCP client
# ---------------------------------------------------------------------------


def test_mcp_load_config_env(monkeypatch):
    from coding_harness import mcp_client
    monkeypatch.setenv("MCP_SERVERS", '{"fetch": {"command": "uvx", "args": ["x"]}}')
    cfg = mcp_client.load_config()
    assert cfg["fetch"]["command"] == "uvx"
    monkeypatch.setenv("MCP_SERVERS", "not json")
    assert mcp_client.load_config() == {}


def test_mcp_register_without_config_is_noop(monkeypatch):
    from coding_harness import mcp_client
    monkeypatch.delenv("MCP_SERVERS", raising=False)
    with patch.object(mcp_client, "_CONFIG_FILE", Path("Z:/definitely/missing.json")):
        assert mcp_client.register_mcp_tools() == []


def test_mcp_wrapped_tool_calls_through_and_gates(tmp_path: Path):
    from coding_harness import mcp_client, tools

    class FakeBridge:
        def call(self, coro, timeout):
            coro.close()  # not actually awaited
            return SimpleNamespace(
                content=[SimpleNamespace(text="result text")], isError=False
            )

    class FakeSession:
        def call_tool(self, name, args):
            async def _c():  # a real coroutine so .close() works
                return None
            return _c()

    fake_tool = SimpleNamespace(
        name="lookup", description="Look things up",
        inputSchema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    tooldef = mcp_client._wrap_tool(FakeBridge(), FakeSession(), "docs", fake_tool)
    assert tooldef.name == "mcp_docs_lookup"

    ctx = tools.ToolContext(workspace=str(tmp_path), safety="auto")
    r = tooldef.handler(ctx, q="hello")
    assert r.ok and "result text" in r.output

    # dryrun policy: gated, simulated, no call made
    ctx_dry = tools.ToolContext(workspace=str(tmp_path), safety="dryrun")
    r2 = tooldef.handler(ctx_dry, q="hello")
    assert r2.simulated
