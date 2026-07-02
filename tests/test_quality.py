"""Tests for the output-quality layer: fuzzy edits + diffs, self-heal checks,
repo map, git checkpoints (/undo), and streaming."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from coding_harness import checkers, gitops, repomap, tools
from coding_harness.control import CONTROL
from coding_harness.tools import ToolContext, _edit_file, _fuzzy_locate, _reindent


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=str(tmp_path), safety="auto")


@pytest.fixture
def events():
    seen: list = []
    CONTROL.set_listener(lambda ev: seen.append(ev))
    yield seen
    CONTROL.set_listener(None)


# ---------------------------------------------------------------------------
# Fuzzy edit matching
# ---------------------------------------------------------------------------


def test_fuzzy_locate_unique():
    text = "def f():\n    x = 1\n    return x\n"
    span, n = _fuzzy_locate(text, "x = 1\nreturn x")  # wrong indentation
    assert n == 1 and span == (1, 3)


def test_fuzzy_locate_ambiguous():
    text = "a\nb\na\nb\n"
    span, n = _fuzzy_locate(text, "a\nb")
    assert span is None and n == 2


def test_reindent_shifts_block():
    out = _reindent("x = 2\nif x:\n    y = 3", "    x = 1", "x = 1")
    assert out == "    x = 2\n    if x:\n        y = 3"


def test_edit_file_fuzzy_fallback(ctx: ToolContext, tmp_path: Path, events):
    f = tmp_path / "m.py"
    f.write_text("def f():\n    x = 1\n    return x\n", encoding="utf-8")
    # Model got the indentation wrong — exact match would fail.
    r = _edit_file(ctx, "m.py", "x = 1\nreturn x", "x = 2\nreturn x + 1")
    assert r.ok and "fuzzy-matched" in r.output
    assert f.read_text(encoding="utf-8") == "def f():\n    x = 2\n    return x + 1\n"
    assert any(ev.kind == "diff" for ev in events)


def test_edit_file_exact_still_works(ctx: ToolContext, tmp_path: Path):
    f = tmp_path / "m.py"
    f.write_text("a = 1\n", encoding="utf-8")
    r = _edit_file(ctx, "m.py", "a = 1", "a = 2")
    assert r.ok and "fuzzy" not in r.output
    assert f.read_text(encoding="utf-8") == "a = 2\n"


def test_edit_file_not_found_reports(ctx: ToolContext, tmp_path: Path):
    (tmp_path / "m.py").write_text("a = 1\n", encoding="utf-8")
    r = _edit_file(ctx, "m.py", "zzz", "yyy")
    assert not r.ok and "not found" in r.error


def test_write_file_emits_diff(ctx: ToolContext, tmp_path: Path, events):
    r = tools._write_file(ctx, "new.py", "print('hi')\n")
    assert r.ok
    assert any(ev.kind == "diff" for ev in events)


# ---------------------------------------------------------------------------
# Self-heal checkers
# ---------------------------------------------------------------------------


def test_quick_check_python_syntax_error(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text("def f(:\n    pass\n", encoding="utf-8")
    report = checkers.quick_check("bad.py", str(tmp_path))
    assert report  # some checker (ruff or py_compile) flags it


def test_quick_check_python_ok(tmp_path: Path):
    good = tmp_path / "good.py"
    good.write_text("def f():\n    return 1\n", encoding="utf-8")
    assert checkers.quick_check("good.py", str(tmp_path)) is None


def test_quick_check_json(tmp_path: Path):
    (tmp_path / "a.json").write_text("{bad json", encoding="utf-8")
    assert checkers.quick_check("a.json", str(tmp_path))
    (tmp_path / "b.json").write_text('{"ok": true}', encoding="utf-8")
    assert checkers.quick_check("b.json", str(tmp_path)) is None


def test_quick_check_unknown_ext_ignored(tmp_path: Path):
    (tmp_path / "x.zzz").write_text("whatever", encoding="utf-8")
    assert checkers.quick_check("x.zzz", str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Repo map
# ---------------------------------------------------------------------------


def test_repo_map_lists_source_files(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def alpha():\n    pass\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("class Beta:\n    pass\n", encoding="utf-8")
    repomap.invalidate()
    m = repomap.build_repo_map(str(tmp_path))
    assert "main.py" in m and "pkg/mod.py" in m
    block = repomap.repo_map_block(str(tmp_path))
    assert block.startswith("\nREPO MAP")


def test_repo_map_empty_workspace(tmp_path: Path):
    repomap.invalidate()
    assert repomap.build_repo_map(str(tmp_path)) == ""
    assert repomap.repo_map_block(str(tmp_path)) == ""


# ---------------------------------------------------------------------------
# Git checkpoint + restore
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "tracked.txt").write_text("original\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def test_snapshot_and_restore_roundtrip(repo: Path):
    ws = str(repo)
    # Pre-run state: tracked file modified + one untracked file present.
    (repo / "tracked.txt").write_text("user edit\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("keep me\n", encoding="utf-8")

    sha = gitops.snapshot_worktree(ws)
    assert sha

    # "Agent" wreaks havoc: modifies, deletes, creates.
    (repo / "tracked.txt").write_text("agent broke this\n", encoding="utf-8")
    (repo / "untracked.txt").unlink()
    (repo / "agent_new.txt").write_text("agent artifact\n", encoding="utf-8")
    assert gitops.snapshot_diff_stat(ws, sha)

    ok, msg = gitops.restore_snapshot(ws, sha)
    assert ok, msg
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "user edit\n"
    assert (repo / "untracked.txt").read_text(encoding="utf-8") == "keep me\n"
    assert not (repo / "agent_new.txt").exists()


def test_snapshot_non_git_returns_empty(tmp_path: Path):
    assert gitops.snapshot_worktree(str(tmp_path)) == ""


def test_restore_bad_sha_fails_cleanly(repo: Path):
    ok, msg = gitops.restore_snapshot(str(repo), "0" * 40)
    assert not ok and "not found" in msg


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_invoke_aggregates_and_flags(capsys):
    from langchain_core.messages import AIMessageChunk

    from coding_harness import executor

    class FakeLLM:
        def stream(self, messages):
            yield AIMessageChunk(content="Hel")
            yield AIMessageChunk(content="lo")

    CONTROL.reset()
    resp, streamed = executor._stream_invoke(FakeLLM(), [])
    assert streamed is True
    assert resp.content == "Hello"
    assert "Hello" in capsys.readouterr().out


def test_stream_invoke_falls_back_on_error():
    from langchain_core.messages import AIMessage

    from coding_harness import config, executor

    class BrokenLLM:
        def stream(self, messages):
            raise RuntimeError("no streaming")

    CONTROL.reset()
    with patch.object(config, "_invoke_with_retry", return_value=AIMessage(content="fallback")):
        resp, streamed = executor._stream_invoke(BrokenLLM(), [])
    assert streamed is False and resp.content == "fallback"


def test_run_executor_streams_final_answer(ctx: ToolContext, capsys):
    from langchain_core.messages import AIMessageChunk

    from coding_harness import config, executor
    from coding_harness.tokens import LEDGER

    class FakeLLM:
        def stream(self, messages):
            yield AIMessageChunk(content="The answer ")
            yield AIMessageChunk(content="is 42.")

    LEDGER.reset()
    CONTROL.reset()
    with (
        patch.object(config, "get_llm_for_task", return_value=FakeLLM()),
        patch.object(executor, "_try_bind_tools", return_value=None),
    ):
        result = executor.run_executor(
            task="q", ctx=ctx, max_steps=3, step_label="t", stream=True
        )
    assert result.stopped_reason == "done"
    assert result.streamed is True
    assert result.summary == "The answer is 42."
    assert "The answer is 42." in capsys.readouterr().out
