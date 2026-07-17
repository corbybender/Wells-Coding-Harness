"""Tests for project memory (AGENTS.md): load, nested-merge, and write-back."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from wells import memory


# ---------------------------------------------------------------------------
# Root-only (existing behavior)
# ---------------------------------------------------------------------------


def test_load_empty_when_no_agents_md(tmp_path: Path):
    mem = memory.load(str(tmp_path))
    assert mem.exists is False
    assert mem.text == ""


def test_load_root_only(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("# root conventions\nuse tabs\n", encoding="utf-8")
    mem = memory.load(str(tmp_path))
    assert mem.exists is True
    assert "use tabs" in mem.text
    assert mem.path == tmp_path / "AGENTS.md"


# ---------------------------------------------------------------------------
# Nested merge (monorepo per-package conventions)
# ---------------------------------------------------------------------------


def test_nested_agents_md_merged_after_root(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("root: use snake_case\n", encoding="utf-8")
    pkg = tmp_path / "packages" / "api"
    pkg.mkdir(parents=True)
    (pkg / "AGENTS.md").write_text("api package: all routes need auth middleware\n",
                                    encoding="utf-8")

    mem = memory.load(str(tmp_path))
    assert "root: use snake_case" in mem.text
    assert "all routes need auth middleware" in mem.text
    # Root content precedes nested content (general before specific).
    assert mem.text.index("snake_case") < mem.text.index("auth middleware")
    # Nested block is labeled with its directory.
    assert "packages/api/AGENTS.md" in mem.text


def test_nested_agents_md_works_with_no_root_file(tmp_path: Path):
    """A repo with only per-package AGENTS.md (no root one) must still merge."""
    pkg = tmp_path / "svc"
    pkg.mkdir()
    (pkg / "AGENTS.md").write_text("svc-specific rule\n", encoding="utf-8")
    mem = memory.load(str(tmp_path))
    assert mem.exists is True
    assert "svc-specific rule" in mem.text
    assert mem.path == tmp_path / "AGENTS.md"  # path always points at root slot


def test_nested_merge_orders_shallowest_first(tmp_path: Path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "AGENTS.md").write_text("deep rule\n", encoding="utf-8")
    shallow = tmp_path / "a"
    (shallow / "AGENTS.md").write_text("shallow rule\n", encoding="utf-8")

    mem = memory.load(str(tmp_path))
    assert mem.text.index("shallow rule") < mem.text.index("deep rule")


def test_nested_discovery_skips_common_noise_dirs(tmp_path: Path):
    for noisy in (".git", "node_modules", "__pycache__", ".venv"):
        d = tmp_path / noisy / "sub"
        d.mkdir(parents=True)
        (d / "AGENTS.md").write_text("should never be found\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("root\n", encoding="utf-8")

    mem = memory.load(str(tmp_path))
    assert "should never be found" not in mem.text


def test_nested_discovery_capped(tmp_path: Path):
    for i in range(memory._MAX_NESTED_FILES + 5):
        d = tmp_path / f"pkg{i}"
        d.mkdir()
        (d / "AGENTS.md").write_text(f"rule {i}\n", encoding="utf-8")
    found = memory._discover_nested(tmp_path)
    assert len(found) == memory._MAX_NESTED_FILES


def test_nested_discovery_never_raises_on_permission_error(tmp_path: Path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("root\n", encoding="utf-8")
    with patch.object(memory, "_discover_nested", side_effect=OSError("denied")):
        mem = memory.load(str(tmp_path))  # must not raise
    assert "root" in mem.text


# ---------------------------------------------------------------------------
# Write-back still targets the root file only
# ---------------------------------------------------------------------------


def test_append_lesson_writes_root_not_nested(tmp_path: Path):
    pkg = tmp_path / "packages" / "api"
    pkg.mkdir(parents=True)
    (pkg / "AGENTS.md").write_text("api rule\n", encoding="utf-8")

    with patch("wells.safety.gate") as gate:
        gate.return_value.allowed = True
        path = memory.append_lesson(
            str(tmp_path), goal="fix bug", summary="fixed it",
        )
    assert path == tmp_path / "AGENTS.md"
    assert (pkg / "AGENTS.md").read_text(encoding="utf-8") == "api rule\n"  # untouched
    assert "fix bug" in (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
