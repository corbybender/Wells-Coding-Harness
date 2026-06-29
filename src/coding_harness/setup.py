"""Auto-setup on first run: build indexer, prompt for workspace, auto-index."""

import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def _ensure_indexer_built() -> bool:
    """Build wells-index if not already installed.

    Returns True if indexer is available (either already installed or successfully built).
    """
    try:
        import wells_index  # noqa: F401
        return True
    except ImportError:
        pass

    # Try to build from local source
    wells_root = Path(__file__).parent.parent.parent
    indexer_dir = wells_root / "wells-index"

    if not indexer_dir.exists():
        console.print("[yellow]wells-index source not found. Indexer unavailable.[/yellow]")
        return False

    console.print("[cyan]Building indexer (first time only, this may take a minute)...[/cyan]")
    try:
        # Use maturin develop for faster, in-place build
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "maturin"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        result = subprocess.run(
            ["maturin", "develop"],
            cwd=str(indexer_dir),
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            console.print(
                f"[yellow]Indexer build failed. Skipping.\n{result.stderr[:500]}[/yellow]"
            )
            return False

        console.print("[green]✓ Indexer built successfully[/green]")
        return True
    except Exception as e:
        console.print(f"[yellow]Could not build indexer: {e}[/yellow]")
        return False


def _prompt_for_workspace() -> str | None:
    """Ask user for workspace path on first run."""
    from pathlib import Path

    console.print("\n[bold cyan]First run setup[/bold cyan]")
    console.print("Enter the path to your project (or press Enter to skip indexing for now):")
    console.print("Example: Q:\\myproject  or  /home/me/myproject\n")

    try:
        path_input = input("> ").strip()
        if not path_input:
            return None

        path = Path(path_input).expanduser().resolve()
        if not path.exists():
            console.print(f"[red]Path does not exist: {path}[/red]")
            return None
        if not path.is_dir():
            console.print(f"[red]Not a directory: {path}[/red]")
            return None

        return str(path)
    except KeyboardInterrupt:
        return None


def _auto_index_workspace(workspace: str) -> bool:
    """Auto-index the workspace on first run."""
    from coding_harness import index_tools
    from coding_harness.tools import ToolContext

    console.print(f"\n[cyan]Indexing {workspace}...[/cyan]")
    try:
        ctx = ToolContext(workspace=workspace)
        result = index_tools.index_workspace(ctx)
        if result.ok:
            console.print(f"[green]{result.output}[/green]")
            return True
        else:
            console.print(f"[yellow]Indexing incomplete: {result.error or result.output}[/yellow]")
            return False
    except Exception as e:
        console.print(f"[yellow]Could not index workspace: {e}[/yellow]")
        return False


def first_run_setup() -> None:
    """Run setup on first use: build indexer, ask for workspace, auto-index."""
    from coding_harness import config

    # Check if already set up (workspace defined, indexer available)
    if config.WORKSPACE_ROOT != os.getcwd():
        # Workspace already configured
        return

    # Try to build indexer
    indexer_ok = _ensure_indexer_built()
    if not indexer_ok:
        console.print("[yellow]Indexer not available. Using grep for code search.[/yellow]")
        return

    # Prompt for workspace
    workspace = _prompt_for_workspace()
    if not workspace:
        return

    # Save to .env
    try:
        from coding_harness import settings
        settings.update_env_file(Path(".env"), {"WORKSPACE_ROOT": workspace})
        os.environ["WORKSPACE_ROOT"] = workspace
    except Exception as e:
        console.print(f"[yellow]Could not save workspace to .env: {e}[/yellow]")

    # Auto-index
    _auto_index_workspace(workspace)
    console.print()
