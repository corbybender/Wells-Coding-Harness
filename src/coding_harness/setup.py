"""Auto-setup on first run: install Rust, build indexer, prompt for workspace, auto-index."""

import os
import subprocess
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def _ensure_rust_installed() -> bool:
    """Check if Rust is installed; if not, install it via rustup.

    Returns True if Rust is available (already or after install).
    Raises if install fails (not silent degradation).
    """
    # Check if rustc exists
    try:
        result = subprocess.run(
            ["rustc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass

    # Rust not found; try to install via rustup
    console.print("[cyan]Installing Rust toolchain (needed for indexer)...[/cyan]")

    # Windows
    if sys.platform == "win32":
        console.print("[cyan]Downloading rustup installer...[/cyan]")
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; "
                "Invoke-WebRequest -Uri 'https://win.rustup.rs' -OutFile 'rustup-init.exe'; "
                ".\\rustup-init.exe -y",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            err_msg = result.stderr or result.stdout or "(no output)"
            raise RuntimeError(
                f"Rust installation failed:\n{err_msg}\n\n"
                "Install manually from: https://rustup.rs"
            )
    else:
        # macOS/Linux
        console.print("[cyan]Downloading rustup...[/cyan]")
        result = subprocess.run(
            [
                "sh",
                "-c",
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            err_msg = result.stderr or result.stdout or "(no output)"
            raise RuntimeError(
                f"Rust installation failed:\n{err_msg}\n\n"
                "Install manually from: https://rustup.rs"
            )

    # Verify installation
    result = subprocess.run(
        ["rustc", "--version"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode == 0:
        console.print("[green]✓ Rust installed successfully[/green]")
        return True
    else:
        raise RuntimeError(
            "Rust installed but rustc verification failed. "
            "Check installation: rustup.rs"
        )


def _ensure_indexer_built() -> bool:
    """Build wells-index if not already installed.

    Returns True if indexer is available (either already installed or successfully built).
    Raises with diagnostic error if build fails.
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
        raise RuntimeError(
            f"wells-index source not found at {indexer_dir}. "
            "This should not happen if Wells is properly installed."
        )

    # Try maturin develop (preferred)
    console.print("[cyan]Building wells-index from source (requires Rust)...[/cyan]")
    result = subprocess.run(
        ["maturin", "develop"],
        cwd=str(indexer_dir),
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode == 0:
        try:
            import wells_index  # noqa: F401
            console.print("[green]✓ wells-index built successfully[/green]")
            return True
        except ImportError:
            pass

    # Fallback: use uv pip
    console.print("[cyan]Trying fallback: uv pip install -e wells-index...[/cyan]")
    result = subprocess.run(
        ["uv", "pip", "install", "-e", str(indexer_dir)],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode == 0:
        try:
            import wells_index  # noqa: F401
            console.print("[green]✓ wells-index installed successfully[/green]")
            return True
        except ImportError:
            pass

    # Both methods failed — show diagnostic error
    maturin_err = result.stderr or result.stdout or "(no output)"
    raise RuntimeError(
        f"Failed to build wells-index.\n\n"
        f"Error:\n{maturin_err}\n\n"
        f"Ensure Rust is installed (https://rustup.rs) and try again."
    )


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
    """Run setup on first use: install Rust, build indexer, ask for workspace, auto-index.

    Shows diagnostic errors; system still works without indexer (grep fallback).
    Only runs once (tracked by marker file in ~/.wells/).
    """
    try:
        # Check if setup has already run (marker file in ~/.wells/)
        marker_file = Path.home() / ".wells" / ".setup_complete"
        if marker_file.exists():
            return

        from coding_harness import config

        # Check if already set up (workspace defined, indexer available)
        if config.WORKSPACE_ROOT != os.getcwd():
            # Workspace already configured — mark setup as done
            marker_file.parent.mkdir(parents=True, exist_ok=True)
            marker_file.touch()
            return

        # Try to build indexer
        try:
            _ensure_rust_installed()
        except RuntimeError as e:
            console.print(f"[yellow]Warning: Rust installation skipped.\n{e}[/yellow]")
            console.print("[yellow]Indexer will not be available; falling back to grep.[/yellow]\n")
            return

        try:
            _ensure_indexer_built()
        except RuntimeError as e:
            console.print(f"[yellow]Warning: Indexer build failed.\n{e}[/yellow]")
            console.print("[yellow]Falling back to grep for code search.[/yellow]\n")
            return

        # Prompt for workspace (only if indexer succeeded)
        workspace = _prompt_for_workspace()
        if not workspace:
            return

        # Save to .env
        try:
            from coding_harness import settings
            settings.update_env_file(Path(".env"), {"WORKSPACE_ROOT": workspace})
            os.environ["WORKSPACE_ROOT"] = workspace
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save workspace to .env: {e}[/yellow]")

        # Auto-index
        _auto_index_workspace(workspace)
        console.print()

        # Mark setup as complete so it doesn't run again
        marker_file = Path.home() / ".wells" / ".setup_complete"
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        marker_file.touch()
    except Exception as e:
        # Unexpected error — show it
        console.print(f"[yellow]Unexpected error during setup: {e}[/yellow]")
        console.print("[yellow]Continuing with indexer unavailable.[/yellow]\n")
        # Still mark setup as complete so we don't retry on every startup
        try:
            marker_file = Path.home() / ".wells" / ".setup_complete"
            marker_file.parent.mkdir(parents=True, exist_ok=True)
            marker_file.touch()
        except Exception:
            pass
