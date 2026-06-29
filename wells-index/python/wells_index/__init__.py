"""Wells Repository Indexer — Fast structural code indexing for large repositories."""

try:
    from wells_index._core import IndexEngine
except ImportError as e:
    raise ImportError(
        "Failed to import wells_index C extension. "
        "Make sure to build with: `maturin develop` in the wells-index directory"
    ) from e

__version__ = "0.1.0"
__all__ = ["IndexEngine"]
