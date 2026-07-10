# wells-index

Fast structural repository indexer for [Wells](https://github.com/corbybender/Wells). Uses Tree-sitter for language parsing, SQLite for storage, and BLAKE3 for incremental hashing.

## Install

Prebuilt wheels (Linux / macOS / Windows, Python 3.12 + 3.13) are on PyPI:

```bash
pip install wells-index
```

> **Note:** the 0.1.0 wheels had a defect — files were indexed but zero
> symbols were extracted. Use **0.1.1 or later**. Wells detects the stale core
> at runtime (`/doctor`) and self-repairs from the repo-bundled binaries.

## Features

- **Multi-language symbol extraction** — Python, JavaScript, TypeScript, Go, Rust, Java, C, C++
- **Incremental indexing** — Only re-parses changed files (via BLAKE3 hashing)
- **Compressed storage** — SQLite database with LZ4 compression
- **Fast queries** — O(1) symbol lookups, reference finding, call site discovery
- **98% token reduction** — Compared to grep-based code retrieval
- **PyO3 bindings** — Native Python extension for seamless integration

## Building

### Prerequisites

- Rust 1.70+
- Python 3.12+
- `maturin` >= 1.7

### Setup

1. Clone the repository and navigate to the `wells-index` directory:
   ```bash
   cd Wells/wells-index
   ```

2. **Vendor tree-sitter grammar sources** (one-time setup):
   
   The build system expects tree-sitter grammar C sources in `grammars/<language>/src/`:
   
   ```bash
   # Create grammar directories
   mkdir -p grammars/{python,javascript,typescript/typescript,go,rust,java,c,cpp}/src
   
   # Copy parser.c and scanner.c from official tree-sitter repos
   # Example for Python:
   curl https://raw.githubusercontent.com/tree-sitter/tree-sitter-python/master/src/parser.c \
       -o grammars/python/src/parser.c
   ```
   
   Or use git submodules:
   ```bash
   git submodule add https://github.com/tree-sitter/tree-sitter-python.git grammars/python
   git submodule add https://github.com/tree-sitter/tree-sitter-javascript.git grammars/javascript
   # ... etc for other languages
   ```

3. Build the extension:
   ```bash
   maturin develop
   ```

4. Verify installation:
   ```bash
   python -c "from wells_index import IndexEngine; print(IndexEngine.__doc__)"
   ```

## Usage

### Command Line

```bash
# Index the current directory
python -c "from wells_index import IndexEngine; e = IndexEngine('.'); e.index(); print(e.stats())"
```

### Python API

```python
from wells_index import IndexEngine

# Create indexer for workspace
engine = IndexEngine("/path/to/repo")

# Build/update index
stats = engine.index()
print(f"Indexed {stats['files_indexed']} files")

# Query the index
symbols = engine.find_symbol("MyClass")
for sym in symbols:
    print(f"{sym['file_path']}:{sym['start_line']} - {sym['name']} ({sym['kind']})")

# Find all references to a symbol
refs = engine.find_references("authenticate")
for ref in refs:
    print(f"{ref['file_path']}:{ref['start_line']}")

# Find all callers of a function
callers = engine.find_callers("process_request")
for caller in callers:
    print(f"{caller['file_path']}:{caller['start_line']} calls process_request")

# Prefix/substring search
results = engine.search_symbols("MyClass", limit=20)
for r in results:
    print(r)

# List all symbols in a file
symbols = engine.list_in_file("src/main.py")
for sym in symbols:
    print(f"  {sym['name']} ({sym['kind']})")

# Get repository stats
stats = engine.stats()
print(f"Total files: {stats['total_files']}")
print(f"Total symbols: {stats['total_symbols']}")
print(f"Total edges: {stats['total_edges']}")

# Clear index
engine.clear()
```

## Index Format

The index is stored in `.wells_index/index.db` (relative to the workspace root):

- **Compressed**: LZ4 compression applied on flush
- **Incremental**: BLAKE3 file hashes skip unchanged files
- **Portable**: SQLite database readable by standard tools

## Grammar Support

| Language | Status | Extension | Tree-sitter Repo |
|---|---|---|---|
| Python | ✓ | `.py` | tree-sitter-python |
| JavaScript | ✓ | `.js`, `.mjs`, `.cjs` | tree-sitter-javascript |
| TypeScript | ✓ | `.ts`, `.tsx` | tree-sitter-typescript |
| Go | ✓ | `.go` | tree-sitter-go |
| Rust | ✓ | `.rs` | tree-sitter-rust |
| Java | ✓ | `.java` | tree-sitter-java |
| C | ✓ | `.c`, `.h` | tree-sitter-c |
| C++ | ✓ | `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hh` | tree-sitter-cpp |

## Symbol Kinds

- `class` — Class/struct/interface definition
- `function` — Top-level function definition
- `method` — Method within a class
- `variable` — Variable/field definition
- `module` — Module/file-level scope

## Edge Kinds

- `calls` — Function/method call
- `references` — Symbol reference or usage
- `inherits` — Class inheritance or trait impl
- `imports` — Module import

## Performance

On a typical large repository (10k+ files, 1M+ symbols):

- **Initial indexing**: 5-10 seconds on modern hardware
- **Incremental update**: <100ms for changed files
- **Query latency**: <1ms for symbol lookups
- **Storage**: ~10-20% of source code size

## Architecture

- **Language detection**: File extension-based with shebang fallback
- **Parsing**: Tree-sitter C library (via Rust bindings)
- **Parallelism**: `rayon` for multi-core file scanning and parsing
- **Storage**: `rusqlite` with integer-mapped symbol names
- **Hashing**: BLAKE3 for incremental change detection
- **Compression**: LZ4 on database at rest

## Known Limitations

- No semantic analysis (only syntax-based extraction)
- Symbol deduplication not yet implemented
- Cross-file call resolution is name-based (not type-aware)

## Development

Run tests:
```bash
maturin develop
pytest tests/ -v
```

Build in release mode:
```bash
maturin build --release
```

## Releasing

Wheels are built and published by the repo's
[`release-index.yml`](../.github/workflows/release-index.yml) workflow
(maturin on Linux/macOS/Windows × Python 3.12/3.13, PyPI trusted publishing):

```bash
# bump version in Cargo.toml + pyproject.toml + python/wells_index/__init__.py
git tag index-v0.1.2 && git push origin index-v0.1.2
```

The prebuilt `.pyd` files committed under `python/wells_index/` are the
no-toolchain fallback used by `wells.bat` and Wells' stale-core self-repair;
refresh them from the built wheels when the Rust source changes.

## License

MIT OR Apache-2.0
