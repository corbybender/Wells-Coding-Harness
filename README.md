<p align="center">
  <img src="wells_logo.png" alt="Wells — the tripod coding robot" width="260">
</p>

<h1 align="center">Wells</h1>

<p align="center">
  <a href="https://github.com/corbybender/Wells-Coding-Harness/actions/workflows/ci.yml"><img src="https://github.com/corbybender/Wells-Coding-Harness/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/wells-index/"><img src="https://img.shields.io/pypi/v/wells-index?label=wells-index" alt="wells-index on PyPI"></a>
</p>

A local, **model-agnostic agentic coding platform**: a full-screen terminal TUI
plus an orchestration engine of autonomous tool-using agents
(`planner → architect → coder → tester → reviewer → finisher`) that actually
read files, make edits, run tests, and verify their own work — Claude-Code /
OpenCode style. **Provider-agnostic**: drive it with Z.ai GLM, OpenAI,
Anthropic, OpenRouter, Ollama, or any OpenAI-compatible endpoint. Ships with a
Rust structural repo index (`wells-index`), an MCP server *and* MCP client,
git-checkpointed undo, and a deterministic verification layer.

## What it does

```
START → indexer → planner ──(simple plan)──────────┐
                     │ (complex)                    ▼
                  architect ─────────────────────► coder → tester ──(tests FAIL)──┐
                                                     ▲         │ (pass/unknown)   │
                                                     │         ▼                  │
                                                summarizer ◄─ reviewer ◄──────────┘
                                                     ▲         │(INCOMPLETE)
                                                     └─────────┘
                                                               │(COMPLETE / cap)
                                                    finisher (memory + git/PR) → END
```

- **Indexer** builds/refreshes the structural repo index (symbols, references,
  call graph) before anything else runs.
- **Planner** is agentic: it investigates the codebase with read-only tools
  (index-first lookups, plus a `parallel_research` fan-out that runs 2–4
  read-only subagents concurrently), then writes a concrete plan with exact
  files and line numbers — and labels it `SIMPLE` or `COMPLEX`.
- **Architect** validates complex plans; simple plans skip straight to the
  coder (one less LLM call).
- **Coder** drives the agentic executor: reads, edits, creates files, and runs
  verification inside your workspace. Edits are whitespace-tolerant
  (an indentation slip in the model's match string no longer wastes a
  round-trip) and every applied change shows a colorized diff live.
  After each write, the harness itself runs the fastest checker for the file
  type (ruff/py_compile, `node --check`, JSON parse) and injects failures
  into the model's next observation — broken code is caught in milliseconds,
  not a tester round-trip later.
- **Tester** runs a *deterministic gate first*: if the repo has a recognizable
  test setup, the harness executes the suite and records the exit code as
  ground truth. Green suite → the LLM interpretation pass is skipped entirely.
  Red suite → routes straight back to the coder (reviewer skipped) with the
  failure report as feedback.
- **Reviewer** independently verifies the work (reads changed files, re-runs
  tests) and emits `COMPLETE` / `INCOMPLETE`. Tester + reviewer route to the
  cheap model profile when one is configured (`CHEAP_VERIFY`).
- **Summarizer** condenses durable context on loop iterations (bounded by
  `MAX_ITERATIONS`).
- **Finisher** writes a lesson to `AGENTS.md` project memory and optionally
  creates a `wells/<slug>` branch + commit + PR.

The session is **checkpointed after every node**, so a crash loses at most one
node's work and `/resume` continues from the last state. Every run also
snapshots your working tree first — `/undo` reverts everything a run changed.

## The TUI

Running `wells` with no arguments opens the full-screen TUI: scrollable output
log, multi-line prompt (Shift+Enter for newlines, ↑/↓ history, persisted
across sessions), and an always-on status bar showing workspace, model, live
token count **and dollar cost**, operating mode, pinned-file count, and — while
running — the current agent activity (`coder-1 · step 12/60`, current tool).
**Escape cancels a running task** cooperatively at the next step boundary.
Answers stream token-by-token.

| Command | What it does |
|---|---|
| `/mode plan\|approve\|auto\|dryrun` | Switch operating mode (read-only / confirm each change / full autonomy / simulate) |
| `/add <path>` / `/drop <path>` / `/context` | Pin files into every prompt (guaranteed context, token-trimmed) |
| `/undo` | Revert everything the last run changed (automatic pre-run git checkpoint) |
| `/config` | Modal settings panel — all settings grouped, edit in place, saves to `.env` |
| `/mcp` | Modal MCP server manager — add / enable / disable / test / remove servers |
| `/rules` | Operating rules + open liabilities (`list` / `reload` / `discharge <id>`) |
| `/orchestrate` | Route the next message through the full planning graph |
| `/resume` / `/sessions` | Continue a previous session / browse history |
| `/index` | Build or refresh the structural repo index |
| `/doctor` | Diagnose the environment (model ping + latency, API key, TLS, index health, git, checkers) |
| `/export [path]` | Save the session transcript to a file |
| `/status` `/info` `/help` `/clear` `/quit` | Status panel, effective config, command list, clear history, exit |

Under `approve` mode, destructive tool calls (writes, shell commands, MCP
calls) pause the run and ask y/N in the TUI. `AUTO_COMMIT=1` (opt-in) commits
each successful run with an LLM-generated Conventional Commits message and a
Wells authorship trailer.

## Provider profiles (model-agnostic)

Models are configured as named **profiles**. Any number can coexist; one is
*active*, one optionally *cheap* (used for summarization/classification and,
with `CHEAP_VERIFY`, the tester/reviewer).

| Profile name | Provider kind | Notes |
|---|---|---|
| `zai` (default) | `openai` (OpenAI-compatible) | Z.ai GLM via the **coding endpoint** `/api/coding/paas/v4/`. Backward-compatible with legacy `ZAI_*` vars. |
| `openai` | `openai` | OpenAI directly |
| `openrouter` | `openai` | OpenRouter (hundreds of models) |
| `anthropic` | `anthropic` | Requires `pip install langchain-anthropic` |
| `ollama` | `ollama` | Local models; requires `pip install langchain-ollama` |
| `local` | `openai` | Any local vLLM / Ollama OpenAI shim |
| `together` / `groq` / `fireworks` / `deepseek` / `mistral` | `openai` | One-line setup |
| `google` / `bedrock` / `azure` | provider-specific | Optional provider packages |

A profile is configured with three env vars:

```bash
MODEL_<name>=<model-id>            # required
API_KEY_<name>=<key>               # if the provider needs one
BASE_URL_<name>=<url>              # for OpenAI-compatible endpoints
```

Select which profiles exist and which is active:

```bash
MODEL_PROFILES=zai,openrouter,local
MODEL_PROFILE=openrouter           # the active profile
MODEL_PROFILE_CHEAP=zai            # optional: cheaper model for subtasks
```

Optional provider packages are imported lazily — the harness runs out-of-the-box
with only `langchain-openai` (the OpenAI-compatible path covers Z.ai, OpenAI,
OpenRouter, Together, Groq, Fireworks, local vLLM, Ollama's OpenAI shim, …).

Dollar costs are estimated from a built-in rate table (GLM / GPT / Claude /
DeepSeek / local); pin exact rates per profile with
`MODEL_PRICE_<profile>=<in>,<out>` ($/1M tokens).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.

### Option A — Cloned standalone (no install needed)

After `git clone`, use the launcher script at the repo root. It handles the
venv automatically — no `cd`, no `uv run`, no install step:

```bash
git clone https://github.com/corbybender/Wells-Coding-Harness.git
cd Wells-Coding-Harness

./wells config          # first run: set up your provider
./wells info            # show effective configuration
./wells                 # open the TUI
./wells "your goal"     # run the harness single-shot on THIS repo
```

Windows: use `wells.bat` instead of `./wells`.

### Option B — Drive a DIFFERENT project (embedding)

Wells can operate on any project, not just itself. Clone it anywhere, then
point it at your project with `--workspace`:

```bash
# From your project root, with Wells cloned as a subfolder:
./Wells-Coding-Harness/wells --workspace . "add JWT auth to the Express app"

# Or an absolute path:
./Wells-Coding-Harness/wells --workspace /home/me/myapp "fix the failing tests"

# Preview only (plan mode — describe edits without applying):
./Wells-Coding-Harness/wells --workspace . --plan "refactor the data layer"
```

All file operations, shell commands, and tests run inside the `--workspace`
directory; Wells' own source is never touched.

### Option C — Global install (available everywhere)

Install once, then `wells` (or `coding-harness`) is on your PATH:

```bash
uv tool install Wells-Coding-Harness    # or: pipx install .
wells "your goal"                        # from any directory
wells --workspace /path/to/project "goal"
```

**Note:** During installation, `uv` may show a warning about hardlinks across
filesystems (`WARN: Hardlink or symlink copy required…`). It's harmless;
suppress it with `UV_LINK_MODE=copy`.

### Manual setup (any option)

```bash
cp .env.example .env             # then edit .env with your API key
# or run the interactive menu:
./wells config
```

## CLI

Both `wells` and `coding-harness` work identically (they're the same entry point).

```
wells                                     # launch the TUI
wells "<goal>"                            # run the full harness (single-shot)
wells --workspace /path "fix the bug"     # run against another project
wells --safety dryrun "goal"              # force dry-run (preview only)
wells --plan "<goal>"                     # plan mode: plan edits, don't apply
wells config                              # interactive settings menu (terminal)
wells info                                # show effective configuration
wells principles                          # show active operating principles (AGENT.md)
wells --version                           # show version
wells "<goal>" MAX_ITERATIONS=5           # inline setting override
```

In the TUI, `/config` opens the modal settings panel instead (same schema,
same `.env` persistence).

## Safety model

The agent operates inside a **workspace root** (path escapes blocked) and a
**safety policy** for writes, shell commands, and MCP tool calls:

| Mode (`/mode` or `HARNESS_SAFETY`) | Behaviour |
|---|---|
| `auto` (default) | Execute immediately, confined to `WORKSPACE_ROOT`. Destructive commands (`rm -rf /`, `mkfs`, …) are always blocked. |
| `approve` | Every destructive action pauses the run and asks y/N in the TUI. |
| `dryrun` | Never execute — describe what *would* happen. Truly side-effect free. |
| `plan` (`PLAN_MODE=1`) | All mutating tools simulate; reads still work. Preview exactly what would change. |

Two extra safety nets regardless of mode: every run **snapshots the working
tree** (including untracked files) to a hidden git commit before starting —
`/undo` restores it — and `MAX_RUN_TOKENS` hard-caps a run's spend.

## Operating rules — deterministic, not hopeful

Prompted rules are probabilistic: every model eventually forgets a wall of
rules at prompt top. Wells enforces rules in tiers, strongest first:

1. **Tool-boundary enforcement** (`.wells/rules.yaml`, merged over
   `~/.wells/rules.yaml`): every tool call is checked *before* execution.
   `block` refuses outright, `confirm` pauses for y/N, `warn` injects the rule
   into the model's next observation, and `liability` registers a stateful
   obligation — e.g. *a rented GPU was started and must be terminated*.
   **A run cannot silently end with an open liability**: Wells attempts an
   automatic discharge pass, marks the run INCOMPLETE otherwise, shows a red
   `⚠ LIABILITY` badge in the status bar, warns on next startup, and keeps
   the ledger in `~/.wells/liabilities.json` so even a crash can't lose track
   of a running paid resource.
2. **Moment-of-relevance injection**: when a rule fires, its text lands in
   the exact tool observation the model reads next — one rule, at the moment
   it applies — plus open liabilities pinned into the never-pruned working
   memory.
3. **Prompt + audit**: the workspace `RULES.md` (universal, incident-derived
   rules) is injected into every system prompt, and the reviewer audits
   compliance — violations force the INCOMPLETE loop.

Manage with `/rules` (list, reload after editing, `discharge <id>` to
acknowledge a manually-closed resource). Default rules ship globally on first
run: GPU-rental teardown tracking, force-push/hard-reset confirmation,
bulk-rsync confirmation, auth-preflight and monitor-quality warnings.
Kill-switch: `RULES_ENFORCE=0`; auto-discharge: `RULES_AUTODISCHARGE`.

## Repository index (wells-index)

Wells ships a Rust structural indexer ([`wells-index`](wells-index/) —
[on PyPI](https://pypi.org/project/wells-index/)): tree-sitter parsing for 8
languages, SQLite + LZ4 storage, BLAKE3 incremental hashing. It powers:

- **Index-first tools** — `find_symbol`, `find_references`, `find_callers`,
  `search_symbols`, `list_symbols`: exact file:line answers instead of grep
  walls (~98% fewer tokens per lookup).
- **The repo map** — a compressed *files → key symbols* map injected into
  planner/coder prompts, **ranked by relevance to the current goal**, so the
  model starts knowing where things live instead of spending steps on
  discovery.
- A **background file watcher** keeps the index live during a session; the
  indexer node refreshes it before every orchestrate run. `/doctor` detects a
  stale native core and self-repairs from the repo-bundled binaries.

## Behavioral principles (AGENT.md)

Every agent in the harness — regardless of which model you've configured — is
governed by the same behavioral constitution: the **operating principles** in
`AGENT.md`. These 11 rules (Think Before Coding, Simplicity First, Surgical
Changes, Goal-Driven Execution, Deterministic First, Budget Everything, Verify
Before Trust, Fail Loud, Isolate Side Effects, Check Before Declaring Done,
Evidence Over Confidence) are **always injected** into every agent's system
prompt, so the harness behaves consistently whether you drive it with GLM, GPT,
Claude, Gemini, or a local model.

This is distinct from per-project `AGENTS.md` memory:

| | `AGENT.md` (bundled) | `AGENTS.md` (per-project) |
|---|---|---|
| **Purpose** | Behavioral rules — *how* the agent works | Project knowledge — *what* it knows about this repo |
| **Scope** | Every run, every agent, every project | One project; accumulates over runs |
| **Ship location** | Inside the harness package | The workspace root |
| **Who writes it** | The harness authors (you can override) | The harness finisher + you |

### Override precedence (highest first)

1. **`WELLS_PRINCIPLES` env var** — point at any file path. Use this for
   organization-wide principles across all projects.
2. **`AGENT.md` in the workspace root** — lets a team customize the rules for
   one project. Version-controlled with that project.
3. **The bundled `AGENT.md`** — the default constitution shipped with the
   harness. Always present as a baseline.

Inspect the active principles with `wells principles` or the MCP
`get_principles` tool.

## MCP — server *and* client

### Server: drive Wells from other agents

The harness exposes its capabilities as a
[Model Context Protocol](https://modelcontextprotocol.io) server over stdio,
so external agent clients (Claude Code, OpenCode, Codex CLIs, Gemini CLI, …)
can invoke the harness:

```bash
coding-harness-mcp          # console script
```

Exposed tools include `run_agent_task` (full loop), `plan_task`,
`review_code`, `run_executor`, `spawn_subagent`, `search_repo`, `read_file`,
`run_command`, `git_status`, `get_memory`, `compress_logs`,
`get_harness_info`, and `get_principles`.

```json
{
  "mcpServers": {
    "coding-harness": { "command": "coding-harness-mcp", "args": [] }
  }
}
```

### Client: give Wells external tools

Wells also connects *out* to stdio MCP servers (databases, docs, GitHub,
memory banks) and registers their tools for the agent as
`mcp_<server>_<tool>`. Configure via the **`/mcp` modal manager** in the TUI
(add / enable / disable / test / remove — no JSON editing), the `/mcp add …`
subcommands, or by editing `~/.wells/mcp.json` directly (created on first run
with ready-to-enable samples: fetch, filesystem, github, postgres, sqlite,
memory). The `MCP_SERVERS` env var (JSON) overrides the file. Every external
call passes the safety gate, so `approve` and `dryrun` apply to MCP tools too.

## Project structure

```
src/coding_harness/
├── main.py            # CLI entry: run / config / info / principles
├── cli.py             # REPL command layer: slash commands, run paths
├── tui.py             # Textual TUI: log, prompt, status bar, modals
├── control.py         # run control: cooperative cancel, activity, UI events
├── settings.py        # settings schema + .env persistence
├── config.py          # env vars, budgets, workspace/safety knobs
├── providers.py       # named provider profiles → chat-model factory
├── pricing.py         # dollar-cost estimation from the token ledger
├── state.py           # TypedDict LangGraph state
├── graph.py           # LangGraph workflow with conditional routing
├── runtime.py         # run_step(): LLM call + usage capture (reasoning nodes)
├── executor.py        # agentic tool loop: native+text tools, masking, streaming
├── tools.py           # repo tools: read/glob/grep/write/edit/shell/subagents
├── checkers.py        # post-edit self-heal: ruff / node --check / json
├── repomap.py         # goal-ranked repo map (files → key symbols)
├── safety.py          # workspace confinement + auto/approve/dryrun gate
├── subagents.py       # parallel read-only research fan-out
├── memory.py          # AGENTS.md project memory
├── gitops.py          # branch/commit/PR + working-tree snapshots (/undo)
├── finisher.py        # post-run memory write-back + git/PR node
├── sessions.py        # session persistence, /resume, per-node checkpoints
├── tokens.py          # token estimation, thread-safe ledger, usage report
├── context.py         # categorized, budget-trimmed prompt assembly
├── compress.py        # log/output compressor
├── summarize.py       # rolling task-state summarizer
├── index_tools.py     # wells-index bindings + stale-core self-repair
├── index_watcher.py   # background incremental re-indexing
├── mcp_server.py      # MCP server (Wells as a tool provider)
├── mcp_client.py      # MCP client (external tools for the agent)
├── logo.py            # TUI glyph lockup
├── principles.py      # AGENT.md injection
└── agents/            # planner / architect / coder / tester / reviewer
wells-index/           # Rust structural indexer (tree-sitter + SQLite)
.github/workflows/     # ci.yml (pytest) + release-index.yml (PyPI wheels)
```

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `MODEL_PROFILES` | `zai` | Comma-separated list of configured profile names |
| `MODEL_PROFILE` | `zai` | Active profile for main reasoning/coding |
| `MODEL_PROFILE_CHEAP` | _(blank)_ | Profile for low-stakes subtasks (defaults to active) |
| `MODEL_<name>` / `API_KEY_<name>` / `BASE_URL_<name>` | — | Per-profile model, key, endpoint |
| `MODEL_PRICE_<name>` | _(rate table)_ | Exact $/1M rates: `in,out` |
| `WORKSPACE_ROOT` | `cwd` | Directory the agent is confined to |
| `HARNESS_SAFETY` | `auto` | `auto` / `approve` / `dryrun` (or use `/mode`) |
| `PLAN_MODE` | `0` | When on, mutating tools simulate |
| `MAX_ITERATIONS` | `0` (no limit) | Max coder↔reviewer loops |
| `MAX_TOOL_STEPS` | `0` (no limit) | Max tool-call rounds per executor run |
| `PLANNER_MAX_STEPS` / `TESTER_MAX_STEPS` / `REVIEWER_MAX_STEPS` / `SUBAGENT_MAX_STEPS` | `0` (no limit) | Per-agent step caps |
| `MAX_RUN_TOKENS` | `0` (off) | Hard token cap per run; warns at 80% |
| `SELF_CHECK` | `1` | Post-edit lint/syntax self-heal |
| `CHEAP_VERIFY` | `1` | Route tester/reviewer to the cheap profile |
| `AUTO_COMMIT` | `0` | Commit each successful run (Conventional Commits) |
| `STREAM_OUTPUT` | `1` | Stream answers token-by-token |
| `INDEX_AUTO_UPDATE` | `1` | Keep the repo index fresh automatically |
| `MCP_SERVERS` | _(blank)_ | JSON server map; overrides `~/.wells/mcp.json` |
| `SHELL_TIMEOUT` | `120` | Max seconds for a single shell command |
| `TOKEN_BUDGET_MAX_INPUT` | `24000` | Input budget per call (above this, trims) |
| `SUMMARIZE_ON_LOOP` | `1` | Replace durable context with a summary on loops |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` | `180` / `5` | Per-call timeout / transient-error retries |
| `WELLS_OPEN_PR` | `0` | When `1`, the finisher pushes + opens a PR via `gh` |
| `WELLS_PRINCIPLES` | _(bundled)_ | Path to a custom AGENT.md constitution |
| `BLOCKED_COMMANDS` | _(see source)_ | `\|`-separated regex patterns always refused |

Legacy `ZAI_*` variables keep working unchanged — they seed the built-in `zai`
profile.

## Token & cost optimization

| Component | What it does |
|---|---|
| **Estimator + Ledger** | tiktoken-based, auto-calibrated; thread-safe per-step actuals from `usage_metadata` |
| **Dollar pricing** | Live cost in the status bar and run footers |
| **Observation masking** | Old tool outputs compressed to typed one-liners; AI reasoning turns kept verbatim |
| **Working memory** | Compact structured state (files read/modified, failed approaches, test status) injected every round — prevents re-reads and repeated failures |
| **Repo map** | Goal-ranked structure injection — fewer discovery steps |
| **Deterministic gates** | Real test runs and fast checkers replace LLM judgment calls where possible |
| **Summarizer + trimming** | Rolling task-state summary on loops; categorized budget trimming |
| **Model router** | Cheap profile for summarization/classification/verification |

## Tests & CI

```bash
uv run pytest -q          # 224 tests
```

The suite covers provider resolution, tool confinement + every safety mode,
the executor loop (mocked model — no API credits needed), cancellation and
budget stops, graph routing (complexity skip, test-gate fail-fast), fuzzy
edits, self-heal checkers, repo-map ranking, git snapshot/undo, pricing, MCP
client CRUD, and the settings persistence. GitHub Actions runs it on every
push/PR (`ci.yml`); `release-index.yml` builds and publishes `wells-index`
wheels (Linux/macOS/Windows × Python 3.12/3.13) to PyPI on an `index-v*` tag.

## Roadmap

- Parallel *write* steps via worktree-per-subagent isolation (reads already fan out).
- SSE/HTTP MCP client transport (stdio today).
- Prompt-cache-friendly masking batches (measure `cache_read` deltas first).
- Embedding-based retrieval for very large repos.
- Async task tracking for MCP `run_agent_task`.
