<p align="center">
  <img src="wells_logo.png" alt="Wells — the tripod coding robot" width="260">
</p>

<h1 align="center">Wells</h1>

A local, **model-agnostic agentic coding harness**. Give it a software
development goal and it runs an orchestration loop of
`planner → architect → coder → tester → reviewer → finisher`, where the
coder/tester/reviewer are **autonomous tool-using agents** that actually
read files, make edits, run tests, and verify their own work — Claude-Code /
OpenHands style. The harness is **provider-agnostic**: drive it with Z.ai GLM,
OpenAI, Anthropic, OpenRouter, Ollama, or any OpenAI-compatible endpoint.

## What it does

```
START → planner → architect → coder → tester → reviewer → decision
         ^                                           |
         |____ summarizer ← (if INCOMPLETE) _________|
                                                     ↓
                                          finisher (memory + git/PR) → END
```

- **Planner / architect** turn the goal into a plan + architecture (reads
  `AGENTS.md` project memory).
- **Coder** is an agentic loop: it uses `read_file` / `glob` / `grep` /
  `write_file` / `edit_file` / `run_command` / `run_tests` / `spawn_subagent`
  to actually implement the goal in your workspace, then verifies its work.
- **Tester** runs the real test suite and reports pass/fail with file:line refs.
- **Reviewer** independently re-checks the work (reads changed files, re-runs
  tests) and emits `COMPLETE` / `INCOMPLETE`.
- On `INCOMPLETE`, the **summarizer** condenses durable context and the loop
  returns to the coder (bounded by `MAX_ITERATIONS`).
- **Finisher** writes a lesson to `AGENTS.md` (so the harness learns across
  runs) and optionally creates a `wells/<slug>` branch + commit + PR.

Everything goes through a **token-optimization layer** (estimator + calibration,
per-category context trimming, log compression, rolling summaries, model router,
cache-friendly prompts) and a **workspace confinement + safety policy** layer.

## Provider profiles (model-agnostic)

Models are configured as named **profiles**. Any number can coexist; one is
*active*, one optionally *cheap* (used for summarization/compression).

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

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.

### Option A — Cloned standalone (no install needed)

After `git clone`, use the launcher script at the repo root. It handles the
venv automatically — no `cd`, no `uv run`, no install step:

```bash
git clone https://github.com/corbybender/Wells-Coding-Harness.git
cd Wells-Coding-Harness

./wells config          # first run: set up your provider (interactive menu)
./wells info            # show effective configuration
./wells "your goal"     # run the harness on THIS repo
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

**Note:** During installation, `uv` may show a warning about hardlinks across filesystems:
```
WARN: Hardlink or symlink copy required, try setting UV_LINK_MODE=copy
```

This is harmless and cosmetic. It happens when `uv` tries to link package files across different filesystems (e.g., C: and D: drives on Windows, or different mount points on Linux). To suppress the warning, set the environment variable before install:

```bash
# Windows (PowerShell)
$env:UV_LINK_MODE = "copy"
uv tool install Wells-Coding-Harness

# Windows (cmd)
set UV_LINK_MODE=copy && uv tool install Wells-Coding-Harness

# macOS / Linux
UV_LINK_MODE=copy uv tool install Wells-Coding-Harness
```

Once installed, the harness itself handles this automatically — you won't see the warning again.

### Manual setup (any option)

```bash
cp .env.example .env             # then edit .env with your API key
# or run the interactive menu:
./wells config
```

## CLI

Both `wells` and `coding-harness` work identically (they're the same entry point).

```
wells                                     # launch the interactive chat REPL
wells "<goal>"                            # run the full harness (single-shot)
wells --workspace /path "fix the bug"     # run against another project
wells --safety dryrun "goal"              # force dry-run (preview only)
wells --plan "<goal>"                     # plan mode: plan edits, don't apply
wells config                              # interactive settings menu
wells info                                # show effective configuration
wells principles                          # show active operating principles (AGENT.md)
wells --version                           # show version
wells "<goal>" MAX_ITERATIONS=5           # inline setting override
```

Flags can be combined and work with subcommands:

```
wells --workspace . --safety approve info   # show config for THIS dir, approve mode
wells --workspace ../other-repo config      # edit settings (writes .env in repo root)
```

Running `wells` with no arguments opens a rich, interactive TUI (similar to Claude Code or OpenCode). It maintains conversational context across multiple goals and streams output tokens live to your terminal. Type goals naturally, or use slash commands:

| Command | What it does |
|---|---|
| `/mode plan\|approve\|auto\|dryrun` | Switch operating mode (read-only / confirm each change / full autonomy / simulate) |
| `/add <path>` / `/drop <path>` / `/context` | Pin files into every prompt (guaranteed context) |
| `/undo` | Revert everything the last run changed (automatic pre-run git checkpoint) |
| `/orchestrate` | Route the next message through the full planner→coder→tester→reviewer graph |
| `/resume` / `/sessions` | Continue a previous session / browse history |
| `/index` | Build or refresh the structural repo index |
| `/doctor` | Diagnose the environment (model, TLS, index, git, checkers) |
| `/export [path]` | Save the session transcript to a file |
| `/status` `/config` `/help` `/quit` | Status panel, settings, command list, exit |

The status bar always shows workspace, model, live token count + dollar cost, operating mode, and (while running) the current agent activity — Escape cancels a running task, ↑/↓ recall prompt history, Shift+Enter inserts a newline.

External MCP servers can be plugged in via `MCP_SERVERS` (JSON) or `~/.wells/mcp.json` — their tools appear to the agent as `mcp_<server>_<tool>`. Opt-in `AUTO_COMMIT=1` commits each successful run with a generated Conventional Commits message.

### Interactive settings menu

`coding-harness config` shows every setting grouped (Providers, Run, Tokens,
LLM) and lets you change any value by number/name, switch or add provider
profiles, and persist to `.env` — all in one place. Changes apply live and are
written back comment-preserving.

```
================================================================
 Wells harness — current settings
================================================================
[Providers]
  MODEL_PROFILE                openrouter
  ...
> p) Switch / edit provider profile (fast path)
> +) Add a new provider profile
> MAX_ITERATIONS        edit by env-var name
> s) Save & exit     q) Quit without saving     w) Write .env now
```

## Safety model

The agent operates inside a **workspace root** (path escapes blocked) and a
**safety policy** for writes and shell commands:

| `HARNESS_SAFETY` | Behaviour |
|---|---|
| `auto` (default) | Execute immediately, confined to `WORKSPACE_ROOT`. Destructive commands (`rm -rf /`, `mkfs`, …) are always blocked. |
| `approve` | Require an approval callback; degrades to dry-run when no callback is wired. |
| `dryrun` | Never execute — describe what *would* happen. Truly side-effect free. |

`PLAN_MODE=1` forces all mutating tools to simulate (reads still work), so you
can preview exactly what the agent would change.

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
`get_principles` tool. Override by dropping an `AGENT.md` in your project root
or setting `WELLS_PRINCIPLES=/path/to/your-rules.md`.

## Project structure

```
src/coding_harness/
├── main.py            # CLI entry: run / config / info
├── settings.py        # interactive settings menu + .env persistence
├── config.py          # env vars, budgets, workspace/safety knobs
├── providers.py       # named provider profiles → chat-model factory (Layer 0)
├── state.py           # TypedDict LangGraph state
├── graph.py           # LangGraph workflow (planner→…→reviewer→finisher)
├── runtime.py         # run_step(): LLM call + usage capture (reasoning nodes)
├── executor.py        # agentic tool-calling loop (Layer 2) — native + text fallback
├── tools.py           # repo tools: read/list/glob/grep/write/edit/shell/subagent
├── safety.py          # workspace confinement + auto/approve/dryrun gate
├── subagents.py       # parallel research/fix subagents (Layer 3)
├── memory.py          # AGENTS.md project memory (Layer 3)
├── gitops.py          # branch/commit/diff/PR via git + gh (Layer 3)
├── finisher.py        # post-run memory write-back + git/PR node
├── tokens.py          # token estimation, ledger, usage report
├── context.py         # categorized, budget-trimmed prompt assembly
├── compress.py        # log/output compressor
├── summarize.py       # rolling task-state summarizer
├── mcp_server.py      # MCP server exposing all capabilities
└── agents/            # planner / architect / coder / tester / reviewer
```

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `MODEL_PROFILES` | `zai` | Comma-separated list of configured profile names |
| `MODEL_PROFILE` | `zai` | Active profile for main reasoning/coding |
| `MODEL_PROFILE_CHEAP` | _(blank)_ | Profile for low-stakes subtasks (defaults to active) |
| `MODEL_<name>` | — | Model id for profile `<name>` |
| `API_KEY_<name>` | — | API key for profile `<name>` |
| `BASE_URL_<name>` | — | OpenAI-compatible base URL for profile `<name>` |
| `WORKSPACE_ROOT` | `cwd` | Directory the agent is confined to |
| `HARNESS_SAFETY` | `auto` | `auto` / `approve` / `dryrun` |
| `PLAN_MODE` | `0` | When on, coder plans but does not apply edits |
| `MAX_ITERATIONS` | `3` | Max coder↔reviewer loops |
| `MAX_TOOL_STEPS` | `25` | Max tool-call rounds per executor run |
| `SHELL_TIMEOUT` | `120` | Max seconds for a single shell command |
| `TOKEN_BUDGET_MAX_INPUT` | `24000` | Input budget per call (above this, trims) |
| `SUMMARIZE_ON_LOOP` | `1` | Replace durable context with a summary on loop iterations |
| `SUMMARIZE_THRESHOLD` | `1500` | Tokens above which durable context is summarized |
| `LLM_TIMEOUT` | `180` | Per-call timeout |
| `LLM_MAX_RETRIES` | `5` | Retry attempts for transient errors |
| `WELLS_OPEN_PR` | `0` | When `1`, the finisher pushes + opens a PR via `gh` |
| `BLOCKED_COMMANDS` | _(see source)_ | `\|`-separated regex patterns always refused |

### Legacy `ZAI_*` variables

Existing `.env` files using `ZAI_API_KEY` / `ZAI_MODEL` / `ZAI_ENDPOINT` keep
working unchanged — they seed the built-in `zai` profile. Explicit
`MODEL_zai` / `BASE_URL_zai` / `API_KEY_zai` vars take precedence.

## MCP server

The harness exposes its capabilities as a [Model Context Protocol](https://modelcontextprotocol.io)
server over stdio, so external agent clients (Claude Code, OpenCode, Codex CLIs,
Gemini CLI, …) can invoke the harness.

```bash
coding-harness-mcp          # console script
python -m coding_harness    # same thing
```

### Exposed tools (13)

| Tool | Description |
|---|---|
| `run_agent_task` | Full harness loop (planner→…→reviewer→finisher) with workspace + safety overrides |
| `plan_task` | Planner + architect only (fast) |
| `review_code` | Reviewer only on provided context |
| `run_executor` | Single autonomous executor loop for an arbitrary task |
| `spawn_subagent` | Focused research (read-only) or fix subagent |
| `search_repo` | Glob + grep search (read-only) |
| `read_file` | Read a workspace file (read-only) |
| `run_command` | Run a shell command (confined, blocklisted, gated) |
| `git_status` | Git status + diff stat (read-only) |
| `get_memory` | Read project memory (`AGENTS.md`) |
| `compress_logs` | Compress log output (ANSI/dup/tail) |
| `get_harness_info` | Effective configuration |

### Client configuration

```json
{
  "mcpServers": {
    "coding-harness": { "command": "coding-harness-mcp", "args": [] }
  }
}
```

## Token optimization

| Component | What it does |
|---|---|
| **Estimator** | tiktoken-based, auto-calibrated against actual API responses |
| **TokenLedger** | Per-step actuals (input/output/reasoning/cache_read) from `usage_metadata` |
| **Token Usage Report** | End-of-run report with per-step table + category breakdown + savings |
| **ContextManager** | Categorized chunks, stable-prefix ordering, priority budget trimming |
| **Compressor** | ANSI strip, duplicate/blank collapse, tail-window, traceback preserve |
| **Summarizer** | Rolling task-state summary on loop iterations (threshold-guarded) |
| **Model Router** | Cheaper model for summarization/compression via `MODEL_PROFILE_CHEAP` |
| **Prompt-Cache Prefix** | `SystemMessage` + deterministic chunk order (cache-friendly) |

## Tests

```bash
uv run python -m pytest tests/ -v
```

The suite covers provider-profile resolution, the coding-endpoint precedence,
tool confinement + every safety mode, the agentic executor loop (with a mock
model so it runs without API credits), MCP tool registration, and the
settings-menu `.env` persistence.

## Roadmap

- Async task tracking for MCP `run_agent_task` (return a task ID, poll later).
- Per-call ledger isolation for concurrent MCP requests.
- Embedding-based code retrieval (replace full-repo injection for large repos).
