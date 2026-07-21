<p align="center">
  <img src="wells_logo.png" alt="Wells — the tripod coding robot" width="260">
</p>

<h1 align="center">Wells</h1>

<p align="center">
  <a href="https://github.com/corbybender/Wells/actions/workflows/ci.yml"><img src="https://github.com/corbybender/Wells/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/wells-index/"><img src="https://img.shields.io/pypi/v/wells-index?label=wells-index" alt="wells-index on PyPI"></a>
</p>

A local, **model-agnostic agentic coding platform**: a full-screen terminal TUI
plus an orchestration engine of autonomous tool-using agents
(`planner → architect → coder → tester → reviewer → finisher`) that actually
read files, make edits, run tests, and verify their own work — Claude-Code /
OpenCode style. **Provider-agnostic**: drive it with Z.ai GLM, OpenAI,
Anthropic, OpenRouter, Ollama, or any OpenAI-compatible endpoint. Ships with a
Rust structural repo index (`wells-index`), an MCP server *and* MCP client,
git-checkpointed undo, a deterministic verification layer, **agent skills**
(load-on-domain know-how), **CodeAct** (sandboxed code execution), and
**background agents** (concurrent fan-out).

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
| `/skills` | Modal skills manager — list / view / add / edit / remove `SKILL.md` know-how |
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

After `git clone`, run the one-time setup script for your OS. It puts the
`wells` command on your PATH — no package build, no PyPI, works the same on
Mac, Linux, and Windows:

```bash
git clone https://github.com/corbybender/Wells.git
cd Wells

./install.sh             # Mac/Linux — or install.ps1 on Windows (PowerShell)
# open a new terminal, then:

wells config              # first run: set up your provider
wells info                 # show effective configuration
wells                      # open the TUI
wells "your goal"          # run the harness single-shot on THIS repo
```

`wells` itself still handles the venv/deps automatically on every run — the
installer only wires up the command name. You can also skip the installer
and call the launcher directly: `./wells` (Mac/Linux) or `wells.bat`
(Windows).

### Option B — Drive a DIFFERENT project (embedding)

Wells can operate on any project, not just itself. Clone it anywhere, then
point it at your project with `--workspace`:

```bash
# From your project root, with Wells cloned as a subfolder:
./Wells/wells --workspace . "add JWT auth to the Express app"

# Or an absolute path:
./Wells/wells --workspace /home/me/myapp "fix the failing tests"

# Preview only (plan mode — describe edits without applying):
./Wells/wells --workspace . --plan "refactor the data layer"
```

All file operations, shell commands, and tests run inside the `--workspace`
directory; Wells' own source is never touched.

### Option C — Global install (available everywhere)

Install once, then `wells` (or `wells`) is on your PATH:

```bash
uv tool install Wells    # or: pipx install .
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

Both `wells` and `wells` work identically (they're the same entry point).

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

## Agent capabilities

Three capability layers let the claw *teach itself*, *compute*, and *work in
parallel* — each is a self-contained, feature-gated module that plugs into the
same executor loop and safety model. Together they address the core insight
from the [Microsoft Agent Framework "scaling the claw"](https://devblogs.microsoft.com/agent-framework/agent-harness-scaling-the-claw-or-harness-capabilities/)
article: stuffing every instruction into the system prompt doesn't scale, some
questions need *computation* not *reasoning*, and blocking fan-out wastes the
parent agent's time.

### Skills — load-on-demand know-how

#### The problem

Stuffing every how-to into the system prompt bloats context and dilutes focus.
Wells already injects `AGENT.md` principles, `RULES.md`, a goal-ranked repo
map, and pinned context into every prompt — adding domain-specific how-to
(tutorials, runbooks, architecture deep-dives) to that always-on block would
starve the budget for the actual task.

#### The solution: progressive disclosure

**Skills** are small `SKILL.md` files that package a chunk of know-how. The
agent sees only each skill's **name and one-line description** up front
(injected into the system prompt as a compact index), and loads the **full
body** on demand via the `load_skill` tool — *only when a request matches that
skill*. Context stays small and focused; domain how-to scales without bloating
every call.

This is the natural complement to `AGENTS.md` memory:

| | `AGENTS.md` (project memory) | Skills |
|---|---|---|
| **Content** | *Accumulated facts* about a repo (what the harness learned) | *How-to procedures* authored by you |
| **Visibility** | Always-on (small, trimmed by the budget) | Name + description always visible; body loads on demand |
| **Who writes it** | The harness finisher + you | You (via `/skills` menu or by hand) |
| **Size** | Kept small (budget-trimmed) | Can be large (only loaded when relevant) |

#### SKILL.md format

Each skill lives in `skills/<name>/SKILL.md` with YAML front-matter + markdown
body:

```markdown
---
name: release-checklist
description: How to cut and publish a new release of this project.
---

1. Bump the version in `package.json` and `Cargo.toml`
2. Update `CHANGELOG.md` with the changes since the last tag
3. Run the full test suite: `uv run pytest -q`
4. Tag: `git tag -a v0.x.0 -m "Release v0.x.0"`
5. Push tags: `git push --tags`
6. The CD pipeline publishes to PyPI automatically
```

The front-matter fields:

| Field | Required | Purpose |
|---|---|---|
| `name` | Yes (defaults to folder name) | The identifier the agent uses with `load_skill` |
| `description` | Recommended | One line shown in the always-on index — should help the agent decide when to load it |

The body is free-form markdown — instructions, code blocks, links, diagrams.
It's capped at ~8 KB when loaded (truncated with a notice) so a single skill
can't blow the context budget.

#### Discovery

Skills are discovered from (first match wins, so earlier roots can shadow):

1. `<workspace>/skills/` — the conventional location
2. Any extra directory in `WELLS_SKILLS_PATHS` (path-separator list)

Two layouts are supported:

```
skills/
  release/SKILL.md        ← skill folder (recommended)
  add-provider/SKILL.md
  SKILL.md                ← a single skill at the skills/ root
```

Discovery is cached by directory mtime, so editing or adding a skill file
invalidates the cache automatically. The `skills.clear_cache()` call after
every mutation (create/update/delete) ensures the next read sees the change.

#### How the agent uses skills

1. Every system prompt includes a compact index:

   ```
   === AVAILABLE SKILLS (load with the load_skill tool) ===
   - release-checklist: How to cut and publish a new release of this project.
   - add-provider: How to add a new model provider profile.
   Call load_skill(name) to load a skill's full instructions when a request
   matches one. Do not load a skill unless it is relevant.
   === END SKILLS ===
   ```

2. When a request matches a skill (e.g. "cut a release"), the agent calls
   `load_skill("release-checklist")` and the full body lands in its context.
3. If no skill matches, nothing is loaded — zero overhead.

#### Managing skills with `/skills`

Manage skills from the TUI or CLI:

| Command | What it does |
|---|---|
| `/skills` | Open the modal manager — list, view, add, edit, remove (keyboard-driven) |
| `/skills list` | List all discovered skills (name + description) |
| `/skills show <name>` | Print the full `SKILL.md` file |
| `/skills add <name>` | Create a new skill (interactive; or use the modal form with Ctrl+S) |
| `/skills edit <name>` | Edit an existing skill's description and/or body |
| `/skills remove <name>` | Delete a skill (its folder + file) |

**TUI modal** — bare `/skills` opens a full-screen manager:

| Key | Action |
|---|---|
| `↑` `↓` | Select a skill |
| `Enter` | View the full `SKILL.md` |
| `a` | Add a new skill (form: name, description, markdown body editor) |
| `e` | Edit the selected skill |
| `d` | Remove the selected skill |
| `Esc` | Close |

In the add/edit form: `Ctrl+S` saves, `Esc` cancels.

All mutations go through the **safety gate** (plan/approve/dryrun apply),
**validate the skill name** (lowercase letters, digits, hyphens — blocks path
traversal), and only delete skills **under the workspace `skills/` tree**
(skills loaded from `WELLS_SKILLS_PATHS` can't be deleted from the menu).

**Configuration:**

| Variable | Default | Description |
|---|---|---|
| `WELLS_SKILLS` | `1` | Discover skills and expose `load_skill` (set `0` to disable) |
| `WELLS_SKILLS_PATHS` | _(blank)_ | Extra skill search dirs (OS path-separator list) |

### CodeAct — let it compute

#### The problem

Some questions are *calculations*, not lookups: "what's the total LOC across
the changed files", "does this regex match all 12 of these strings", "generate
the cartesian product of these test configurations", "count how many functions
transitively call `auth_check`". Doing arithmetic in the model's head or
eyeballing a regex is exactly what the harness's "Deterministic First" and
"Verify Before Trust" principles say not to do.

#### The solution: `run_code`

**CodeAct** gives the agent a `run_code` tool: it writes a small Python
snippet, the harness runs it in a **workspace-confined subprocess**, and
returns structured `stdout` / `stderr` / `exit code`. The agent gets a clean,
bounded result to reason over — no guessing.

**Example tool calls the agent makes:**

```python
# Count lines changed in the working tree
import subprocess
diff = subprocess.check_output(["git", "diff", "--stat"]).decode()
print(diff)

# Validate a regex against test strings
import re
pat = re.compile(r'^\d{4}-\d{2}-\d{2}$')
for s in ["2024-01-15", "99-1-1", "2024-13-40"]:
    print(f"{s}: {'match' if pat.match(s) else 'no match'}")
```

#### Confinement + guardrails

| Guardrail | What it does |
|---|---|
| **Workspace-confined** | `cwd` = workspace, so `open("src/utils.py")` works for repo inspection |
| **Source screening** | Refuses code containing `os.system`, `subprocess`, `popen`, `__import__`, or `fork()` — use `run_command` for shell work |
| **Deny-list screening** | The same `BLOCKED_COMMANDS` regex list that screens `run_command` is applied to the source text (catches `rm -rf /` in a string literal) |
| **Output truncation** | stdout capped at 8 KB, stderr at 4 KB — a runaway `print` in a loop can't blow the budget |
| **Timeout** | Hard wall-clock cap (default 30s via `CODEACT_TIMEOUT`; also bounded by `SHELL_TIMEOUT`) |
| **Safety gate** | Honours plan/dry-run/approve modes like every other mutating tool |

**Why a confined subprocess, not Hyperlight/Monty?** Zero extra dependencies —
works out of the box everywhere Python runs. Workspace confinement + the
existing safety gate + the deny-list give the same first line of defense the
article's `LocalShellExecutor` relies on. A Docker-isolated executor is a
future option for untrusted input.

**Configuration:**

| Variable | Default | Description |
|---|---|---|
| `WELLS_CODEACT` | `1` | Expose the `run_code` tool (set `0` to disable) |
| `CODEACT_TIMEOUT` | `30` | Max seconds for a single `run_code` execution |

### Background agents — concurrent fan-out

#### The problem

`parallel_research` already fans out 2–4 read-only research subagents in
parallel — but it **blocks**: the parent agent waits for all subagents to
finish before it can do anything else. If one subagent takes 30 seconds, the
parent is stuck for 30 seconds. The fan-out timing is also the *tool's*
decision, not the agent's.

#### The solution: start / check / collect

**Background agents** flip the blocking pattern to the async start / check /
collect model from the article. The agent gets three tools:

| Tool | What it does | Returns |
|---|---|---|
| `bg_start` | Launch a sub-agent on a background daemon thread | Handle id (e.g. `bg-1`) — immediately, does not block |
| `bg_status` | Poll all background agents | List with status (`running`/`done`/`error`/`cancelled`) + elapsed seconds |
| `bg_collect` | Collect a finished agent's report (once) | The subagent's full report, or "still running" if it isn't done |

The fan-out becomes the **agent's decision**: it starts N tasks, keeps working
(reading files, making edits, running tests), and collects results when
convenient — checking back periodically with `bg_status`.

**Example workflow the agent drives:**

```
bg_start(task="research the auth module's token validation flow")    → bg-1
bg_start(task="research the database migration history")             → bg-2
bg_start(task="find all callers of the deprecated API")              → bg-3

  # Agent keeps working while they run:
  read_file("src/main.py")
  edit_file("src/main.py", ...)
  run_tests()

bg_status                                                            → bg-1: done, bg-2: done, bg-3: running
bg_collect(id="bg-1")                                                → report from auth research
bg_collect(id="bg-2")                                                → report from migration research
  # bg-3 still running — collect later or move on
```

#### Roles — research, fix, worktree

| Role | Edits? | Where | Use when |
|---|---|---|---|
| `research` (default) | No | — | Read-only investigation; safe to fan out widely |
| `fix` | Yes | Parent workspace directly | One editor in flight, or edits target disjoint files |
| `worktree` | Yes | Its own isolated `git worktree`, cherry-picked into the parent on `bg_collect` | Multiple write-fan-outs target overlapping areas, or whenever isolation is cheaper than reasoning about interleaving |

The `worktree` role is what unblocks **parallel write steps**: two
`bg_start role=worktree` agents run genuinely concurrently against their own
checkouts (shared object store — fast, disk-cheap), and `bg_collect` merges
each one's commit back into the parent. On conflict the cherry-pick is
aborted and the diff is returned to the parent agent for manual re-apply — no
surprise merges, no semantic guesses. Requires git; non-git workspaces get an
error pointing at `role=fix`.

```
bg_start(task="refactor the auth middleware", role="worktree")        → bg-1
bg_start(task="refactor the session middleware", role="worktree")     → bg-2
bg_start(task="refactor the rate-limiter",   role="worktree")         → bg-3

  # All three edit concurrently in their own worktrees; the parent's
  # working tree is untouched until each is collected.

bg_status                                                            → all three: done
bg_collect(id="bg-1")                                                → merged into parent
bg_collect(id="bg-2")                                                → CONFLICT, diff returned
bg_collect(id="bg-3")                                                → merged into parent
```

#### Lifecycle + safety

| Property | Behaviour |
|---|---|
| **Concurrency** | Each sub-agent runs on a daemon thread (LLM calls are I/O-bound, matching `parallel_research`) |
| **Registry** | Process-wide, keyed by short stable ids (`bg-1`, `bg-2`, …); resets at the start of each executor run so slots don't leak |
| **Collect once** | A result is collected at most once, then cleared — keeps memory bounded across a long run |
| **Recursion blocked** | A sub-agent cannot start its own background agents (`ctx.subagent` is checked at dispatch) |
| **Cooperative cancellation** | Escape / `CONTROL.cancel()` marks running slots as cancelled; pending threads check at step boundaries |
| **Worktree reaping** | For `role="worktree"`, the worktree + branch are reaped on collect and on `reset()` — a cancelled run never leaks disk |
| **Roles** | `role=research` (read-only, default), `role=fix` (parent workspace), or `role=worktree` (isolated checkout, merged on collect) |
| **Safety gate** | Each sub-agent's tool calls pass through the same safety gate as the parent |

**Contrast with `parallel_research`:**

| | `parallel_research` | Background agents (`bg_*`) |
|---|---|---|
| **Blocking** | Yes — parent waits for all to finish | No — returns immediately, collect later |
| **Fan-out timing** | Tool's decision (2–4 fixed) | Agent's decision (any number) |
| **Parent works during?** | No | Yes |
| **Read-only?** | Yes | `research` = yes; `fix` / `worktree` = can edit |
| **Isolation** | N/A (read-only) | `fix` = parent workspace; `worktree` = own checkout |
| **Use case** | Quick parallel exploration | Long-running fan-out the parent checks back on |

**Configuration:**

| Variable | Default | Description |
|---|---|---|
| `WELLS_BG_AGENTS` | `1` | Expose `bg_start`/`bg_status`/`bg_collect` (set `0` to disable) |
| `WELLS_BG_WORKTREES` | `1` | Allow `bg_start role=worktree` (isolated git worktree per sub-agent; set `0` to refuse the role without disabling the bg tools) |


## MCP — server *and* client

### Server: drive Wells from other agents

The harness exposes its capabilities as a
[Model Context Protocol](https://modelcontextprotocol.io) server over stdio,
so external agent clients (Claude Code, OpenCode, Codex CLIs, Gemini CLI, …)
can invoke the harness:

```bash
wells-mcp          # console script
```

Exposed tools include `run_agent_task` (full loop), `plan_task`,
`review_code`, `run_executor`, `spawn_subagent`, `search_repo`, `read_file`,
`run_command`, `git_status`, `get_memory`, `compress_logs`,
`get_harness_info`, and `get_principles`.

```json
{
  "mcpServers": {
    "wells": { "command": "wells-mcp", "args": [] }
  }
}
```

### Client: give Wells external tools

Wells also connects *out* to MCP servers (databases, docs, GitHub, memory
banks) and registers their tools for the agent as `mcp_<server>_<tool>`.
**Two transports are supported:**

| Transport | Spec shape | Use when |
|---|---|---|
| **stdio** | `{"command": "...", "args": [...]}` | Local subprocess — the classic MCP servers (`uvx mcp-server-fetch`, `npx @modelcontextprotocol/server-*`) |
| **HTTP** (streamable-http) | `{"url": "https://...", "headers": {...}}` | Remote MCP server speaking the newer spec (default when `url` is present) |
| **SSE** (legacy) | `{"url": "https://...", "transport": "sse"}` | Remote server that only speaks the older SSE protocol |

Configure via the **`/mcp` modal manager** in the TUI (add / enable /
disable / test / remove — no JSON editing), the `/mcp add …` subcommands
(auto-routes: a second arg starting with `http(s)://` becomes an HTTP
server; otherwise it's stdio), or by editing `~/.wells/mcp.json` directly
(created on first run with ready-to-enable samples: fetch, filesystem,
github, postgres, sqlite, memory, plus HTTP/SSE templates). The
`MCP_SERVERS` env var (JSON) overrides the file. Every external call
passes the safety gate, so `approve` and `dryrun` apply to MCP tools too.

## Project structure

```
src/wells/
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
├── skills.py          # Agent skills: discoverable SKILL.md, load-on-demand
├── codeact.py         # CodeAct: sandboxed run_code tool
├── background.py      # Background agents: bg_start/bg_status/bg_collect (research/fix/worktree)
├── worktree.py        # Per-subagent git worktrees (role=worktree isolation + cherry-pick)
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
| `WELLS_MASK_BATCH` | `4` | Batch-stable masking: don't re-mask until the cutoff has advanced this many rounds past the last batch (0 = mask every round, the old behavior) |
| `WELLS_SKILLS` | `1` | Discover `skills/<name>/SKILL.md` and expose `load_skill` (on/off) |
| `WELLS_SKILLS_PATHS` | _(blank)_ | Extra skill search dirs (path-separator list) |
| `WELLS_CODEACT` | `1` | Expose the sandboxed `run_code` tool for in-context computation |
| `CODEACT_TIMEOUT` | `30` | Max seconds for a single `run_code` execution |
| `WELLS_BG_AGENTS` | `1` | Expose `bg_start` / `bg_status` / `bg_collect` for concurrent fan-out |
| `WELLS_BG_WORKTREES` | `1` | Allow `bg_start role=worktree` (isolated git worktree per sub-agent) |

Legacy `ZAI_*` variables keep working unchanged — they seed the built-in `zai`
profile.

## Token & cost optimization

| Component | What it does |
|---|---|
| **Estimator + Ledger** | tiktoken-based, auto-calibrated; thread-safe per-step actuals from `usage_metadata` |
| **Dollar pricing** | Live cost in the status bar and run footers |
| **Observation masking** | Old tool outputs compressed to typed one-liners; AI reasoning turns kept verbatim |
| **Batch-stable masking** | Masking fires in batches (`_MASK_BATCH_ROUNDS`), so the provider's prompt cache stays warm between batches instead of being invalidated every round. `wells analyze` reports cache breaks round-by-round |
| **Working memory** | Compact structured state (files read/modified, failed approaches, test status) injected every round — prevents re-reads and repeated failures |
| **Repo map** | Goal-ranked structure injection — fewer discovery steps |
| **Deterministic gates** | Real test runs and fast checkers replace LLM judgment calls where possible |
| **Summarizer + trimming** | Rolling task-state summary on loops; categorized budget trimming |
| **Model router** | Cheap profile for summarization/classification/verification |

## Tests & CI

```bash
uv run pytest -q          # 650 tests
```

The suite covers provider resolution, tool confinement + every safety mode,
the executor loop (mocked model — no API credits needed), cancellation and
budget stops, graph routing (complexity skip, test-gate fail-fast), fuzzy
edits, self-heal checkers, repo-map ranking, git snapshot/undo, pricing, MCP
client CRUD, background-agent worktree lifecycle (create/merge/conflict/reap),
and the settings persistence. GitHub Actions runs it on every
push/PR (`ci.yml`); `release-index.yml` builds and publishes `wells-index`
wheels (Linux/macOS/Windows × Python 3.12/3.13) to PyPI on an `index-v*` tag.

## Roadmap

- Embedding-based retrieval for very large repos.
