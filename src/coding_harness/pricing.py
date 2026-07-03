"""Dollar-cost estimation from the token ledger.

Users think in dollars, not tokens. This maps model labels (``profile:model``)
to $/1M-token rates and prices a run from the ledger's per-step records.

Rates are looked up by substring match on the model name, so ``zai:glm-5.2``
matches the ``glm-5`` entry. The table holds approximate list prices; exact
rates can be pinned per profile with ``MODEL_PRICE_<profile>=in,out`` (dollars
per 1M tokens, e.g. ``MODEL_PRICE_zai=0.6,2.2``). Unknown models price as
None and the UI simply omits the dollar figure.
"""

from __future__ import annotations

import os

# (input $/1M, output $/1M) — approximate list prices; longest match wins.
_RATE_TABLE: dict[str, tuple[float, float]] = {
    # Z.ai GLM
    "glm-5": (0.60, 2.20),
    "glm-4.7": (0.60, 2.20),
    "glm-4": (0.50, 1.80),
    # OpenAI
    "gpt-5.2": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "o3": (2.00, 8.00),
    # Anthropic
    "claude-fable": (25.00, 125.00),
    "claude-opus": (15.00, 75.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-haiku": (0.80, 4.00),
    # DeepSeek
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek": (0.27, 1.10),
    # Mistral / Meta via routers (rough)
    "mistral-large": (2.00, 6.00),
    "llama": (0.50, 1.00),
    # Local endpoints are free
    "ollama": (0.0, 0.0),
    "local": (0.0, 0.0),
}


def rate_for(model_label: str) -> tuple[float, float] | None:
    """Return ($/1M in, $/1M out) for a ledger model label, or None if unknown."""
    label = (model_label or "").lower()

    # Explicit per-profile override: MODEL_PRICE_<profile>=in,out
    profile = label.split(":", 1)[0] if ":" in label else ""
    if profile:
        raw = os.environ.get(f"MODEL_PRICE_{profile}") or os.environ.get(
            f"MODEL_PRICE_{profile.upper()}"
        )
        if raw:
            try:
                i, o = (float(x) for x in raw.split(",", 1))
                return (i, o)
            except Exception:
                pass

    best: tuple[float, float] | None = None
    best_len = 0
    for key, rate in _RATE_TABLE.items():
        if key in label and len(key) > best_len:
            best, best_len = rate, len(key)
    return best


def ledger_cost(steps) -> float | None:
    """Price a list of ledger StepUsage records. None when no model is priceable."""
    total = 0.0
    priced_any = False
    for s in steps:
        rate = rate_for(getattr(s, "model", ""))
        if rate is None:
            continue
        priced_any = True
        total += s.input_tokens * rate[0] / 1e6 + s.output_tokens * rate[1] / 1e6
    return total if priced_any else None


def run_cost() -> float | None:
    """Dollar cost of the current run (ledger resets at run start)."""
    from coding_harness.tokens import LEDGER

    with LEDGER._lock:
        steps = list(LEDGER.steps)
    return ledger_cost(steps)


def fmt(cost: float | None) -> str:
    """Human dollar string: $0.0042 → '$0.004', $1.234 → '$1.23', None → ''."""
    if cost is None:
        return ""
    if cost >= 0.10:
        return f"${cost:.2f}"
    if cost >= 0.01:
        return f"${cost:.3f}"
    return f"${cost:.4f}"
