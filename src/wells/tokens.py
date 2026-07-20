"""Token estimation, budgeting, and usage accounting.

Phase 1 (observability) of the token-optimization layer. The estimator is
model-agnostic: it uses tiktoken as a cross-model baseline and then *calibrates*
itself against the real ``input_tokens`` returned by the API, so per-category
estimates track the provider's actual tokenization over a run.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

# tiktoken.get_encoding() used to run at import time. On a first run with no
# local cache it fetches the BPE vocab over HTTPS (openaipublic.blob.core.
# windows.net) — behind a TLS-intercepting corporate proxy that request can
# hang for several seconds before failing, and since this module sits at the
# top of wells.tui's import chain, that delay landed before the app painted
# anything at all. Load it in a background thread instead: estimates use the
# char-count fallback (the estimator self-calibrates against real API usage
# anyway — see module docstring) until the encoder is ready, then switch over
# transparently.
_ENC = None
_ENC_LOCK = threading.Lock()


def _load_encoder() -> None:
    global _ENC
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return
    with _ENC_LOCK:
        _ENC = enc


threading.Thread(target=_load_encoder, name="wells-tiktoken-load", daemon=True).start()

_CHARS_PER_TOKEN_FALLBACK = 3.5

# Single-element holder so calibration can be updated from anywhere.
_CALIBRATION = [1.0]


def _raw_estimate(text: str) -> int:
    """Uncalibrated token estimate (tiktoken if ready, else char heuristic)."""
    if not text:
        return 0
    enc = _ENC
    if enc is not None:
        return len(enc.encode(text))
    return max(1, int(round(len(text) / _CHARS_PER_TOKEN_FALLBACK)))


def estimate_tokens(text: str) -> int:
    """Calibrated token estimate for a chunk of text."""
    return max(1, int(round(_raw_estimate(text) * _CALIBRATION[0])))


def calibrate(text: str, actual_input_tokens: int) -> None:
    """Blend the estimator toward the provider's real tokenization.

    Called from :func:`wells.runtime.run_step` after every model call,
    using the actual ``input_tokens`` reported in ``usage_metadata``.
    """
    raw = _raw_estimate(text)
    if raw <= 0 or actual_input_tokens <= 0:
        return
    observed = actual_input_tokens / raw
    _CALIBRATION[0] = round(0.7 * _CALIBRATION[0] + 0.3 * observed, 3)


@dataclass
class TokenBudget:
    """Per-call token budget. Trimming targets ``max_input_tokens`` minus the
    output reservation so the model always has room to answer."""

    max_input_tokens: int = 24000
    reserved_output_tokens: int = 4000

    @property
    def input_allowance(self) -> int:
        return max(0, self.max_input_tokens - self.reserved_output_tokens)


@dataclass
class StepUsage:
    step: str
    task_type: str
    model: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_tokens: int
    category_tokens: dict
    saved_by_trim: int
    saved_by_summary: int


class TokenLedger:
    """Thread-safe accumulator of per-step token usage."""

    def __init__(self) -> None:
        self.steps: list[StepUsage] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.steps.clear()

    def record(
        self,
        *,
        step: str,
        task_type: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
        cache_read_tokens: int = 0,
        category_tokens: dict | None = None,
        saved_by_trim: int = 0,
        saved_by_summary: int = 0,
    ) -> None:
        with self._lock:
            self.steps.append(
                StepUsage(
                    step=step,
                    task_type=task_type,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                    cache_read_tokens=cache_read_tokens,
                    category_tokens=dict(category_tokens or {}),
                    saved_by_trim=saved_by_trim,
                    saved_by_summary=saved_by_summary,
                )
            )

    def totals(self) -> dict:
        with self._lock:
            steps = list(self.steps)
        if not steps:
            return {
                "input": 0,
                "output": 0,
                "reasoning": 0,
                "cache_read": 0,
                "saved_trim": 0,
                "saved_summary": 0,
                "calls": 0,
            }
        return {
            "input": sum(s.input_tokens for s in steps),
            "output": sum(s.output_tokens for s in steps),
            "reasoning": sum(s.reasoning_tokens for s in steps),
            "cache_read": sum(s.cache_read_tokens for s in steps),
            "saved_trim": sum(s.saved_by_trim for s in steps),
            "saved_summary": sum(s.saved_by_summary for s in steps),
            "calls": len(steps),
        }

    def category_totals(self) -> dict:
        agg: dict = {}
        with self._lock:
            steps = list(self.steps)
        for s in steps:
            for cat, n in s.category_tokens.items():
                agg[cat] = agg.get(cat, 0) + n
        return dict(sorted(agg.items(), key=lambda kv: kv[1], reverse=True))

    def format_report(self) -> str:
        t = self.totals()
        if t["calls"] == 0:
            return "(no model calls recorded)"

        lines: list[str] = []
        bar = "=" * 99
        lines.append(bar)
        lines.append("TOKEN USAGE REPORT")
        lines.append(bar)
        header = (
            f"{'STEP':<20} {'TASK':<14} {'IN':>7} {'OUT':>6} {'REASON':>7} "
            f"{'CACHE':>6} {'CACHE%':>6} {'SAVE.TR':>8} {'SAVE.SU':>8}"
        )
        lines.append(header)
        lines.append("-" * 99)
        with self._lock:
            steps = list(self.steps)
        for s in steps:
            denom = s.input_tokens + s.cache_read_tokens
            eff = (s.cache_read_tokens / denom) if denom > 0 else 0.0
            lines.append(
                f"{s.step:<20} {s.task_type:<14} {s.input_tokens:>7} "
                f"{s.output_tokens:>6} {s.reasoning_tokens:>7} "
                f"{s.cache_read_tokens:>6} {eff:>5.0%} {s.saved_by_trim:>8} {s.saved_by_summary:>8}"
            )
        lines.append("-" * 99)
        denom = t["input"] + t["cache_read"]
        overall_eff = (t["cache_read"] / denom) if denom > 0 else 0.0
        lines.append(
            f"{'TOTAL (' + str(t['calls']) + ' calls)':<20} {'':<14} "
            f"{t['input']:>7} {t['output']:>6} {t['reasoning']:>7} "
            f"{t['cache_read']:>6} {overall_eff:>5.0%} {t['saved_trim']:>8} {t['saved_summary']:>8}"
        )
        lines.append("")

        grand = t["input"] + t["output"]
        lines.append(f"Grand total tokens billed (in+out): {grand:,}")
        lines.append(
            f"  - reasoning tokens (model-internal): {t['reasoning']:,}  "
            f"({(t['reasoning'] / grand * 100):.0f}% of total)"
        )
        if t["cache_read"]:
            lines.append(
                f"  - prompt-cache hits (saved at provider): {t['cache_read']:,}"
            )
        total_saved = t["saved_trim"] + t["saved_summary"]
        if total_saved:
            pct = total_saved / (t["input"] + total_saved) * 100
            lines.append(
                f"Estimated input tokens avoided: {total_saved:,}  "
                f"(trim {t['saved_trim']:,} + summary {t['saved_summary']:,}) ~{pct:.0f}% of sent input"
            )
        lines.append("")

        cat = self.category_totals()
        if cat:
            lines.append("Estimated input-token spend by category:")
            cat_total = sum(cat.values()) or 1
            for name, n in cat.items():
                lines.append(f"  {name:<26} {n:>7,}  ({n / cat_total * 100:4.1f}%)")
        lines.append(bar)
        return "\n".join(lines)


# Process-wide singleton; main() resets it at the start of each run.
LEDGER = TokenLedger()
