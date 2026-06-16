#!/usr/bin/env python3
# Ported from QuantaAlpha (https://github.com/QuantaAlpha/QuantaAlpha), MIT License, Copyright (c) Ziyi Tang et al.
"""reskin screen — max-common-subtree similarity matrix of candidate factors vs incumbent factors.

Role: before candidate factors enter the IC gauntlet, use AST structural similarity as a **pure early
warning** filter, flagging "suspected reskin" (high structural overlap) candidates for human / downstream
review. This filter **only warns, never decides** — the decision over a factor's truth / retention always
belongs to the IC gauntlet (factor_eval / scorer); structural similarity is only a clue.

Usage:
    # text file: one candidate per line, "name = expression" or a bare expression
    python -m alphagauntlet.wq101.reskin_screen --candidates candidates.txt --threshold 0.6

    # json file: {"name": "expression", ...} or ["expression", ...]
    python -m alphagauntlet.wq101.reskin_screen --candidates candidates.json --json-out report.json

Similarity semantics see dsl.similarity: max_common_subtree / min(size1, size2) ∈ [0,1].

Read-only reference discipline: this module modifies no existing module; only depends on the same-package dsl.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional, Tuple

try:  # in-package run
    from alphagauntlet.wq101 import dsl
except ImportError:  # fallback for direct-script run
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from alphagauntlet.wq101 import dsl


# --------------------------------------------------------------------------- #
# Incumbent registry (19 entries): name -> expression string.
# ou_rev_press has no pure-DSL form (OU AR(1) residual mean reversion), so it is marked with a sentinel
# and skipped in structural matching (reported as N/A).
# Composite factors (mom/vol/meta_combo) are cross-sectional equal-weight rank composites, RANK_CS(subfactor)/N
# summed; the subfactor names and N parse as bare-identifier VarNodes in the DSL, so they are structurally comparable.
# --------------------------------------------------------------------------- #
NO_DSL = "__NO_DSL__"

INCUMBENTS: Dict[str, str] = {
    "mom_24h": "DELTA($close, 24) / $close",
    "mom_72h": "DELTA($close, 72) / $close",
    "mom_168h": "DELTA($close, 168) / $close",
    "rev_6h": "MULTIPLY(-1, DELTA($close, 6) / $close)",
    "vol_24h": "TS_STD(TS_PCTCHANGE($close, 1), 24)",
    "vol_72h": "TS_STD(TS_PCTCHANGE($close, 1), 72)",
    "liq_ratio": "TS_MEAN($volume, 24) / (TS_MEAN($volume, 72) + 1e-12)",
    "rsi_14": "RSI($close, 14)",
    "atr_pct": "ATR($high, $low, $close, 14) / ($close + 1e-12) * 100",
    "dist_ema200": "$close / (EMA($close, 200) + 1e-12) - 1",
    "ema_slope": "(EMA($close, 20) - EMA($close, 50)) / (EMA($close, 50) + 1e-12)",
    "vol_price": (
        "MULTIPLY(DELTA($close, 24) / $close, "
        "(TS_MEAN($volume, 24) / (TS_MEAN($volume, 96) + 1e-12) - 1))"
    ),
    "amihud_illiq_168h": (
        "LOG(TS_MEAN(ABS(TS_PCTCHANGE($close, 1)) / ($volume * $close + 1e-12), 168) + 1e-15)"
    ),
    "downside_var_ratio_168h": (
        "TS_SUM(WHERE(LT(DELTA(LOG($close), 1), 0), POW(DELTA(LOG($close), 1), 2), 0), 168) "
        "/ (TS_SUM(POW(DELTA(LOG($close), 1), 2), 168) + 1e-12)"
    ),
    "tail_ratio_168h": (
        "TS_QUANTILE($ret, 168, 0.95) / (ABS(TS_QUANTILE($ret, 168, 0.05)) + 1e-12)"
    ),
    "ou_rev_press": NO_DSL,
    "mom_combo": "RANK_CS(mom_24h) / N + RANK_CS(mom_72h) / N + RANK_CS(mom_168h) / N",
    "vol_combo": "RANK_CS(vol_24h) / N + RANK_CS(vol_72h) / N + RANK_CS(atr_pct) / N",
    "meta_combo": (
        "RANK_CS(vol_24h) / N + RANK_CS(vol_72h) / N + RANK_CS(atr_pct) / N "
        "+ RANK_CS(mom_24h) / N + RANK_CS(mom_72h) / N + RANK_CS(mom_168h) / N"
    ),
}

DEFAULT_THRESHOLD = 0.6


# --------------------------------------------------------------------------- #
# Core scoring
# --------------------------------------------------------------------------- #
def _safe_similarity(expr_a: str, expr_b: str) -> Optional[float]:
    """Similarity of two expressions; if either is the NO_DSL sentinel -> None (skip). Parse failure raises ValueError upward."""
    if expr_a == NO_DSL or expr_b == NO_DSL:
        return None
    return dsl.similarity(expr_a, expr_b)


def screen_candidate(
    cand_name: str,
    cand_expr: str,
    incumbents: Dict[str, str] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """One candidate vs all incumbents: similarities + suspected-reskin flag (warning only).

    Returns dict:
      name, expr, parse_error (optional),
      vs_incumbents: {incumbent_name: sim or None},
      max_sim, max_match_incumbent, flagged(bool), flagged_incumbents(list)
    """
    incumbents = INCUMBENTS if incumbents is None else incumbents
    out = {
        "name": cand_name,
        "expr": cand_expr,
        "vs_incumbents": {},
        "max_sim": 0.0,
        "max_match_incumbent": None,
        "flagged": False,
        "flagged_incumbents": [],
    }
    # First confirm the candidate itself parses.
    try:
        dsl.parse(cand_expr)
    except ValueError as e:
        out["parse_error"] = str(e)
        return out

    max_sim = -1.0
    max_inc = None
    for inc_name, inc_expr in incumbents.items():
        sim = _safe_similarity(cand_expr, inc_expr)
        out["vs_incumbents"][inc_name] = sim
        if sim is None:
            continue
        if sim > max_sim:
            max_sim = sim
            max_inc = inc_name
        if sim >= threshold:
            out["flagged_incumbents"].append((inc_name, sim))
    out["max_sim"] = max(max_sim, 0.0)
    out["max_match_incumbent"] = max_inc
    out["flagged"] = len(out["flagged_incumbents"]) > 0
    out["flagged_incumbents"].sort(key=lambda kv: kv[1], reverse=True)
    return out


def candidate_cross_matrix(
    candidates: Dict[str, str], threshold: float = DEFAULT_THRESHOLD
) -> Tuple[List[str], List[List[Optional[float]]], List[Tuple[str, str, float]]]:
    """Candidates vs each other similarity matrix + suspected-reskin candidate pairs (>=threshold).

    Returns (names, matrix, flagged_pairs). matrix[i][j] = sim(names[i], names[j]), diagonal = 1.0,
    parse-failure cell = None.
    """
    names = list(candidates.keys())
    n = len(names)
    matrix: List[List[Optional[float]]] = [[None] * n for _ in range(n)]
    flagged_pairs: List[Tuple[str, str, float]] = []
    parsed_ok = {}
    for nm in names:
        try:
            dsl.parse(candidates[nm])
            parsed_ok[nm] = True
        except ValueError:
            parsed_ok[nm] = False
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0 if parsed_ok[names[i]] else None
                continue
            if not (parsed_ok[names[i]] and parsed_ok[names[j]]):
                matrix[i][j] = None
                continue
            sim = dsl.similarity(candidates[names[i]], candidates[names[j]])
            matrix[i][j] = sim
            if i < j and sim >= threshold:
                flagged_pairs.append((names[i], names[j], sim))
    flagged_pairs.sort(key=lambda t: t[2], reverse=True)
    return names, matrix, flagged_pairs


def run_screen(
    candidates: Dict[str, str], threshold: float = DEFAULT_THRESHOLD
) -> dict:
    """Full pre-screen: each candidate vs the 19 incumbents + candidates vs each other matrix. Returns a structured report dict."""
    per_candidate = [
        screen_candidate(name, expr, INCUMBENTS, threshold)
        for name, expr in candidates.items()
    ]
    cross_names, cross_matrix, cross_flagged = candidate_cross_matrix(candidates, threshold)
    return {
        "threshold": threshold,
        "n_incumbents": len(INCUMBENTS),
        "n_incumbents_dsl": sum(1 for v in INCUMBENTS.values() if v != NO_DSL),
        "n_candidates": len(candidates),
        "per_candidate": per_candidate,
        "candidate_cross": {
            "names": cross_names,
            "matrix": cross_matrix,
            "flagged_pairs": cross_flagged,
        },
        "verdict_note": (
            "This table only warns of structural reskin clues; it does not decide retention. Final keep/drop is decided by the IC gauntlet."
        ),
    }


# --------------------------------------------------------------------------- #
# Input parsing
# --------------------------------------------------------------------------- #
def load_candidates(path: str) -> Dict[str, str]:
    """Load candidates from a json or text file.

    json: supports {"name": "expression"} or ["expression", ...] (the latter auto-named cand_000..).
    text: one per line; supports "name = expression" or a bare expression; '#'-leading lines and blank lines ignored.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    stripped = raw.lstrip()
    if path.lower().endswith(".json") or stripped[:1] in "{[":
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        if isinstance(data, list):
            return {f"cand_{i:03d}": str(v) for i, v in enumerate(data)}
        raise ValueError(f"Unsupported json shape in {path}: {type(data)}")
    # text
    out: Dict[str, str] = {}
    for i, line in enumerate(raw.splitlines()):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" in s and not _looks_like_comparison(s):
            name, expr = s.split("=", 1)
            out[name.strip()] = expr.strip()
        else:
            out[f"cand_{len(out):03d}"] = s
    return out


def _looks_like_comparison(s: str) -> bool:
    """Roughly judge whether a line's '=' is part of ==/<=/>=/!= comparison (vs a name=expression assignment)."""
    idx = s.find("=")
    if idx <= 0:
        return False
    prev = s[idx - 1]
    nxt = s[idx + 1] if idx + 1 < len(s) else ""
    return prev in "<>!=" or nxt == "="


# --------------------------------------------------------------------------- #
# Text report rendering
# --------------------------------------------------------------------------- #
def _fmt_sim(v: Optional[float]) -> str:
    return " N/A " if v is None else f"{v:5.3f}"


def render_text_report(report: dict) -> str:
    lines: List[str] = []
    thr = report["threshold"]
    lines.append("=" * 72)
    lines.append("reskin screen report — warning only, no decision (decision belongs to the IC gauntlet)")
    lines.append("=" * 72)
    lines.append(
        f"threshold={thr}  incumbents={report['n_incumbents']} "
        f"(parseable {report['n_incumbents_dsl']}, ou_rev_press has no DSL, skipped)  "
        f"candidates={report['n_candidates']}"
    )
    lines.append("")
    lines.append("[candidate vs 19 incumbents] each candidate's top hit + whether it triggers a warning")
    lines.append("-" * 72)
    lines.append(f"{'candidate':<14}{'max sim':>10}  {'most similar incumbent':<24}{'warn':>6}")
    for pc in report["per_candidate"]:
        if pc.get("parse_error"):
            lines.append(f"{pc['name']:<14}{'parse fail':>10}  {pc['parse_error'][:40]}")
            continue
        flag = "yes" if pc["flagged"] else "no"
        lines.append(
            f"{pc['name']:<14}{pc['max_sim']:>10.3f}  "
            f"{str(pc['max_match_incumbent']):<24}{flag:>6}"
        )
        if pc["flagged"]:
            for inc, sim in pc["flagged_incumbents"]:
                lines.append(f"    +- warn vs {inc}: {sim:.3f} (>= {thr})")
    lines.append("")
    lines.append("[candidate vs 19 incumbents] full similarity matrix")
    lines.append("-" * 72)
    inc_names = list(INCUMBENTS.keys())
    # transposed display: rows=candidates, columns=incumbents (incumbent names abbreviated)
    short = {nm: nm[:8] for nm in inc_names}
    header = f"{'cand\\inc':<14}" + "".join(f"{short[n]:>9}" for n in inc_names)
    lines.append(header)
    for pc in report["per_candidate"]:
        if pc.get("parse_error"):
            continue
        row = f"{pc['name']:<14}"
        for n in inc_names:
            row += f"{_fmt_sim(pc['vs_incumbents'].get(n)):>9}"
        lines.append(row)
    lines.append("")
    lines.append("[candidate vs candidate] mutual similarity matrix")
    lines.append("-" * 72)
    cnames = report["candidate_cross"]["names"]
    cmat = report["candidate_cross"]["matrix"]
    if cnames:
        header = f"{'':<14}" + "".join(f"{c[:8]:>9}" for c in cnames)
        lines.append(header)
        for i, cn in enumerate(cnames):
            row = f"{cn:<14}" + "".join(f"{_fmt_sim(cmat[i][j]):>9}" for j in range(len(cnames)))
            lines.append(row)
    cf = report["candidate_cross"]["flagged_pairs"]
    if cf:
        lines.append("")
        lines.append(f"suspected-reskin candidate pairs (>= {thr}):")
        for a, b, sim in cf:
            lines.append(f"    {a} <-> {b}: {sim:.3f}")
    lines.append("")
    lines.append(report["verdict_note"])
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reskin_screen",
        description="candidate vs incumbent factor AST reskin pre-screen (warning only, no decision)",
    )
    p.add_argument(
        "--candidates", "-c", required=True,
        help="candidate file: .json ({name:expr} or [expr]) or .txt (one name=expr or bare expression per line)",
    )
    p.add_argument(
        "--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
        help=f"suspected-reskin warning threshold ∈[0,1], default {DEFAULT_THRESHOLD}",
    )
    p.add_argument(
        "--json-out", default=None,
        help="optional: write the structured report to this json path",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not (0.0 <= args.threshold <= 1.0):
        print(f"[error] threshold must be ∈[0,1], got {args.threshold}", file=sys.stderr)
        return 2
    try:
        candidates = load_candidates(args.candidates)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"[error] failed to load candidates {args.candidates}: {e}", file=sys.stderr)
        return 2
    if not candidates:
        print(f"[error] {args.candidates} parsed no candidates", file=sys.stderr)
        return 2

    report = run_screen(candidates, args.threshold)
    print(render_text_report(report))

    if args.json_out:
        # the matrix does not put SubtreeMatch into json; per_candidate's vs_incumbents is already pure numbers.
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[json] report written: {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
