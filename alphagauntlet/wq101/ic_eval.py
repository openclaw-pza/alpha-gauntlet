#!/usr/bin/env python3
"""WQ101 IC evaluation — math transcribed verbatim from factor_eval.evaluate_factor, not a self-invented simplification.

Reuse contract (identical to factor_eval.py):
- horizon=24 as the main (default); non-overlapping sampling iloc[::h], sliced after aligning on the common DatetimeIndex.
- Rank IC = within-row (cross-sectional) rank Pearson = Spearman (vectorized).
- t-stat is shrunk via an n_eff correction from the IC-series lag-1 autocorrelation.
- Insufficient-sample gate: valid sections < 20 -> verdict='insufficient samples'.
- Per-section valid-symbol gate: this module uses the WQ101 degenerate guard MIN_COINS_SECTION=8 (stricter
  than factor_eval's MIN_COINS=4); a cross-section with near-zero rank variance is also skipped.

Input: a factor wide frame (index=time, columns=symbols) + a close wide frame. forward return is computed
internally with panel_io.forward_return_panel (consistent with factor_eval._forward_return).
"""
import numpy as np

from alphagauntlet.wq101 import panel_io

# Main horizon, consistent with the main evaluation
DEFAULT_H = 24
HORIZONS = [1, 4, 24]

MIN_COINS_SECTION = 8     # WQ101 degenerate guard: skip a section with < 8 valid symbols (stricter than factor_eval's 4)
SKIP_FRAC_DEGEN = 0.50    # a factor with > 50% of timestamps skipped -> DEGENERATE
N_PERIODS_MIN = 20        # consistent with factor_eval: valid sections < 20 counts as insufficient samples


def _verdict(abs_ic, abs_t):
    """Grading — transcribed verbatim from factor_eval._verdict."""
    if abs_t < 2.0:
        return "noise (t<2, not even in-sample significant, do not use)"
    if abs_ic >= 0.05 and abs_t >= 5.0:
        return "strong (|IC|>=0.05 and |t|>=5, in-sample significant, worth validating)"
    if abs_ic >= 0.03:
        return "medium (weak predictive power, combine with other factors)"
    return "weak (significant but |IC| too small, limited practical value)"


def section_ic_series(fac_wide, close_wide, h=DEFAULT_H):
    """Return the per-section Rank IC series (np.ndarray) after non-overlapping sampling, plus diagnostics.

    Strictly replicates the sampling and IC computation of factor_eval.evaluate_factor, only raising the
    per-section valid-symbol gate from 4 to 8, and counting skipped sections (zero-variance / insufficient
    valid symbols) for the DEGENERATE decision.

    Returns dict:
        ics            : np.ndarray, per-section IC (already dropna'd)
        ics_index      : pd.DatetimeIndex, the sampling times aligned with ics (for sub-windowing/yearly splits)
        n_sampled      : total sections after non-overlapping sampling (after alignment, before filtering)
        n_skipped      : number of sections skipped (valid symbols<8 or cross-sectional zero variance)
        skip_frac      : n_skipped / n_sampled
    """
    fwd_wide = panel_io.forward_return_panel(close_wide, h)
    common = fac_wide.index.intersection(fwd_wide.index)
    fac_mat = fac_wide.loc[common].iloc[::h]      # non-overlapping sampling (step = h)
    fwd_mat = fwd_wide.loc[common].iloc[::h]

    valid = fac_mat.notna() & fwd_mat.notna()
    fac_v = fac_mat.where(valid)
    fwd_v = fwd_mat.where(valid)

    n_sampled = len(fac_v)
    # within-row (cross-sectional) rank -> rank Pearson = Rank IC
    fr = fac_v.rank(axis=1)
    rr = fwd_v.rank(axis=1)
    fc = fr.sub(fr.mean(axis=1), axis=0)
    rc = rr.sub(rr.mean(axis=1), axis=0)
    num = (fc * rc).sum(axis=1)
    den = np.sqrt((fc ** 2).sum(axis=1) * (rc ** 2).sum(axis=1))
    ic_raw = (num / den).replace([np.inf, -np.inf], np.nan)

    # WQ101 degenerate guard: skip sections with valid symbols < 8 or cross-sectional zero variance (den≈0)
    eff_coins = valid.sum(axis=1)
    keep = (eff_coins >= MIN_COINS_SECTION) & (den > 1e-12) & ic_raw.notna()
    ics_series = ic_raw[keep]
    n_skipped = int(n_sampled - keep.sum())
    skip_frac = (n_skipped / n_sampled) if n_sampled else 1.0

    return {
        "ics": ics_series.to_numpy(),
        "ics_index": ics_series.index,
        "n_sampled": int(n_sampled),
        "n_skipped": n_skipped,
        "skip_frac": float(skip_frac),
    }


def _t_from_ics(ics):
    """Compute mean_ic, icir, n_eff, t_stat from the IC series — transcribed verbatim from factor_eval."""
    n = len(ics)
    mean_ic = float(ics.mean())
    std_ic = float(ics.std(ddof=1))
    if std_ic > 1e-9:
        icir = mean_ic / std_ic
    else:
        icir = float(np.sign(mean_ic) * 10.0) if abs(mean_ic) > 1e-9 else 0.0
    if n >= 3:
        ic_c = ics - ics.mean()
        denom = float((ic_c ** 2).sum())
        rho = float((ic_c[:-1] * ic_c[1:]).sum() / denom) if denom > 1e-12 else 0.0
        rho = max(-0.99, min(0.99, rho))
        n_eff = n * (1.0 - rho) / (1.0 + rho)
        n_eff = max(2.0, min(float(n), n_eff))
    else:
        n_eff = float(n)
    t_stat = icir * np.sqrt(n_eff)
    return mean_ic, std_ic, icir, n_eff, t_stat


def evaluate(fac_wide, close_wide, h=DEFAULT_H):
    """Full-sample evaluation of a single factor at horizon h. Returns a dict isomorphic to
    factor_eval.evaluate_factor (plus WQ101 degenerate fields). The math is identical, only the per-section
    valid-symbol gate is 8 plus a degenerate guard.
    """
    sec = section_ic_series(fac_wide, close_wide, h)
    ics = sec["ics"]
    n = len(ics)
    degenerate = sec["skip_frac"] > SKIP_FRAC_DEGEN

    if n < N_PERIODS_MIN:
        return {
            "horizon_h": h, "n_periods": int(n), "verdict": "insufficient samples",
            "mean_ic": None, "t_stat": None, "n_eff": None,
            "n_sampled": sec["n_sampled"], "n_skipped": sec["n_skipped"],
            "skip_frac": round(sec["skip_frac"], 3),
            "degenerate": bool(degenerate),
        }

    mean_ic, std_ic, icir, n_eff, t_stat = _t_from_ics(ics)
    verdict = "DEGENERATE" if degenerate else _verdict(abs(mean_ic), abs(t_stat))
    return {
        "horizon_h": h, "n_periods": int(n), "n_eff": round(n_eff, 1),
        "mean_ic": round(mean_ic, 4), "std_ic": round(std_ic, 4),
        "icir": round(icir, 3), "t_stat": round(t_stat, 2),
        "ic_positive_rate": round(float((ics > 0).mean()), 3),
        "abs_mean_ic": round(abs(mean_ic), 4),
        "direction": ("positive (factor up -> future up)" if mean_ic > 0
                      else "negative (factor up -> future down, use inverted)"),
        "n_sampled": sec["n_sampled"], "n_skipped": sec["n_skipped"],
        "skip_frac": round(sec["skip_frac"], 3),
        "degenerate": bool(degenerate),
        "verdict": verdict,
    }
