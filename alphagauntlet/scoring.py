#!/usr/bin/env python3
"""Scoring channel — scorecard-derived empirical weights x live factor values.

Selection moves from a legacy 4-bucket (roc/vol/funding/liq hand-set) scheme to a **gauntlet-validated full
factor set**: factors in the scorecard (factor_scores.json) whose h24 |t| >= T_MIN, including the adopted WQ101
factors and the adopted LLM-generated factors. The weight = mean_ic (the sign is the direction, same philosophy
as factor_eval.ic_weights).

Pre-registered rules (no tunable knobs; changing a rule = changing code, leaving a trace):
- Selection: h24 |t_stat| >= T_MIN (3.5, same threshold as the gauntlet R1) and non-composite
  (a composite factor is a rank average of its components; selecting it together with its components would double-count).
- Weight: w = mean_ic (h24); score = Σ w_i × (pct_i - 50)/50, normalized by the Σ|w| over the factors each
  symbol actually computed (so short, still-warming symbols are not structurally penalized).
- Factor coverage guard: a factor computed on < 50% of candidate symbols -> dropped this round;
  usable factors < MIN_FACTORS (4) -> give up entirely (the caller falls back to legacy).
- Freshness guard: scorecard generated_at older than feval.MAX_AGE_H -> return None (a stale IC must not place live orders).

Layering: this module **only scores, never places orders**; order placement still goes through the single
choke point. fail-soft: any failure returns None, the caller falls back to the legacy path and warns — a
scoring upgrade must never become a new "can't compute -> halt" lock.

Data convention: uses the upstream-passed **live closed** klines (~299 bars, covering the deepest lookback);
it does not read local feather (feather refreshes roughly twice a day; a stale value should not place a live order).
"""
import os

import numpy as np
import pandas as pd

from alphagauntlet import factor_eval as feval

T_MIN = 3.5
HORIZON = "h24"
MIN_FACTOR_COVERAGE = 0.5   # a factor must cover at least 50% of candidate symbols
MIN_FACTORS = 4             # lower bound on usable factor count; below this fall back to legacy
MAX_AGE_D = 14.0            # weights file older than 14 days -> treated as stale, fall back to legacy (full-history IC barely moves weekly)
WQ_NAMES = ("wq_a005", "wq_a020", "wq_a024", "wq_a077")
LLM1_NAMES = ("llm1_4_05", "llm1_3_07", "llm1_2_08")
R2_NAMES = ("r2v2_01_02", "r2v2_03_04", "r2v2_05_03", "r2v2_06_01")
PANEL_NAMES = WQ_NAMES + LLM1_NAMES + R2_NAMES   # cross-sectional formula factors (need the whole-pool wide frame, computed separately from per-symbol time-domain factors)
_ADV_NS = (15, 20, 30, 40, 50, 60, 120, 180)   # same as panel_io.load_field_panel
# Weights file path is configurable: ALPHAGAUNTLET_WEIGHTS_PATH, else ./state/learning_weights.json.
WEIGHTS_PATH = os.environ.get(
    "ALPHAGAUNTLET_WEIGHTS_PATH", os.path.join(".", "state", "learning_weights.json"))
# Why not read factor_scores.json directly: a production scorecard may be periodically overwritten by a
# short-window evaluation (n_tail, only ~1.4 years) -> under a short window, t>=3.5 leaves only 2-3 volatility
# factors and the gauntlet's full-history-validated factor set gets washed out. Scoring weights must be pinned
# to a **full-history** evaluation: only a full-history score_all(n_tail=None) refreshes this file.


def _panel_fns():
    """Cross-sectional formula factor function table (same source as factor_eval._inject_wq101/_inject_llm1).
    Import failure is fail-soft at the call site."""
    from alphagauntlet.wq101 import batch1, batch2, batch4, batch7
    from alphagauntlet.wq101 import llm_round2_adopted as r2
    return {"wq_a005": batch1.alpha_wq005, "wq_a020": batch1.alpha_wq020,
            "wq_a024": batch2.alpha_024, "wq_a077": batch4.alpha_077,
            "llm1_4_05": batch7.alpha_llm1_4_05, "llm1_3_07": batch7.alpha_llm1_3_07,
            "llm1_2_08": batch7.alpha_llm1_2_08, **r2.ALPHAS}


def select_weights(report):
    """From a **full-history** score_all report dict, select scoring weights: non-composite factors with h24 |t|>=T_MIN, w=mean_ic."""
    out = {}
    if not isinstance(report, dict) or not report.get("ok"):
        return out
    for name, meta in (report.get("factors") or {}).items():
        if name in feval.COMPOSITES:
            continue
        m = ((meta or {}).get("by_horizon") or {}).get(HORIZON) or {}
        t, ic = m.get("t_stat"), m.get("mean_ic")
        if t is None or ic is None:
            continue
        if abs(float(t)) >= T_MIN and float(ic) != 0.0:
            out[name] = float(ic)
    return out


def export_weights(report, path=None):
    """Full-history score_all report -> write state/learning_weights.json (atomic replace). Returns the written doc or None.
    Called by factor_eval.score_all(n_tail=None, save=True) after saving the scorecard; a short-window run never calls it."""
    w = select_weights(report)
    if not w:
        return None
    doc = {"ok": True, "source": "score_all_full", "t_min": T_MIN, "horizon": HORIZON,
           "generated_at": report.get("generated_at"), "data_through": report.get("data_through"),
           "weights": w}
    path = path or WEIGHTS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import json
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return doc


def gauntlet_weights(doc=None, path=None):
    """Read scoring weights {factor: mean_ic}. Missing file / stale (>MAX_AGE_D days) / empty -> None (caller falls back to legacy)."""
    if doc is None:
        import json
        p = path or WEIGHTS_PATH
        if not os.path.exists(p):
            return None
        try:
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(doc, dict) or not doc.get("ok"):
        return None
    gen = doc.get("generated_at")
    if not gen:
        return None
    try:
        import datetime as _dt
        ts = _dt.datetime.fromisoformat(str(gen).replace("Z", ""))
        age_d = (_dt.datetime.utcnow() - ts).total_seconds() / 86400.0
        if age_d > MAX_AGE_D:
            return None
    except Exception:  # noqa: BLE001 broken timestamp -> treat as stale
        return None
    w = doc.get("weights")
    if not isinstance(w, dict) or not w:
        return None
    return {k: float(v) for k, v in w.items() if isinstance(v, (int, float)) and float(v) != 0.0} or None


def _pct_ranks(d):
    """{sym: val} -> {sym: cross-sectional percentile 0..100} (median ties split evenly)."""
    syms = list(d)
    a = np.array([d[s] for s in syms], dtype=float)
    out = {}
    for i, s in enumerate(syms):
        v = a[i]
        out[s] = round((float((a < v).sum()) + 0.5 * float((a == v).sum())) / len(a) * 100.0, 1)
    return out


def factor_values_live(dfs, want):
    """Live factor last-row values. dfs={sym: closed OHLCV df (with a date column)}, want=set of factor names needed.
    Returns {factor: {sym: value}} (NaN / uncomputable skipped; a single-symbol/single-factor failure does not affect the rest)."""
    vals = {}
    want = set(want)
    td_want = want - set(PANEL_NAMES)
    # —— time-domain + wavelet: per-symbol compute_factors take the last row ——
    if td_want:
        for sym, df in dfs.items():
            try:
                d = df.set_index("date") if "date" in df.columns else df
                fac = feval.compute_factors(d)
                for f in td_want:
                    if f in fac.columns:
                        v = fac[f].iloc[-1]
                        if pd.notna(v) and np.isfinite(v):
                            vals.setdefault(f, {})[sym] = float(v)
            except Exception:  # noqa: BLE001 a single-symbol failure does not affect the rest
                continue
    # —— cross-sectional formula factors (WQ101 + adopted LLM): hand-build the wide-frame fields (same as panel_io) and compute over the whole pool at once ——
    wq_want = [f for f in want if f in PANEL_NAMES]
    if wq_want:
        try:
            fns = _panel_fns()
            base = {}
            for field in ("open", "high", "low", "close", "volume"):
                base[field] = pd.DataFrame({
                    sym: (df.set_index("date") if "date" in df.columns else df)[field].astype(float)
                    for sym, df in dfs.items()}).sort_index()
            P = dict(base)
            P["vwap"] = (P["high"] + P["low"] + P["close"]) / 3.0
            P["returns"] = P["close"].pct_change()
            for n in _ADV_NS:
                P[f"adv{n}"] = P["volume"].rolling(n, min_periods=n).mean() * P["close"]
            for f in wq_want:
                try:
                    wide = fns[f](P).replace([np.inf, -np.inf], np.nan)
                    last = wide.iloc[-1]
                    for sym, v in last.items():
                        if pd.notna(v):
                            vals.setdefault(f, {})[sym] = float(v)
                except Exception:  # noqa: BLE001 a single-factor failure does not affect the rest
                    continue
        except Exception:  # noqa: BLE001 wq101 package unavailable -> only time-domain factors participate
            pass
    return vals


def score_pool(dfs, weights):
    """Score a candidate pool by the scorecard weights.
    Returns (rows, used, dropped): rows=[{symbol, score, top_factor, n_factors}] sorted by score descending;
    used={factor: w} factors that actually participated; dropped=[factors dropped for insufficient coverage].
    usable factors < MIN_FACTORS -> (None, used, dropped), caller falls back to legacy."""
    syms = list(dfs)
    vals = factor_values_live(dfs, set(weights))
    used, dropped = {}, []
    need = max(2, int(np.ceil(len(syms) * MIN_FACTOR_COVERAGE)))
    for f, w in weights.items():
        if len(vals.get(f) or {}) >= need:
            used[f] = w
        else:
            dropped.append(f)
    if len(used) < MIN_FACTORS:
        return None, used, dropped
    contrib = {s: {} for s in syms}
    for f, w in used.items():
        pcts = _pct_ranks(vals[f])
        for s, p in pcts.items():
            contrib[s][f] = w * (p - 50.0) / 50.0
    rows = []
    for s in syms:
        c = contrib[s]
        if len(c) < MIN_FACTORS:        # this symbol is missing too many factors (warmup/data holes); skip this round
            continue
        wsum = sum(abs(used[f]) for f in c) or 1.0
        rows.append({"symbol": s, "score": round(sum(c.values()) / wsum, 3),
                     "top_factor": max(c, key=lambda k: abs(c[k])), "n_factors": len(c)})
    rows.sort(key=lambda x: x["score"], reverse=True)
    return (rows or None), used, dropped
