#!/usr/bin/env python3
"""Factor evaluation (FactorEvaluator) — compute the cross-sectional predictive power (Rank-IC / ICIR / t-stat) of factors.

Methodology (borrowed from alphalens, not cloned — alphalens targets thousands of stocks; this is self-built for
a small 9-30 symbol cross-section):
- Rank IC: at each time section, the Spearman rank correlation of the factor value with the forward return
  (robust to outliers, unitless, the cross-sectional standard choice).
- ICIR = mean(IC)/std(IC); t-stat = ICIR×sqrt(n_periods). **To keep the t-stat valid, IC is sampled on
  non-overlapping windows (step = horizon)**, avoiding the spuriously inflated significance from overlapping
  forward windows' autocorrelation.
- Quantile spread: per section, top tercile mean return − bottom tercile (N=9 can only split into 3 layers).
- IC decay: compare multiple horizons (1h/4h/24h) to see a factor's effective horizon.

Small-cross-section honesty: at N=9~30, a single-section IC is extremely noisy (corr on 9 points has a standard
error ≈0.35). **A single-period IC is nearly meaningless; only the multi-period ICIR/t-stat is trustworthy**;
and t-stat>2 only means "in-sample significant", not future-effective. This tool gives a checkup of "which
factor's signal is relatively stronger", not a profit guarantee.
"""
import datetime
import json
import math
import os

import numpy as np
import pandas as pd
import talib

from alphagauntlet.regime import POOL          # the shared symbol universe (small caps included; symbols missing data are skipped)
from alphagauntlet import wavelet_factors       # time-frequency / OU factors (wavelet energy ratio + OU reversion pressure, anti-lookahead, see that module)

# Data directory and state directory are configurable via environment variables; no deployment path is hard-coded.
DATA_DIR = os.environ.get("ALPHAGAUNTLET_DATA_DIR", os.path.join(".", "data"))
STATE = os.environ.get("ALPHAGAUNTLET_STATE_DIR", os.path.join(".", "state"))
SCORES_PATH = os.path.join(STATE, "factor_scores.json")

HORIZONS = [1, 4, 24]      # forward return bars (1h/4h/1d on 1h tf)
MIN_COINS = 4              # minimum symbols in a section's IC computation (< this the IC is untrustworthy, skip the section)
N_QUANTILE = 3             # a small cross-section can only split into 3 layers (terciles)
MAX_AGE_H = 72             # a scorecard older than 72h is stale; ic_weights falls back to hand-set weights (prevent a stale IC misleading selection)


# ============ factor definitions (all computable directly from OHLCV, zero external data) ============
def compute_factors(df):
    """Compute each factor's time series for one symbol's OHLCV. Returns DataFrame(index=date, columns=factor names).
    Each factor has a brief 'directional intuition'; the actual direction is determined empirically by the IC sign (not preset)."""
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    low = df["low"].astype(float)
    v = df["volume"].astype(float)
    ca = c.to_numpy()
    out = pd.DataFrame(index=df.index)
    # momentum family (cross-sectional momentum is the strongest empirical factor in crypto, past winners keep winning)
    out["mom_24h"] = c.pct_change(24)
    out["mom_72h"] = c.pct_change(72)
    out["mom_168h"] = c.pct_change(168)
    # short-term reversal (ultra-short momentum often reverses)
    out["rev_6h"] = -c.pct_change(6)
    # volatility (low-vol vs high-vol premium, sign empirical)
    ret = c.pct_change()
    out["vol_24h"] = ret.rolling(24).std()
    out["vol_72h"] = ret.rolling(72).std()
    # volume / liquidity (the second-largest factor in Gu-Kelly-Xiu)
    out["liq_ratio"] = v.rolling(24).mean() / (v.rolling(72).mean() + 1e-12)
    # technical indicators
    out["rsi_14"] = pd.Series(talib.RSI(ca, 14), index=df.index)
    atr = talib.ATR(h.to_numpy(), low.to_numpy(), ca, 14)
    out["atr_pct"] = pd.Series(atr / (ca + 1e-12) * 100, index=df.index)
    ema200 = talib.EMA(ca, 200)
    out["dist_ema200"] = pd.Series(ca / (ema200 + 1e-12) - 1, index=df.index)
    ema20 = talib.EMA(ca, 20); ema50 = talib.EMA(ca, 50)
    out["ema_slope"] = pd.Series((ema20 - ema50) / (ema50 + 1e-12), index=df.index)
    # volume-price divergence proxy: return × volume change
    out["vol_price"] = c.pct_change(24) * (v.rolling(24).mean() / (v.rolling(96).mean() + 1e-12) - 1)
    # microstructure / distribution-shape family
    # Amihud (2002) illiquidity: price impact per dollar of volume, log-compressed across symbols' 6 orders of magnitude; zero-volume fake bars set NaN first to avoid division by zero
    dv = (v * c).replace(0, np.nan)
    illiq = ret.abs() / (dv + 1e-12)
    out["amihud_illiq_168h"] = np.log(illiq.rolling(168, min_periods=120).mean() + 1e-15)
    # realized semivariance share (Barndorff-Nielsen 2010): downside variance / total variance, ∈[0,1] naturally cross-symbol comparable (selling-pressure release -> rebound, positive)
    dlogp = np.log(c).diff()
    r2dn = dlogp.where(dlogp < 0, 0.0) ** 2
    out["downside_var_ratio_168h"] = r2dn.rolling(168, min_periods=120).sum() / (
        (dlogp ** 2).rolling(168, min_periods=120).sum() + 1e-12)
    # nonparametric tail ratio P95/|P05|: a fat right tail = lottery-type overpriced; rolling.quantile uses linear interpolation, same as the audit np.percentile
    q95 = ret.rolling(168, min_periods=120).quantile(0.95)
    q05 = ret.rolling(168, min_periods=120).quantile(0.05)
    out["tail_ratio_168h"] = q95 / (q05.abs() + 1e-12)
    # time-frequency / statistical-arbitrage family (wavelet energy ratio + OU reversion pressure, end-of-window scalar anti-lookahead, see wavelet_factors.py)
    wof = wavelet_factors.wavelet_ou_factors(df)
    for col in wof.columns:
        out[col] = wof[col]
    return out


FACTOR_NAMES = ["mom_24h", "mom_72h", "mom_168h", "rev_6h", "vol_24h", "vol_72h",
                "liq_ratio", "rsi_14", "atr_pct", "dist_ema200", "ema_slope", "vol_price",
                "amihud_illiq_168h", "downside_var_ratio_168h", "tail_ratio_168h"] \
    + wavelet_factors.SCORED_COLS   # + ou_rev_press (redundant lf_slope / low-vol proxy mr_speed excluded, see wavelet_factors.py)

FACTOR_DESC = {
    "mom_24h": "24h momentum (cross-sectional momentum, crypto's strongest empirical factor)",
    "mom_72h": "72h momentum (mid-term trend continuation)",
    "mom_168h": "weekly momentum (long momentum)",
    "rev_6h": "6h short-term reversal (ultra-short momentum often reverses, negated)",
    "vol_24h": "24h realized volatility (low/high-vol premium, sign empirical)",
    "vol_72h": "72h realized volatility",
    "liq_ratio": "liquidity warming (recent 24h mean volume / recent 72h baseline)",
    "rsi_14": "RSI14 (overbought/oversold)",
    "atr_pct": "ATR% (volatility regime)",
    "dist_ema200": "distance from EMA200 (trend position)",
    "ema_slope": "EMA20-50 slope (trend direction)",
    "vol_price": "volume-price agreement (momentum × volume change)",
    "amihud_illiq_168h": "Amihud illiquidity (|ret|/dollar volume 168h mean, log; thin-book impact)",
    "downside_var_ratio_168h": "downside semivariance share (selling-pressure release, positive)",
    "tail_ratio_168h": "tail ratio P95/|P05| (fat right tail = lottery-type)",
}
FACTOR_DESC.update(wavelet_factors.FACTOR_DESC)   # time-frequency / OU factor descriptions (8)


# ============ cross-sectional composite factors ============
# Per-timestamp cross-symbol pct-rank then equal-weight average; only computable at the panel level (a single
# symbol cannot compute a cross-sectional rank), injected by score_all.
COMPOSITES = {
    "mom_combo": (["mom_24h", "mom_72h", "mom_168h"], 2),
    "vol_combo": (["vol_24h", "vol_72h", "atr_pct"], 2),
    "meta_combo": (["vol_24h", "vol_72h", "atr_pct", "mom_24h", "mom_72h", "mom_168h"], 4),
}
COMPOSITE_DESC = {
    "mom_combo": "momentum-family rank composite (24/72/168h equal weight; denoising/lower-turnover, veto layer)",
    "vol_combo": "volatility-family rank composite (v24/v72/atr; low-vol main score)",
    "meta_combo": "vol3+mom3 six-factor rank composite (overall ICIR champion, main IC baseline)",
}


def _inject_composites(panel):
    """Inject the cross-sectional composite factor columns into each symbol's df in the panel. Sections with fewer than min_comp non-NaN components are set NaN."""
    for name, (comps, min_comp) in COMPOSITES.items():
        wides = [pd.DataFrame({s: panel[s][c] for s in panel}) for c in comps]
        pct = [w.rank(axis=1, pct=True) for w in wides]
        stacked = np.stack([p.to_numpy() for p in pct])
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)   # the all-NaN warmup rows' "Mean of empty slice" is harmless
            avg = np.nanmean(stacked, axis=0)
        cnt = np.sum(~np.isnan(stacked), axis=0)
        avg[cnt < min_comp] = np.nan
        wide = pd.DataFrame(avg, index=wides[0].index, columns=wides[0].columns)
        for s in panel:
            panel[s][name] = wide[s].reindex(panel[s].index)
    return panel


# ============ WQ101 gauntlet-adopted factors ============
# 101 formulas pre-registered through three elimination rounds (full-sample |t|>=3.5 + two windows same-signed /
# yearly same-sign rate >=70% / incumbent |ρ|<0.7) + a two-lens assassination (reskin test / recent-decay test),
# leaving 4 survivors. Cross-sectional formula factors, only computable at the panel level (need the whole-pool
# OHLCV wide frame), shadow-scored observe-only: they do not enter the live ic_weights export.
# Implementation in wq101/batch*.py (verbatim transcription, fidelity-reviewed).
WQ101_DESC = {
    "wq_a005": "WQ#005 open vs 10-period vwap mean deviation × |close-vwap| rank gate (intraday structure, t=+4.0)",
    "wq_a020": "WQ#020 gap-open strength (open vs prior H/C/L three-rank product, negated; most orthogonal to incumbents, t=+3.7)",
    "wq_a024": "WQ#024 when the long trend is flat, distance to the 100-period low regression / else 3-period reversal (conditional reversal, champion t=-4.4)",
    "wq_a077": "WQ#077 mid vs vwap deviation decay rank × volume-price correlation decay rank min (mean reversion, t=-3.7)",
}


# ============ LLM-generated round-1 adopted factors ============
# 49 formulas blind-generated (the generating model touched no IC / historical data) -> reskin pre-screen -> freeze + hash
# -> three elimination rounds -> 6 survivors -> two-lens assassination adopts 3.
LLM1_DESC = {
    "llm1_4_05": "blind-gen R1: 72h new-high position 6h offset / 24h volatility normalized (new-high momentum family, t=+5.9 strongest in the library)",
    "llm1_3_07": "blind-gen R1: return jitter - net-displacement cross-sectional z difference (jaggedness / path efficiency inverse, t=-5.4)",
    "llm1_2_08": "blind-gen R1: volume × candle body / range (trade conviction, t=-5.2)",
}


def _inject_llm1(panel, tf="1h", n_tail=None):
    """LLM blind-gen adopted factor injection (same structure as _inject_wq101, provenance accounted separately). fail-soft."""
    try:
        from alphagauntlet.wq101 import batch7, panel_io
        fns = {"llm1_4_05": batch7.alpha_llm1_4_05, "llm1_3_07": batch7.alpha_llm1_3_07,
               "llm1_2_08": batch7.alpha_llm1_2_08}
        P = panel_io.load_field_panel(pool=list(panel), tf=tf, n_tail=n_tail)
    except Exception:  # noqa: BLE001
        return []
    injected = []
    for name, fn in fns.items():
        try:
            wide = fn(P).replace([np.inf, -np.inf], np.nan)
        except Exception:  # noqa: BLE001
            continue
        for s in panel:
            panel[s][name] = (wide[s].reindex(panel[s].index) if s in wide.columns
                              else pd.Series(np.nan, index=panel[s].index))
        injected.append(name)
    return injected


# ============ LLM-generated round-2 adopted factors ============
# 50 blind-generated (generation touched no IC) -> sign + freeze -> a generic evaluator (validated 49/49 against
# the batch7 49-factor gold standard) -> three elimination rounds 46->4 -> final review SAFE TO KEEP.
# Implementation wq101/llm_round2_adopted.py (4 hand-written functions validated cell-by-cell against the evaluator).
R2_DESC = {
    "r2v2_01_02": "blind-gen R2: OHLC volatility estimator, intraday range vol - cross-bar body vol (t=+8.3 strongest in the library)",
    "r2v2_03_04": "blind-gen R2: Roll implied spread, return-series covariance - vwap-series covariance (t=+6.9)",
    "r2v2_05_03": "blind-gen R2: downside co-skewness, return × squared market deviation (t=-5.5)",
    "r2v2_06_01": "blind-gen R2: Kyle lambda price-impact elasticity (t=+5.9, 2026 IC decay, monitor live)",
}


def _inject_r2(panel, tf="1h", n_tail=None):
    """LLM blind-gen round-2 adopted factor injection (same structure as _inject_llm1, provenance accounted separately). fail-soft."""
    try:
        from alphagauntlet.wq101 import llm_round2_adopted as r2, panel_io
        fns = dict(r2.ALPHAS)
        P = panel_io.load_field_panel(pool=list(panel), tf=tf, n_tail=n_tail)
    except Exception:  # noqa: BLE001
        return []
    injected = []
    for name, fn in fns.items():
        try:
            wide = fn(P).replace([np.inf, -np.inf], np.nan)
        except Exception:  # noqa: BLE001
            continue
        for s in panel:
            panel[s][name] = (wide[s].reindex(panel[s].index) if s in wide.columns
                              else pd.Series(np.nan, index=panel[s].index))
        injected.append(name)
    return injected


def _inject_wq101(panel, tf="1h", n_tail=None):
    """Inject the WQ101 adopted factors into each symbol's df in the panel, returning the list of successfully-injected factor names.
    fail-soft: skip an item if the wq101 package / single-factor computation fails (the absence is visible in the
    scorecard n_factors and wq101_adopted fields, not silently swallowing the whole score)."""
    try:
        from alphagauntlet.wq101 import batch1, batch2, batch4, panel_io
        fns = {"wq_a005": batch1.alpha_wq005, "wq_a020": batch1.alpha_wq020,
               "wq_a024": batch2.alpha_024, "wq_a077": batch4.alpha_077}
        P = panel_io.load_field_panel(pool=list(panel), tf=tf, n_tail=n_tail)
    except Exception:  # noqa: BLE001  # research-side dependency missing / data unreadable -> this round's scoring excludes wq factors
        return []
    injected = []
    for name, fn in fns.items():
        try:
            wide = fn(P).replace([np.inf, -np.inf], np.nan)
        except Exception:  # noqa: BLE001
            continue
        for s in panel:
            # fill a missing symbol's column with all NaN (the valid mask filters it naturally), so that if the two
            # loaders' symbol-filtering logic ever diverges, evaluate_factor taking a column by symbol does not
            # KeyError-blow the whole card.
            panel[s][name] = (wide[s].reindex(panel[s].index) if s in wide.columns
                              else pd.Series(np.nan, index=panel[s].index))
        injected.append(name)
    return injected


# ============ data loading and alignment ============
def _load_panel(pool=None, tf="1h", n_tail=None):
    """Load the whole-pool factor panel. Returns {symbol: factor_df}, aligned on a common time index."""
    pool = pool or POOL
    panel = {}
    for sym in pool:
        fn = os.path.join(DATA_DIR, sym.replace("/", "_") + f"-{tf}.feather")
        if not os.path.exists(fn):
            continue
        df = pd.read_feather(fn)
        if n_tail:
            df = df.tail(n_tail + 250).reset_index(drop=True)
        df = df.set_index("date")
        fac = compute_factors(df)
        fac["close"] = df["close"].astype(float)
        panel[sym] = fac
    return panel


def _panel_data_through(panel):
    """The min over each symbol's max close-index = the latest time the whole pool jointly covers (the scorecard data-freshness basis)."""
    try:
        last = [str(df.index.max()) for df in panel.values() if len(df)]
        return min(last) if last else None
    except Exception:  # noqa: BLE001
        return None


def _forward_return(close, h):
    """Forward return h bars later (close[t+h]/close[t]-1), the last h bars NaN."""
    return close.shift(-h) / close - 1.0


def _rank_ic(fvals, rvals):
    """Single-section Rank IC: the rank Pearson (=Spearman) of factor value and return. < MIN_COINS or a constant column returns NaN."""
    f = np.asarray(fvals, float); r = np.asarray(rvals, float)
    mask = np.isfinite(f) & np.isfinite(r)
    if mask.sum() < MIN_COINS:
        return np.nan
    fr = pd.Series(f[mask]).rank().to_numpy()
    rr = pd.Series(r[mask]).rank().to_numpy()
    if fr.std() < 1e-9 or rr.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(fr, rr)[0, 1])


def _quantile_spread(fvals, rvals, q=N_QUANTILE):
    """Top quantile group mean return − bottom quantile group mean return (single section)."""
    f = np.asarray(fvals, float); r = np.asarray(rvals, float)
    mask = np.isfinite(f) & np.isfinite(r)
    if mask.sum() < q:
        return np.nan
    f, r = f[mask], r[mask]
    order = np.argsort(f)
    k = max(1, len(f) // q)
    bottom = r[order[:k]].mean()
    top = r[order[-k:]].mean()
    return float(top - bottom)


def _spread_sampled(fac_v, fwd_v, cap=1500):
    """Sampled mean of the tercile spread (top group mean return − bottom group mean return), capped at cap iterations for speed."""
    fa = fac_v.to_numpy(); fw = fwd_v.to_numpy()
    step = max(1, len(fa) // cap)
    sp = [_quantile_spread(fa[i], fw[i]) for i in range(0, len(fa), step)]
    sp = [s for s in sp if np.isfinite(s)]
    return round(float(np.mean(sp)), 4) if sp else None


def evaluate_factor(panel, factor, h):
    """Evaluate one factor's cross-sectional predictive power at horizon h (non-overlapping sampling, valid t-stat). Vectorized Rank IC."""
    syms = list(panel)
    fac_mat = pd.DataFrame({s: panel[s][factor] for s in syms})
    fwd_mat = pd.DataFrame({s: _forward_return(panel[s]["close"], h) for s in syms})
    common = fac_mat.index.intersection(fwd_mat.index)
    fac_mat = fac_mat.loc[common].iloc[::h]          # non-overlapping sampling (independent forward windows)
    fwd_mat = fwd_mat.loc[common].iloc[::h]
    valid = fac_mat.notna() & fwd_mat.notna()
    rowmask = valid.sum(axis=1) >= MIN_COINS
    fac_v = fac_mat.where(valid)[rowmask]
    fwd_v = fwd_mat.where(valid)[rowmask]
    if len(fac_v) < 20:
        return {"horizon_h": h, "n_periods": int(len(fac_v)), "verdict": "insufficient samples", "mean_ic": None}
    # within-row (cross-sectional) rank -> rank Pearson = Rank IC, fully vectorized
    fr = fac_v.rank(axis=1); rr = fwd_v.rank(axis=1)
    fc = fr.sub(fr.mean(axis=1), axis=0); rc = rr.sub(rr.mean(axis=1), axis=0)
    num = (fc * rc).sum(axis=1)
    den = np.sqrt((fc ** 2).sum(axis=1) * (rc ** 2).sum(axis=1))
    ics = (num / den).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    n = len(ics)
    if n < 20:
        return {"horizon_h": h, "n_periods": n, "verdict": "insufficient samples", "mean_ic": None}
    mean_ic = float(ics.mean()); std_ic = float(ics.std(ddof=1))
    if std_ic > 1e-9:
        icir = mean_ic / std_ic
    else:                                   # IC variance ≈0: a perfectly consistent nonzero IC = very strong (not noise), a zero IC = ineffective
        icir = float(np.sign(mean_ic) * 10.0) if abs(mean_ic) > 1e-9 else 0.0
    # IC-series lag-1 autocorrelation correction of the effective degrees of freedom (adjacent sections are non-independent -> n_eff<n, remove the inflated t-stat)
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
    return {
        "horizon_h": h, "n_periods": n, "n_eff": round(n_eff, 1),
        "mean_ic": round(mean_ic, 4), "std_ic": round(std_ic, 4),
        "icir": round(icir, 3), "t_stat": round(t_stat, 2),
        "t_stat_note": "n_eff already shrunk by the IC-series lag-1 autocorrelation; cross-sectional non-independence not corrected, |t| is still an optimistic upper bound, used only for relative strength",
        "ic_positive_rate": round(float((ics > 0).mean()), 3),
        "quantile_spread": _spread_sampled(fac_v, fwd_v),
        "abs_mean_ic": round(abs(mean_ic), 4),
        "direction": ("positive (factor up -> future up)" if mean_ic > 0 else "negative (factor up -> future down, use inverted)"),
        "verdict": _verdict(abs(mean_ic), abs(t_stat)),
    }


def _verdict(abs_ic, abs_t):
    """Factor strength grading (honest: strict on a small cross-section; the green gate tightened |t|>=3->5 against false significance after n_eff shrinkage)."""
    if abs_t < 2.0:
        return "noise (t<2, not even in-sample significant, do not use)"
    if abs_ic >= 0.05 and abs_t >= 5.0:
        return "strong (|IC|>=0.05 and |t|>=5, in-sample significant, worth validating)"
    if abs_ic >= 0.03:
        return "medium (weak predictive power, combine with other factors)"
    return "weak (significant but |IC| too small, limited practical value)"


def score_all(tf="1h", save=True, n_tail=None):
    """Evaluate all factors × all horizons, produce a scorecard report.
    n_tail: limit each symbol to the last N bars (avoids the wavelet factors timing out over the full ~47k history);
    None = full history (for standalone analysis)."""
    panel = _load_panel(tf=tf, n_tail=n_tail)
    if len(panel) < MIN_COINS:
        return {"ok": False, "reason": f"insufficient usable symbols ({len(panel)}), need >={MIN_COINS}"}
    _inject_composites(panel)                       # cross-sectional composite factors (only computable at the panel level)
    wq_names = _inject_wq101(panel, tf=tf, n_tail=n_tail)   # WQ101 adopted factors
    llm_names = _inject_llm1(panel, tf=tf, n_tail=n_tail)   # LLM blind-gen adopted factors
    r2_names = _inject_r2(panel, tf=tf, n_tail=n_tail)      # LLM blind-gen round-2 adopted factors
    names = FACTOR_NAMES + list(COMPOSITES) + wq_names + llm_names + r2_names
    factors = {}
    for f in names:
        by_h = {f"h{h}": evaluate_factor(panel, f, h) for h in HORIZONS}
        # sort by the 24h main horizon
        main = by_h.get("h24", {})
        factors[f] = {"desc": FACTOR_DESC.get(f) or COMPOSITE_DESC.get(f)
                      or WQ101_DESC.get(f) or LLM1_DESC.get(f) or R2_DESC.get(f, ""),
                      "main_horizon": "h24",
                      "abs_mean_ic": main.get("abs_mean_ic") or 0.0,
                      "by_horizon": by_h}
    ranked = sorted(factors.items(), key=lambda kv: kv[1]["abs_mean_ic"], reverse=True)
    report = {
        "ok": True, "tf": tf, "pool": list(panel), "n_coins": len(panel),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "data_through": _panel_data_through(panel),
        "horizons_h": HORIZONS, "n_factors": len(names),
        "wq101_adopted": wq_names,   # successfully-injected WQ101 adopted factors (should be 4; fewer means injection failed)
        "llm1_adopted": llm_names,   # successfully-injected LLM blind-gen adopted factors (should be 3)
        "r2_adopted": r2_names,      # successfully-injected LLM blind-gen round-2 adopted factors (should be 4)
        "methodology": "Rank IC (Spearman cross-sectional rank correlation) + ICIR + non-overlapping-sampling t-stat + tercile spread",
        "honesty": f"small cross-section N={len(panel)}, single-period IC is noisy; the score is an in-sample checkup, not a profit guarantee. t>=3 still needs paper/backtest re-validation.",
        "ranked_factors": [k for k, _ in ranked],
        "factors": dict(ranked),
    }
    if save:
        os.makedirs(STATE, exist_ok=True)
        tmp = SCORES_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        os.replace(tmp, SCORES_PATH)
        if n_tail is None:
            # a full-history run additionally refreshes the scoring channel weights (state/learning_weights.json).
            # A short window (n_tail) does not refresh: scoring weights must be pinned to full-history validation,
            # not washed out by the t-values of a 1.4-year short window. fail-soft.
            try:
                from alphagauntlet import scoring
                scoring.export_weights(report)
            except Exception:  # noqa: BLE001
                pass
    return report


def load_scores():
    """Read the stored factor scores (for downstream/front-end use). None -> return a hint."""
    if not os.path.exists(SCORES_PATH):
        return {"ok": False, "reason": "not scored yet, run score_all() first"}
    with open(SCORES_PATH, encoding="utf-8") as f:
        return json.load(f)


def ic_weights(horizon="h24"):
    """Export the scorecard's measured IC as selection weights (data-driven selection, replacing hand-set weights).
    Mapping: w_momentum<-mom_24h IC, w_volatility<-vol_24h IC, w_liquidity<-liq_ratio IC; funding has no historical IC, stays 0.
    **Weight sign = IC sign**: if the data shows a negative momentum IC (reversal), w_momentum automatically goes negative
    (selecting recent laggards), correcting a hand-set directional error.

    Freshness guard: scorecard generated_at older than MAX_AGE_H (or no timestamp) -> fall back to hand-set defaults (a stale IC must not mislead selection).
    True shrinkage (not amplification):
    - |t|<2 noise factors get weight 0; a cross-period sign-stability gate (h4 must agree in sign with the main horizon's IC to be enabled).
    - magnitude (|IC|/IC_FULL)**2 square law shrinks small IC harder; |IC|>IC_FULL then a linear bonus (stronger factors get higher weight, capped at 1.5×).
    Returns None when there is no scorecard; always keeps the five keys w_momentum/w_volatility/w_liquidity/w_funding/_source_ic (selection contract)."""
    s = load_scores()
    if not s.get("ok"):
        return None
    # ---- stale guard (before the IC path): over-age / no timestamp falls back to hand-set defaults ----
    gen = s.get("generated_at")
    stale = True
    if gen:
        try:
            ts = datetime.datetime.fromisoformat(gen.replace("Z", ""))
            age_h = (datetime.datetime.utcnow() - ts).total_seconds() / 3600.0
            stale = age_h > MAX_AGE_H
        except Exception:  # noqa: BLE001
            stale = True
    if stale:
        return {"w_momentum": 0.5, "w_volatility": -0.2, "w_liquidity": 0.2, "w_funding": 0.0,
                "_note": f"scorecard stale (generated_at={gen}, >{MAX_AGE_H}h) or no timestamp, falling back to hand-set defaults",
                "_stale": True, "_generated_at": gen, "_source_ic": None}

    # ---- IC export path (not stale) ----
    IC_FULL = 0.05

    def _m(name, h):
        return s.get("factors", {}).get(name, {}).get("by_horizon", {}).get(h, {})

    def _ic(name, h=horizon):
        v = _m(name, h).get("mean_ic")
        return float(v) if v is not None else 0.0

    def _t(name, h=horizon):
        v = _m(name, h).get("t_stat")
        return abs(float(v)) if v is not None else 0.0

    def _w(name, base):
        v = _ic(name); t = _t(name)
        if t < 2.0:                          # not even in-sample significant -> 0
            return 0.0
        ic_h4 = _ic(name, "h4")              # cross-period sign-stability gate: h4 must agree in sign with the main horizon
        if ic_h4 == 0.0 or (ic_h4 > 0) != (v > 0):
            return 0.0
        a = abs(v)
        if a <= IC_FULL:
            mag = (a / IC_FULL) ** 2         # square-law shrinkage (shrinks small IC harder)
        else:
            mag = 1.0 + min((a - IC_FULL) / IC_FULL, 1.0) * 0.5   # linear bonus above 0.05, capped at 1.5×
        return round(math.copysign(base * mag, v), 3)

    return {"w_momentum": _w("mom_24h", 0.5), "w_volatility": _w("vol_24h", 0.5),
            "w_liquidity": _w("liq_ratio", 0.3), "w_funding": 0.0,
            "_note": "IC export (square-law shrinkage + t>=2 gate + h4/main sign-stability gate + linear bonus above 0.05); funding has no IC = 0",
            "_stale": False, "_generated_at": gen,
            "_source_ic": {"mom_24h": _ic("mom_24h"), "vol_24h": _ic("vol_24h"), "liq_ratio": _ic("liq_ratio")}}


if __name__ == "__main__":
    import sys
    tf = sys.argv[1] if len(sys.argv) > 1 else "1h"
    rep = score_all(tf=tf)
    if rep.get("ok"):
        print(f"factor scoring done: {rep['n_factors']} factors × {rep['n_coins']} symbols ({tf})")
        print(f"{'factor':<14}{'|IC|':>7}{'ICIR':>7}{'t-stat':>8}{'spread':>9}  verdict")
        for name in rep["ranked_factors"]:
            m = rep["factors"][name]["by_horizon"]["h24"]
            if m.get("mean_ic") is None:
                print(f"{name:<14}{'insufficient':>7}")
                continue
            print(f"{name:<14}{m['abs_mean_ic']:>7}{m['icir']:>7}{m['t_stat']:>8}"
                  f"{str(m.get('quantile_spread')):>9}  {m['verdict']}")
    else:
        print("failed:", rep.get("reason"))
