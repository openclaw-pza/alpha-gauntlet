#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure market-regime and trend-signal indicators.

This module contains only standalone, side-effect-free functions that take a
price ``DataFrame`` (or array) and return signals / regime labels. There is no
exchange connectivity, no data fetching, no account or order logic here — feed
it OHLCV you already have and it returns deterministic indicators.

Two complementary tools:

* :func:`guru_signal` — a deterministic ComboTrend classifier (triple-EMA
  alignment, Donchian breakout, ADX, EMA200 slope) producing a discrete
  trend label. Use it as a verified directional anchor.
* :func:`hurst_rs` / :func:`hurst_regime` — a bare rescaled-range Hurst
  exponent on log returns, used to label the *persistence* regime
  (trend / chop / random). Hurst measures autocorrelation persistence, NOT
  direction — pair it with :func:`guru_signal` for direction.
"""
import numpy as np
import pandas as pd
import talib


def guru_signal(df):
    """Deterministic ComboTrend signal from an OHLCV DataFrame.

    Classifies the current bar into one of four discrete regimes using a fast/
    medium/slow EMA stack, an EMA200 trend filter, a Donchian breakout test and
    ADX:

    * ``bear_avoid``       — price below EMA200 or EMA200 not rising (downtrend).
    * ``donchian_breakout``— turtle-style breakout with trend confirmation.
    * ``ema_bull``         — triple-EMA bullish alignment.
    * ``neutral``          — none of the above.

    Requires columns ``close``, ``high``, ``low``. Returns a dict of the label
    plus the underlying boolean components and the latest ADX value.
    """
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    f = talib.EMA(c, 27)
    m = talib.EMA(c, 47)
    s = talib.EMA(c, 150)
    e200 = talib.EMA(c, 200)
    adx = talib.ADX(h, low, c, 14)
    dc_high = pd.Series(h).rolling(120).max().shift(1).to_numpy()
    rising = bool(e200[-1] > e200[-25]) if len(e200) > 25 else False
    ema_bull = bool(f[-1] > m[-1] > s[-1])
    bear = bool(c[-1] < e200[-1] or not rising)
    donchian = bool((not np.isnan(dc_high[-1])) and c[-1] >= dc_high[-1]
                    and adx[-1] >= 20 and c[-1] > e200[-1] and rising)
    sig = ("bear_avoid" if bear else "donchian_breakout" if donchian
           else "ema_bull" if ema_bull else "neutral")
    return {"signal": sig, "ema_bull": ema_bull, "donchian_breakout": donchian,
            "bear_regime": bear, "ema200_rising": rising, "adx": round(float(adx[-1]), 1)}


def hurst_rs(series, min_chunk=10):
    """Bare rescaled-range (R/S) Hurst exponent on log returns.

    Simple, monotonic, stable on short windows. Honest caveats:

    * On ~480-bar windows the estimator's null (random walk) centre is ~0.55,
      not 0.5 — short-sample R/S has an inherent upward bias. An analytic
      correction empirically overshoots, so the bare version is kept and the
      regime thresholds (:data:`HURST_TREND_TH` / :data:`HURST_CHOP_TH`) are
      calibrated to that biased null instead.
    * Hurst measures *persistence / autocorrelation* of returns, not the
      direction of price. A drifting trend (random-walk-style uptrend) still
      reads H ~ 0.55 and is indistinguishable from a random walk on persistence
      alone — combine with :func:`guru_signal` for direction.

    Returns a float, or ``None`` when there is insufficient data.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 64:
        return None
    lr = np.diff(np.log(x))
    lr = lr[np.isfinite(lr)]
    n = len(lr)
    if n < 50:
        return None
    max_k = int(np.floor(np.log2(n // min_chunk)))
    sizes = sorted(set(n // (2 ** k) for k in range(max_k + 1) if n // (2 ** k) >= min_chunk))
    if len(sizes) < 2:
        return None
    log_n, log_rs = [], []
    for s in sizes:
        rs_vals = []
        for i in range(n // s):
            chunk = lr[i * s:(i + 1) * s]
            dev = np.cumsum(chunk - chunk.mean())
            R = dev.max() - dev.min()
            S = chunk.std()
            if S > 1e-12 and R > 0:
                rs_vals.append(R / S)
        if rs_vals:
            log_n.append(np.log(s))
            log_rs.append(np.log(np.mean(rs_vals)))
    if len(log_n) < 2:
        return None
    return float(np.polyfit(log_n, log_rs, 1)[0])


# Regime thresholds calibrated for this estimator (bare R/S, ~480-bar 1h
# windows). See the Monte-Carlo null distribution note in :func:`hurst_rs`.
HURST_TREND_TH = 0.58     # H above this = strong persistence (trend-following ok)
HURST_CHOP_TH = 0.47      # H below this = anti-persistence / mean-reversion (chop)


def hurst_regime(df, n=480):
    """Label the persistence regime of a price series via the Hurst exponent.

    Pure function: pass an OHLCV DataFrame (or anything with a ``close`` column)
    and it computes the Hurst exponent over the trailing ``n`` closes and maps
    it to a regime label. No look-ahead — only historical closes are used.

    Regimes:

    * ``trend``   — strong persistence (H > :data:`HURST_TREND_TH`).
    * ``chop``    — anti-persistence / mean reversion (H < :data:`HURST_CHOP_TH`).
    * ``random``  — indeterminate, the most common label.
    * ``unknown`` — insufficient data.

    Returns a dict ``{hurst, regime, ...}``.
    """
    try:
        if df is None or "close" not in getattr(df, "columns", []) or len(df) < 80:
            return {"hurst": None, "regime": "unknown",
                    "note": "insufficient data to determine regime"}
        h = hurst_rs(df["close"].to_numpy(dtype=float)[-n:])
        if h is None:
            return {"hurst": None, "regime": "unknown",
                    "note": "Hurst computation failed (insufficient data)"}
        if h > HURST_TREND_TH:
            regime = "trend"
        elif h < HURST_CHOP_TH:
            regime = "chop"
        else:
            regime = "random"
        return {"hurst": round(h, 3), "regime": regime,
                "th": {"trend": HURST_TREND_TH, "chop": HURST_CHOP_TH},
                "note": "Hurst labels persistence only; read direction from guru_signal"}
    except Exception as e:  # noqa: BLE001 single-series failure is non-fatal
        return {"hurst": None, "regime": "unknown",
                "note": f"hurst_regime error: {type(e).__name__}"}


# --------------------------------------------------------------------------- #
# Tiered universe (large/mid/small) — shared with factor_eval as the default symbol pool.
# --------------------------------------------------------------------------- #
# One cross-sectional universe is fed to the selector; differentiation is via (1) per-tier position caps
# (2) a small-cap liquidity admission gate (3) quarter-Kelly + inverse-volatility sizing naturally shrinking
# high-volatility small-cap positions. Rationale: cross-sectional momentum in crypto is significant only over
# a wide universe; small caps have stronger momentum but higher liquidity/wick/crash-correlation risk, hence
# smaller positions + stricter admission.
TIERS = {
    "large": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
    "mid":   ["DOGE/USDT", "ADA/USDT", "LINK/USDT", "AVAX/USDT", "DOT/USDT", "TRX/USDT", "LTC/USDT"],
    "small": ["NEAR/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "INJ/USDT", "SUI/USDT", "FIL/USDT", "ATOM/USDT"],
}
POOL = TIERS["large"] + TIERS["mid"] + TIERS["small"]   # 20-symbol unified universe
TIER_CAP = {"large": 0.30, "mid": 0.20, "small": 0.12}  # single-position cap as a fraction of equity (smaller for small caps)
TIER_MIN_LIQ = {"large": 0.0, "mid": 0.0, "small": 0.8}  # small-cap liquidity admission gate (selectable only if liq>=this)


def tier_of(symbol):
    """The symbol's tier (large/mid/small); an unregistered symbol is treated as the most conservative small."""
    for t, syms in TIERS.items():
        if symbol in syms:
            return t
    return "small"
