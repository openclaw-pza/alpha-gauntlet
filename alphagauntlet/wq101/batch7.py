#!/usr/bin/env python3
r"""WQ101 batch7 — verbatim transcription of 49 expressions from an LLM-generated, frozen round-1 list.

Source (verbatim, no tuning / no window changes):
- A frozen expression list, content-addressed by a canonical sha256 of the compact, sorted-keys JSON
  (with the sha256/sha256_scope fields removed from the scope). The list is frozen; any edit invalidates it.
- ALPHAS keys = each entry's "name" (llm1_X_YY), no clash with wq001-wq083.

Contract (identical to batch1-5):
- Each fn(P) -> wide DataFrame(index=time, columns=symbols), P=panel_io.load_field_panel().
- Only ops.py / ops_ext.py operators; cross-section axis=1, time-series axis=0, corr/cov use native pandas.
- Warmup NaN passes through, no fillna; inf cleaned uniformly by runner (+/-inf->NaN); a denominator-zero
  guard uses .replace(0, 1e-4) (denominator DataFrame only). The frozen list already carries +1e-9 small
  values, kept verbatim.
- Verbatim discipline: windows/coefficients/thresholds unchanged; changing a formula based on its IC preview
  is forbidden (the list is frozen).

DSL -> operator mapping (where semantics are ambiguous, the dsl.py operator definition wins, noted here):
- RANK_CS_PCT(x)      -> ops.rank_cs_pct(x)         (cross-sectional percentile rank [0,1], axis=1)
- ZSCORE_CS(x)        -> ops_ext.zscore_cs(x)        (cross-sectional z-score, axis=1, ddof=1)
- SKEW_CS / MEDIAN_CS -> ops_ext.skew_cs / median_cs (cross-section, scalar broadcast across the row)
- CORR(x,y,d)         -> ops.corr(x,y,d)             (per-symbol rolling Pearson)
- DELAY/DELTA         -> ops.delay / ops.delta
- TS_MEAN/STD/MIN/MAX/SUM -> ops.ts_*
- TS_RANK(x,d)        -> ops.ts_rank(x,d)            (midpoint rank percentile [0,1] of the last element among the past d)
- TS_ARGMAX/ARGMIN    -> ops.ts_argmax / ops.ts_argmin
      convention: this project's executable operators use a 'lookback offset' (0=current bar, d-1=oldest),
      pinned by the tests as the project standard. dsl.py only does AST parsing and does not define the arg
      execution semantics, so per "the dsl.py operator definition wins" we fall to the project's only
      executable standard operators ops.ts_argmax/ts_argmin (lookback). The frozen list is LLM-generated
      with no external reference library to align to, so the _fwd sign-flip variants used for batch4
      #096/#098 are not introduced. Verbatim transcription.
- TS_SKEW/TS_KURT/TS_MEDIAN/TS_MAD -> ops_ext.ts_*
- PERCENTILE(x,p,q)   -> ops_ext.percentile(x,p,q)   (rolling quantile, q in [0,1])
- HIGHDAY/LOWDAY      -> ops_ext.highday / lowday     (1..p, newest=1)
- COUNT(cond,p)       -> ops_ext.count(cond,p)        (strict min_periods=p)
- SUMIF(x,p,cond)     -> ops_ext.sumif(x,p,cond)
- FILTER(x,cond)      -> ops_ext.filter_(x,cond)      (cond=False -> 0)
- SIGN(x)             -> np.sign(x)
- SIGNED_POWER(x,e)   -> ops.signed_power(x,e)        (sign(x)*|x|^e)
- SEQUENCE(n)         -> ops_ext.sequence(n)          (1D vector [1..n], as the regression fixed x)
- REGBETA(y,x,p)/REGRESI(y,x,p) -> ops_ext.regbeta/regresi(y,x,p)
      signature = (y, x, p), matching the list's REGBETA($close, SEQUENCE(24), 24) argument order.
- DECAY_LINEAR(x,d)   -> ops.decay_linear(x,d)        (newest bar largest weight)
- ABS(x) -> x.abs(); LOG(x) -> np.log(x)
- MAX(a,b)/MIN(a,b)   -> np.maximum/np.minimum         (binary element-wise, not time-series)
- SUBTRACT(a,b)/DIVIDE(a,b) -> a-b / a/b
- WHERE(cond,a,b) / cond?a:b -> ternary, implemented with .where(cond_mask, b) (cond NaN passes through NaN)
- TS_PCTCHANGE(x,d)   -> local _ts_pctchange(x,d) = x/DELAY(x,d) - 1
      dsl.py gives no execution semantics; the incumbent registry's vol_24h=TS_STD(TS_PCTCHANGE($close,1),24)
      (return std) established usage confirms the semantics = d-period percent change (PIT-safe: shift only).
"""
import numpy as np

from alphagauntlet.wq101 import ops, ops_ext


# --------------------------------------------------------------------------- #
# Local helpers (compute fields / semantic helpers only, not in the rolling.apply hot path)
# --------------------------------------------------------------------------- #
def _ts_pctchange(df, d):
    """TS_PCTCHANGE(x, d) = x / DELAY(x, d) - 1.

    The DSL gives no execution semantics; per the incumbent registry's vol_24h usage, this is a d-period
    percent change. The denominator is the raw DELAY(x,d); a 0 there gives inf in pandas -> runner cleans
    it uniformly; warmup NaN passes through.
    """
    return df / ops.delay(df, d) - 1.0


def _time_ramp_like(df):
    """A same-shape 'time-position ramp' wide frame: each column = row order [0,1,2,...] (identical per column).

    Use: CORR($close, SEQUENCE(48), 48) — the second argument is a fixed ascending vector [1..48].
    ops.corr requires two same-shape wide frames; a d-period rolling corr is invariant to a positive affine
    transform (a*t+b) of the independent variable, and SEQUENCE(48)=[1..48] is exactly a positive affine of
    the in-window position [0..47], so the global row-order ramp's rolling window is substituted (each
    48-window covers [t-47..t], which is a positive affine of [1..48], so corr is numerically identical).
    This is the faithful way to land SEQUENCE into ops.corr, without changing the window or semantics.
    """
    n = df.shape[0]
    ramp = np.arange(n, dtype=float)[:, None]
    import pandas as pd
    return pd.DataFrame(np.repeat(ramp, df.shape[1], axis=1),
                        index=df.index, columns=df.columns)


# =========================================================================== #
# Group 1_1 (llm1_1_01 .. llm1_1_10): candle shape / volume-price
# =========================================================================== #
def alpha_llm1_1_01(P):
    # RANK_CS_PCT(($close - $open) / ($high - $low + 1e-9))
    return ops.rank_cs_pct((P["close"] - P["open"]) / (P["high"] - P["low"] + 1e-9))


def alpha_llm1_1_02(P):
    # RANK_CS_PCT(($high - MAX($open, $close)) / ($high - $low + 1e-9))
    return ops.rank_cs_pct(
        (P["high"] - np.maximum(P["open"], P["close"])) / (P["high"] - P["low"] + 1e-9))


def alpha_llm1_1_03(P):
    # RANK_CS_PCT((MIN($open, $close) - $low) / ($high - $low + 1e-9))
    return ops.rank_cs_pct(
        (np.minimum(P["open"], P["close"]) - P["low"]) / (P["high"] - P["low"] + 1e-9))


def alpha_llm1_1_04(P):
    # RANK_CS_PCT(($open - DELAY($close, 1)) / (DELAY($close, 1) + 1e-9))
    dc1 = ops.delay(P["close"], 1)
    return ops.rank_cs_pct((P["open"] - dc1) / (dc1 + 1e-9))


def alpha_llm1_1_05(P):
    # TS_MEAN(SIGN($close - $open) * $volume, 24) / (TS_MEAN($volume, 24) + 1e-9)
    signed_vol = np.sign(P["close"] - P["open"]) * P["volume"]
    return ops.ts_mean(signed_vol, 24) / (ops.ts_mean(P["volume"], 24) + 1e-9)


def alpha_llm1_1_06(P):
    # ZSCORE_CS(($close - $open) / ($close - $open + (ABS($high - $close) + ABS($open - $low)) + 1e-9))
    body = P["close"] - P["open"]
    denom = body + ((P["high"] - P["close"]).abs() + (P["open"] - P["low"]).abs()) + 1e-9
    return ops_ext.zscore_cs(body / denom)


def alpha_llm1_1_07(P):
    # RANK_CS_PCT(TS_SUM(ABS($close - $open), 12) / (TS_SUM($high - $low, 12) + 1e-9))
    num = ops.ts_sum((P["close"] - P["open"]).abs(), 12)
    den = ops.ts_sum(P["high"] - P["low"], 12) + 1e-9
    return ops.rank_cs_pct(num / den)


def alpha_llm1_1_08(P):
    # RANK_CS_PCT(COUNT(($close > $open) && ($volume > DELAY($volume, 1)), 24))
    cond = (P["close"] > P["open"]) & (P["volume"] > ops.delay(P["volume"], 1))
    return ops.rank_cs_pct(ops_ext.count(cond, 24))


def alpha_llm1_1_09(P):
    # ZSCORE_CS(($close - ($high + $low) / 2) / (TS_STD($returns, 48) + 1e-9))
    mid = (P["high"] + P["low"]) / 2.0
    return ops_ext.zscore_cs((P["close"] - mid) / (ops.ts_std(P["returns"], 48) + 1e-9))


def alpha_llm1_1_10(P):
    # RANK_CS_PCT(CORR($close - $open, $volume, 24))
    return ops.rank_cs_pct(ops.corr(P["close"] - P["open"], P["volume"], 24))


# =========================================================================== #
# Group 1_2 (llm1_2_02 .. llm1_2_10): volume-price efficiency / liquidity
# =========================================================================== #
def alpha_llm1_2_02(P):
    # RANK_CS_PCT($returns / ($volume / ($adv20 / $close) + 1e-9))
    denom = P["volume"] / (P["adv20"] / P["close"]) + 1e-9
    return ops.rank_cs_pct(P["returns"] / denom)


def alpha_llm1_2_03(P):
    # ZSCORE_CS(DELTA(LOG($adv20), 24) - DELTA(LOG($close), 24))
    diff = ops.delta(np.log(P["adv20"]), 24) - ops.delta(np.log(P["close"]), 24)
    return ops_ext.zscore_cs(diff)


def alpha_llm1_2_04(P):
    # SUMIF($returns, 24, $volume > TS_MEAN($volume, 24)) / (TS_SUM(ABS($returns), 24) + 1e-9)
    cond = P["volume"] > ops.ts_mean(P["volume"], 24)
    num = ops_ext.sumif(P["returns"], 24, cond)
    den = ops.ts_sum(P["returns"].abs(), 24) + 1e-9
    return num / den


def alpha_llm1_2_05(P):
    # RANK_CS_PCT(TS_ARGMAX($volume, 72)) - RANK_CS_PCT(TS_ARGMAX($close, 72))
    return (ops.rank_cs_pct(ops.ts_argmax(P["volume"], 72))
            - ops.rank_cs_pct(ops.ts_argmax(P["close"], 72)))


def alpha_llm1_2_06(P):
    # ZSCORE_CS(SIGN(DELTA($close, 6)) * TS_PCTCHANGE($volume, 6))
    val = np.sign(ops.delta(P["close"], 6)) * _ts_pctchange(P["volume"], 6)
    return ops_ext.zscore_cs(val)


def alpha_llm1_2_07(P):
    # (TS_RANK($volume, 48) > 0.8) ? RANK_CS_PCT($returns) : -RANK_CS_PCT($returns)
    cond = ops.ts_rank(P["volume"], 48) > 0.8
    pos = ops.rank_cs_pct(P["returns"])
    out = (-pos).where(~cond, pos)
    # condition depends on ts_rank(48): warmup NaN -> condition NaN -> pass through NaN
    return out.where(ops.ts_rank(P["volume"], 48).notna())


def alpha_llm1_2_08(P):
    # RANK_CS_PCT($volume * ABS($close - $open) / ($high - $low + 1e-9))
    val = P["volume"] * (P["close"] - P["open"]).abs() / (P["high"] - P["low"] + 1e-9)
    return ops.rank_cs_pct(val)


def alpha_llm1_2_09(P):
    # ZSCORE_CS(CORR($vwap, $volume, 24) - CORR(DELAY($vwap, 24), $volume, 24))
    diff = (ops.corr(P["vwap"], P["volume"], 24)
            - ops.corr(ops.delay(P["vwap"], 24), P["volume"], 24))
    return ops_ext.zscore_cs(diff)


def alpha_llm1_2_10(P):
    # RANK_CS_PCT(TS_STD($volume, 24) / (TS_MEAN($volume, 24) + 1e-9)) * SIGN(DELTA($vwap, 24))
    cv = ops.rank_cs_pct(ops.ts_std(P["volume"], 24) / (ops.ts_mean(P["volume"], 24) + 1e-9))
    return cv * np.sign(ops.delta(P["vwap"], 24))


# =========================================================================== #
# Group 1_3 (llm1_3_01 .. llm1_3_10): return distribution moments / volatility structure
# =========================================================================== #
def alpha_llm1_3_01(P):
    # RANK_CS_PCT(TS_SKEW($returns, 48))
    return ops.rank_cs_pct(ops_ext.ts_skew(P["returns"], 48))


def alpha_llm1_3_02(P):
    # RANK_CS_PCT(TS_STD($returns, 24) / (TS_MEAN(ABS($returns), 72) + 1e-9))
    val = ops.ts_std(P["returns"], 24) / (ops.ts_mean(P["returns"].abs(), 72) + 1e-9)
    return ops.rank_cs_pct(val)


def alpha_llm1_3_03(P):
    # RANK_CS_PCT((PERCENTILE($returns, 72, 0.9) - PERCENTILE($returns, 72, 0.1)) / (TS_MAD($returns, 72) + 1e-9))
    spread = ops_ext.percentile(P["returns"], 72, 0.9) - ops_ext.percentile(P["returns"], 72, 0.1)
    return ops.rank_cs_pct(spread / (ops_ext.ts_mad(P["returns"], 72) + 1e-9))


def alpha_llm1_3_04(P):
    # RANK_CS_PCT(TS_STD(FILTER($returns, $returns > 0), 48) - TS_STD(FILTER($returns, $returns < 0), 48))
    up = ops.ts_std(ops_ext.filter_(P["returns"], P["returns"] > 0), 48)
    dn = ops.ts_std(ops_ext.filter_(P["returns"], P["returns"] < 0), 48)
    return ops.rank_cs_pct(up - dn)


def alpha_llm1_3_05(P):
    # RANK_CS_PCT(TS_KURT($returns, 168))
    return ops.rank_cs_pct(ops_ext.ts_kurt(P["returns"], 168))


def alpha_llm1_3_06(P):
    # RANK_CS_PCT(CORR($returns, TS_STD($returns, 24), 72))
    return ops.rank_cs_pct(ops.corr(P["returns"], ops.ts_std(P["returns"], 24), 72))


def alpha_llm1_3_07(P):
    # ZSCORE_CS(TS_MEAN(ABS($returns - DELAY($returns, 1)), 24)) - ZSCORE_CS(ABS(TS_MEAN($returns, 24)))
    jitter = ops_ext.zscore_cs(ops.ts_mean((P["returns"] - ops.delay(P["returns"], 1)).abs(), 24))
    netmove = ops_ext.zscore_cs(ops.ts_mean(P["returns"], 24).abs())
    return jitter - netmove


def alpha_llm1_3_08(P):
    # RANK_CS_PCT(SKEW_CS(DELTA($returns, 1)) > 0 ? TS_SKEW($returns, 72) : -TS_SKEW($returns, 72))
    cond = ops_ext.skew_cs(ops.delta(P["returns"], 1)) > 0     # cross-sectional scalar broadcast frame -> same truth across the row
    ts = ops_ext.ts_skew(P["returns"], 72)
    inner = (-ts).where(~cond, ts)
    # condition depends on SKEW_CS (cross-sectional skew): a cross-section with <3 non-NaN symbols -> NaN, passes through
    inner = inner.where(ops_ext.skew_cs(ops.delta(P["returns"], 1)).notna())
    return ops.rank_cs_pct(inner)


def alpha_llm1_3_09(P):
    # RANK_CS_PCT((TS_MAX($high, 24) - TS_MIN($low, 24)) / (TS_SUM(ABS($close - $open), 24) + 1e-9))
    rng = ops.ts_max(P["high"], 24) - ops.ts_min(P["low"], 24)
    den = ops.ts_sum((P["close"] - P["open"]).abs(), 24) + 1e-9
    return ops.rank_cs_pct(rng / den)


def alpha_llm1_3_10(P):
    # RANK_CS_PCT(SIGN(DELTA(TS_STD($returns, 24), 24)) * TS_RANK($volume, 24))
    val = np.sign(ops.delta(ops.ts_std(P["returns"], 24), 24)) * ops.ts_rank(P["volume"], 24)
    return ops.rank_cs_pct(val)


# =========================================================================== #
# Group 1_4 (llm1_4_01 .. llm1_4_10): trend / regression / extreme-value time-series
# =========================================================================== #
def alpha_llm1_4_01(P):
    # RANK_CS_PCT(SUBTRACT(HIGHDAY($high, 48), LOWDAY($low, 48)))
    return ops.rank_cs_pct(ops_ext.highday(P["high"], 48) - ops_ext.lowday(P["low"], 48))


def alpha_llm1_4_02(P):
    # RANK_CS_PCT(DIVIDE(REGRESI($close, SEQUENCE(24), 24), TS_STD($close, 24) + 1e-9))
    resid = ops_ext.regresi(P["close"], ops_ext.sequence(24), 24)
    return ops.rank_cs_pct(resid / (ops.ts_std(P["close"], 24) + 1e-9))


def alpha_llm1_4_03(P):
    # RANK_CS_PCT(SUBTRACT(REGBETA($close, SEQUENCE(12), 12), REGBETA($close, SEQUENCE(48), 48)))
    b12 = ops_ext.regbeta(P["close"], ops_ext.sequence(12), 12)
    b48 = ops_ext.regbeta(P["close"], ops_ext.sequence(48), 48)
    return ops.rank_cs_pct(b12 - b48)


def alpha_llm1_4_04(P):
    # RANK_CS_PCT(REGBETA($volume, SEQUENCE(24), 24) / ($adv50 + 1e-9))
    beta = ops_ext.regbeta(P["volume"], ops_ext.sequence(24), 24)
    return ops.rank_cs_pct(beta / (P["adv50"] + 1e-9))


def alpha_llm1_4_05(P):
    # RANK_CS_PCT(DIVIDE(DELTA(TS_ARGMAX($high, 72), 6), TS_STD($returns, 24) + 1e-9))
    num = ops.delta(ops.ts_argmax(P["high"], 72), 6)
    return ops.rank_cs_pct(num / (ops.ts_std(P["returns"], 24) + 1e-9))


def alpha_llm1_4_06(P):
    # RANK_CS_PCT(REGRESI($high, $low, 48))
    return ops.rank_cs_pct(ops_ext.regresi(P["high"], P["low"], 48))


def alpha_llm1_4_07(P):
    # RANK_CS_PCT(DIVIDE(SUBTRACT($close, TS_MEDIAN($close, 168)), TS_MAD($close, 168) + 1e-9))
    num = P["close"] - ops_ext.ts_median(P["close"], 168)
    return ops.rank_cs_pct(num / (ops_ext.ts_mad(P["close"], 168) + 1e-9))


def alpha_llm1_4_08(P):
    # RANK_CS_PCT(SUBTRACT(REGRESI(LOG($volume), SEQUENCE(24), 24), REGRESI(LOG($close), SEQUENCE(24), 24)))
    rv = ops_ext.regresi(np.log(P["volume"]), ops_ext.sequence(24), 24)
    rc = ops_ext.regresi(np.log(P["close"]), ops_ext.sequence(24), 24)
    return ops.rank_cs_pct(rv - rc)


def alpha_llm1_4_09(P):
    # RANK_CS_PCT(CORR($close, SEQUENCE(48), 48) * TS_STD($returns, 48))
    # SEQUENCE(48) as the CORR independent variable -> substituted by a same-shape time ramp frame (corr invariant to positive affine, see _time_ramp_like)
    ramp = _time_ramp_like(P["close"])
    val = ops.corr(P["close"], ramp, 48) * ops.ts_std(P["returns"], 48)
    return ops.rank_cs_pct(val)


def alpha_llm1_4_10(P):
    # RANK_CS_PCT((TS_ARGMIN($low, 24) < TS_ARGMAX($high, 24)) ? TS_STD($returns, 24) : (0 - TS_STD($returns, 24)))
    amn = ops.ts_argmin(P["low"], 24)
    amx = ops.ts_argmax(P["high"], 24)
    std = ops.ts_std(P["returns"], 24)
    cond = amn < amx
    inner = (0 - std).where(~cond, std)
    # condition depends on argmin/argmax (24-window): pass through NaN where either is NaN
    inner = inner.where(amn.notna() & amx.notna())
    return ops.rank_cs_pct(inner)


# =========================================================================== #
# Group 1_5 (llm1_5_01 .. llm1_5_10): relative momentum / liquidity expansion
# =========================================================================== #
def alpha_llm1_5_01(P):
    # RANK_CS_PCT(TS_MEAN($returns, 24)) - RANK_CS_PCT(TS_MEAN($returns, 72))
    return (ops.rank_cs_pct(ops.ts_mean(P["returns"], 24))
            - ops.rank_cs_pct(ops.ts_mean(P["returns"], 72)))


def alpha_llm1_5_02(P):
    # ZSCORE_CS(DELTA($adv20, 24) / ($adv120 + 1e-9))
    return ops_ext.zscore_cs(ops.delta(P["adv20"], 24) / (P["adv120"] + 1e-9))


def alpha_llm1_5_03(P):
    # ZSCORE_CS(ABS($returns) / (($adv20 / ($volume + 1e-9)) + 1e-9))
    denom = (P["adv20"] / (P["volume"] + 1e-9)) + 1e-9
    return ops_ext.zscore_cs(P["returns"].abs() / denom)


def alpha_llm1_5_04(P):
    # CORR(RANK_CS_PCT($close), DELAY(RANK_CS_PCT($close), 24), 72)
    rc = ops.rank_cs_pct(P["close"])
    return ops.corr(rc, ops.delay(rc, 24), 72)


def alpha_llm1_5_05(P):
    # WHERE(TS_RANK($adv30, 48) > 0.7, RANK_CS_PCT(DELTA($vwap, 12)), 0)
    tr = ops.ts_rank(P["adv30"], 48)
    cond = tr > 0.7
    true_val = ops.rank_cs_pct(ops.delta(P["vwap"], 12))
    out = true_val.where(cond, 0.0)
    # condition depends on ts_rank(48): warmup NaN -> condition NaN -> pass through NaN (no forced 0)
    return out.where(tr.notna())


def alpha_llm1_5_06(P):
    # ZSCORE_CS(($high - $close) / (($high - $low) + 1e-9) - ($close - $low) / (($high - $low) + 1e-9))
    hl = (P["high"] - P["low"]) + 1e-9
    val = (P["high"] - P["close"]) / hl - (P["close"] - P["low"]) / hl
    return ops_ext.zscore_cs(val)


def alpha_llm1_5_07(P):
    # RANK_CS_PCT(SIGNED_POWER(TS_SUM(FILTER($returns, $returns > 0), 48) - TS_SUM(FILTER(-$returns, $returns < 0), 48), 0.5))
    up = ops.ts_sum(ops_ext.filter_(P["returns"], P["returns"] > 0), 48)
    dn = ops.ts_sum(ops_ext.filter_(-P["returns"], P["returns"] < 0), 48)
    return ops.rank_cs_pct(ops.signed_power(up - dn, 0.5))


def alpha_llm1_5_08(P):
    # ZSCORE_CS(TS_STD($returns, 24)) - ZSCORE_CS(TS_STD($returns, 168))
    return (ops_ext.zscore_cs(ops.ts_std(P["returns"], 24))
            - ops_ext.zscore_cs(ops.ts_std(P["returns"], 168)))


def alpha_llm1_5_09(P):
    # RANK_CS_PCT(DECAY_LINEAR($returns / (TS_STD($returns, 48) + 1e-9), 24))
    #   - RANK_CS_PCT(DECAY_LINEAR($adv20 / ($adv60 + 1e-9), 24))
    rar = ops.rank_cs_pct(ops.decay_linear(P["returns"] / (ops.ts_std(P["returns"], 48) + 1e-9), 24))
    liq = ops.rank_cs_pct(ops.decay_linear(P["adv20"] / (P["adv60"] + 1e-9), 24))
    return rar - liq


def alpha_llm1_5_10(P):
    # ZSCORE_CS(REGBETA($close / ($vwap + 1e-9), MEDIAN_CS($close / ($vwap + 1e-9)), 48))
    prem = P["close"] / (P["vwap"] + 1e-9)
    beta = ops_ext.regbeta(prem, ops_ext.median_cs(prem), 48)   # x = cross-sectional median premium (wide broadcast frame)
    return ops_ext.zscore_cs(beta)


# --------------------------------------------------------------------------- #
# Contract: ALPHAS = dict[str, callable], key = frozen-list name (llm1_X_YY), value = fn(P)->wide.
# Order matches the frozen list's factors array (49 entries total).
# --------------------------------------------------------------------------- #
ALPHAS = {
    "llm1_1_01": alpha_llm1_1_01,
    "llm1_1_02": alpha_llm1_1_02,
    "llm1_1_03": alpha_llm1_1_03,
    "llm1_1_04": alpha_llm1_1_04,
    "llm1_1_05": alpha_llm1_1_05,
    "llm1_1_06": alpha_llm1_1_06,
    "llm1_1_07": alpha_llm1_1_07,
    "llm1_1_08": alpha_llm1_1_08,
    "llm1_1_09": alpha_llm1_1_09,
    "llm1_1_10": alpha_llm1_1_10,
    "llm1_2_02": alpha_llm1_2_02,
    "llm1_2_03": alpha_llm1_2_03,
    "llm1_2_04": alpha_llm1_2_04,
    "llm1_2_05": alpha_llm1_2_05,
    "llm1_2_06": alpha_llm1_2_06,
    "llm1_2_07": alpha_llm1_2_07,
    "llm1_2_08": alpha_llm1_2_08,
    "llm1_2_09": alpha_llm1_2_09,
    "llm1_2_10": alpha_llm1_2_10,
    "llm1_3_01": alpha_llm1_3_01,
    "llm1_3_02": alpha_llm1_3_02,
    "llm1_3_03": alpha_llm1_3_03,
    "llm1_3_04": alpha_llm1_3_04,
    "llm1_3_05": alpha_llm1_3_05,
    "llm1_3_06": alpha_llm1_3_06,
    "llm1_3_07": alpha_llm1_3_07,
    "llm1_3_08": alpha_llm1_3_08,
    "llm1_3_09": alpha_llm1_3_09,
    "llm1_3_10": alpha_llm1_3_10,
    "llm1_4_01": alpha_llm1_4_01,
    "llm1_4_02": alpha_llm1_4_02,
    "llm1_4_03": alpha_llm1_4_03,
    "llm1_4_04": alpha_llm1_4_04,
    "llm1_4_05": alpha_llm1_4_05,
    "llm1_4_06": alpha_llm1_4_06,
    "llm1_4_07": alpha_llm1_4_07,
    "llm1_4_08": alpha_llm1_4_08,
    "llm1_4_09": alpha_llm1_4_09,
    "llm1_4_10": alpha_llm1_4_10,
    "llm1_5_01": alpha_llm1_5_01,
    "llm1_5_02": alpha_llm1_5_02,
    "llm1_5_03": alpha_llm1_5_03,
    "llm1_5_04": alpha_llm1_5_04,
    "llm1_5_05": alpha_llm1_5_05,
    "llm1_5_06": alpha_llm1_5_06,
    "llm1_5_07": alpha_llm1_5_07,
    "llm1_5_08": alpha_llm1_5_08,
    "llm1_5_09": alpha_llm1_5_09,
    "llm1_5_10": alpha_llm1_5_10,
}
