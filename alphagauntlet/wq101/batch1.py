#!/usr/bin/env python3
r"""WQ101 batch1 — Alpha#001..#021 (21 alphas), verbatim transcription from the WorldQuant 101 Alphas formula list.

Contract (see batch_api):
- Each fn(P) -> wide DataFrame(index=time, columns=symbols), P=panel_io.load_field_panel() output.
- Keys wq001..wq021 (globally unique, no clash with smoke's wq901-903 or other batches).
- Only ops.py vectorized operators; rolling corr/cov use ops.corr/ops.cov (native pandas).
- NaN/inf policy: warmup NaN passes through, no fillna; inf cleaned uniformly by runner; a
  potentially-zero denominator gets the formula-list .replace(0, 1e-4) small-value guard (denominator only).

Transcription discipline: windows/coefficients/thresholds are not changed. When pseudo-code conflicts
with the original formula, the original formula wins, and the deviation is noted in a comment.
Fractional windows are rounded to integers (none in this batch). Conditional alphas are implemented
faithfully; degeneracy is left to runner guards.

Field convention note (adv units):
- panel_io provides P["adv20"] = volume.rolling(20).mean() * close (dollar volume).
- The WQ101 adv20 in the original formulas = average daily dollar volume, matching the panel
  convention -> use P["adv20"] directly. (Some pseudo-code annotations write it as a plain volume
  mean, which conflicts with the original formula; per the rule, the original formula wins, use P["adv20"].)
"""
import numpy as np

from alphagauntlet.wq101 import ops


# --------------------------------------------------------------------------- #
# Alpha#001
#   rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5
# --------------------------------------------------------------------------- #
def alpha_wq001(P):
    returns = P["returns"]
    # inner: where returns<0 take stddev(returns,20), else close
    inner = P["close"].where(~(returns < 0.0), ops.ts_std(returns, 20))
    sp = ops.signed_power(inner, 2.0)              # SignedPower(., 2.)
    arg = ops.ts_argmax(sp, 5)                     # Ts_ArgMax(., 5)
    return ops.rank_cs(arg) - 0.5                  # rank(.) - 0.5  (cross-sectional rank)


# --------------------------------------------------------------------------- #
# Alpha#002
#   -1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6)
# --------------------------------------------------------------------------- #
def alpha_wq002(P):
    x = ops.rank_cs(ops.delta(np.log(P["volume"]), 2))
    y = ops.rank_cs((P["close"] - P["open"]) / P["open"])
    return -1.0 * ops.corr(x, y, 6)


# --------------------------------------------------------------------------- #
# Alpha#003
#   -1 * correlation(rank(open), rank(volume), 10)
# --------------------------------------------------------------------------- #
def alpha_wq003(P):
    return -1.0 * ops.corr(ops.rank_cs(P["open"]), ops.rank_cs(P["volume"]), 10)


# --------------------------------------------------------------------------- #
# Alpha#004
#   -1 * Ts_Rank(rank(low), 9)
# --------------------------------------------------------------------------- #
def alpha_wq004(P):
    return -1.0 * ops.ts_rank(ops.rank_cs(P["low"]), 9)


# --------------------------------------------------------------------------- #
# Alpha#005
#   rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap))))
# --------------------------------------------------------------------------- #
def alpha_wq005(P):
    vwap = P["vwap"]
    a = ops.rank_cs(P["open"] - (ops.ts_sum(vwap, 10) / 10.0))
    b = -1.0 * ops.rank_cs(P["close"] - vwap).abs()
    return a * b


# --------------------------------------------------------------------------- #
# Alpha#006
#   -1 * correlation(open, volume, 10)
# --------------------------------------------------------------------------- #
def alpha_wq006(P):
    return -1.0 * ops.corr(P["open"], P["volume"], 10)


# --------------------------------------------------------------------------- #
# Alpha#007
#   (adv20 < volume) ? (-1 * ts_rank(abs(delta(close, 7)), 60) * sign(delta(close, 7))) : -1
# Note: adv20 takes P["adv20"] (dollar ADV, matching the original adv20 units; the pseudo-code
#       annotation's plain volume mean conflicts with the original formula, so the original wins).
#       Condition false -> -1.
# --------------------------------------------------------------------------- #
def alpha_wq007(P):
    adv20 = P["adv20"]
    delta_c = ops.delta(P["close"], 7)
    branch = -1.0 * ops.ts_rank(delta_c.abs(), 60) * np.sign(delta_c)
    cond = adv20 < P["volume"]
    # condition true -> branch, false -> -1 (NaN condition passes through to NaN)
    out = branch.where(cond, -1.0)
    return out.where(cond.notna())                 # keep NaN where adv20/volume warmup is NaN


# --------------------------------------------------------------------------- #
# Alpha#008
#   -1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))
# --------------------------------------------------------------------------- #
def alpha_wq008(P):
    combo = ops.ts_sum(P["open"], 5) * ops.ts_sum(P["returns"], 5)
    return -1.0 * ops.rank_cs(combo - ops.delay(combo, 10))


# --------------------------------------------------------------------------- #
# Alpha#009
#   (0 < ts_min(delta(close,1),5)) ? delta(close,1)
#     : ((ts_max(delta(close,1),5) < 0) ? delta(close,1) : (-1 * delta(close,1)))
# --------------------------------------------------------------------------- #
def alpha_wq009(P):
    d1 = ops.delta(P["close"], 1)
    cond_up = ops.ts_min(d1, 5) > 0.0
    cond_dn = ops.ts_max(d1, 5) < 0.0
    out = (-1.0 * d1).where(~(cond_up | cond_dn), d1)
    # condition depends on ts_min/ts_max (5-window) and d1; keep NaN where any is NaN
    valid = ops.ts_min(d1, 5).notna() & ops.ts_max(d1, 5).notna() & d1.notna()
    return out.where(valid)


# --------------------------------------------------------------------------- #
# Alpha#010
#   rank((0 < ts_min(delta(close,1),4)) ? delta(close,1)
#         : ((ts_max(delta(close,1),4) < 0) ? delta(close,1) : (-1 * delta(close,1))))
# --------------------------------------------------------------------------- #
def alpha_wq010(P):
    d1 = ops.delta(P["close"], 1)
    cond_up = ops.ts_min(d1, 4) > 0.0
    cond_dn = ops.ts_max(d1, 4) < 0.0
    inner = (-1.0 * d1).where(~(cond_up | cond_dn), d1)
    valid = ops.ts_min(d1, 4).notna() & ops.ts_max(d1, 4).notna() & d1.notna()
    inner = inner.where(valid)
    return ops.rank_cs(inner)


# --------------------------------------------------------------------------- #
# Alpha#011
#   (rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3))
# --------------------------------------------------------------------------- #
def alpha_wq011(P):
    spread = P["vwap"] - P["close"]
    a = ops.rank_cs(ops.ts_max(spread, 3)) + ops.rank_cs(ops.ts_min(spread, 3))
    return a * ops.rank_cs(ops.delta(P["volume"], 3))


# --------------------------------------------------------------------------- #
# Alpha#012
#   sign(delta(volume, 1)) * (-1 * delta(close, 1))
# --------------------------------------------------------------------------- #
def alpha_wq012(P):
    return np.sign(ops.delta(P["volume"], 1)) * (-1.0 * ops.delta(P["close"], 1))


# --------------------------------------------------------------------------- #
# Alpha#013
#   -1 * rank(covariance(rank(close), rank(volume), 5))
# --------------------------------------------------------------------------- #
def alpha_wq013(P):
    rc = ops.rank_cs(P["close"])
    rv = ops.rank_cs(P["volume"])
    return -1.0 * ops.rank_cs(ops.cov(rc, rv, 5))


# --------------------------------------------------------------------------- #
# Alpha#014
#   (-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)
# --------------------------------------------------------------------------- #
def alpha_wq014(P):
    ret_chg = ops.rank_cs(ops.delta(P["returns"], 3))
    ov_corr = ops.corr(P["open"], P["volume"], 10)
    return (-1.0 * ret_chg) * ov_corr


# --------------------------------------------------------------------------- #
# Alpha#015
#   -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)
# --------------------------------------------------------------------------- #
def alpha_wq015(P):
    c = ops.corr(ops.rank_cs(P["high"]), ops.rank_cs(P["volume"]), 3)
    return -1.0 * ops.ts_sum(ops.rank_cs(c), 3)


# --------------------------------------------------------------------------- #
# Alpha#016
#   -1 * rank(covariance(rank(high), rank(volume), 5))
# --------------------------------------------------------------------------- #
def alpha_wq016(P):
    return -1.0 * ops.rank_cs(ops.cov(ops.rank_cs(P["high"]), ops.rank_cs(P["volume"]), 5))


# --------------------------------------------------------------------------- #
# Alpha#017
#   (-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1)) * rank(ts_rank((volume / adv20), 5))
# Note: adv20 uses P["adv20"] (dollar ADV); volume/adv20 has mismatched units as in the original
#       formula (volume is coin amount, adv20 is dollar amount), but ts_rank only takes time-series
#       relative position and a constant scale factor does not change the time-series rank -> faithful to original.
# --------------------------------------------------------------------------- #
def alpha_wq017(P):
    p1 = -1.0 * ops.rank_cs(ops.ts_rank(P["close"], 10))
    accel = ops.delta(ops.delta(P["close"], 1), 1)
    p2 = ops.rank_cs(accel)
    rel_vol = P["volume"] / P["adv20"]
    p3 = ops.rank_cs(ops.ts_rank(rel_vol, 5))
    return p1 * p2 * p3


# --------------------------------------------------------------------------- #
# Alpha#018
#   -1 * rank((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))
# --------------------------------------------------------------------------- #
def alpha_wq018(P):
    co_diff = P["close"] - P["open"]
    term1 = ops.ts_std(co_diff.abs(), 5)
    term3 = ops.corr(P["close"], P["open"], 10)
    return -1.0 * ops.rank_cs((term1 + co_diff) + term3)


# --------------------------------------------------------------------------- #
# Alpha#019
#   (-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250))))
# --------------------------------------------------------------------------- #
def alpha_wq019(P):
    price_chg = (P["close"] - ops.delay(P["close"], 7)) + ops.delta(P["close"], 7)
    long_ret = ops.rank_cs(1.0 + ops.ts_sum(P["returns"], 250))
    return (-1.0 * np.sign(price_chg)) * (1.0 + long_ret)


# --------------------------------------------------------------------------- #
# Alpha#020
#   -1 * rank(open - delay(high, 1)) * rank(open - delay(close, 1)) * rank(open - delay(low, 1))
# --------------------------------------------------------------------------- #
def alpha_wq020(P):
    r1 = ops.rank_cs(P["open"] - ops.delay(P["high"], 1))
    r2 = ops.rank_cs(P["open"] - ops.delay(P["close"], 1))
    r3 = ops.rank_cs(P["open"] - ops.delay(P["low"], 1))
    return -1.0 * r1 * r2 * r3


# --------------------------------------------------------------------------- #
# Alpha#021
#   ((sma(close,8)+stddev(close,8) < sma(close,2)) OR (sma(volume,20)/volume < 1)) ? -1 : 1
# Note: sma(x,n) = simple moving average = ts_mean(x,n). Returns +/-1 conditional; degeneracy left to runner guards.
# --------------------------------------------------------------------------- #
def alpha_wq021(P):
    close, vol = P["close"], P["volume"]
    sma8 = ops.ts_mean(close, 8)
    std8 = ops.ts_std(close, 8)
    sma2 = ops.ts_mean(close, 2)
    sma20v = ops.ts_mean(vol, 20)
    cond1 = (sma8 + std8) < sma2
    cond2 = (sma20v / vol) < 1.0
    hit = cond1 | cond2                            # True -> -1, False -> 1
    res = hit.astype(float) * -2.0 + 1.0           # True->-1.0, False->1.0
    # keep NaN over the longest-window warmup (sma20v/std8)
    valid = sma8.notna() & std8.notna() & sma2.notna() & sma20v.notna() & vol.notna()
    return res.where(valid)


# Contract: ALPHAS = dict[str, callable], key=wq{three-digit id}, value=fn(P)->wide DataFrame
ALPHAS = {
    "wq001": alpha_wq001,   # Alpha#001
    "wq002": alpha_wq002,   # Alpha#002
    "wq003": alpha_wq003,   # Alpha#003
    "wq004": alpha_wq004,   # Alpha#004
    "wq005": alpha_wq005,   # Alpha#005
    "wq006": alpha_wq006,   # Alpha#006
    "wq007": alpha_wq007,   # Alpha#007
    "wq008": alpha_wq008,   # Alpha#008
    "wq009": alpha_wq009,   # Alpha#009
    "wq010": alpha_wq010,   # Alpha#010
    "wq011": alpha_wq011,   # Alpha#011
    "wq012": alpha_wq012,   # Alpha#012
    "wq013": alpha_wq013,   # Alpha#013
    "wq014": alpha_wq014,   # Alpha#014
    "wq015": alpha_wq015,   # Alpha#015
    "wq016": alpha_wq016,   # Alpha#016
    "wq017": alpha_wq017,   # Alpha#017
    "wq018": alpha_wq018,   # Alpha#018
    "wq019": alpha_wq019,   # Alpha#019
    "wq020": alpha_wq020,   # Alpha#020
    "wq021": alpha_wq021,   # Alpha#021
}
