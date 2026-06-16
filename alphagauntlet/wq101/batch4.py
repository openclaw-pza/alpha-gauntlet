#!/usr/bin/env python3
"""WQ101 batch4 — fourth batch of 20 alphas, verbatim transcription.

Formulas verbatim from the WorldQuant 101 Alphas formula list, the matching "### Alpha#NN" sections
(original formula wins; when pseudo-code conflicts with the original, the original wins, deviation noted in comments).
All time-series/cross-sectional ops go through alphagauntlet.wq101.ops vectorized operators; no rolling.apply in the hot path.

Key range: batch4 uses wq061..wq080 (batch1=wq001-020, batch2=wq021-040, batch3=wq041-060,
batch4 continues wq061-080; globally unique, smoke occupies the wq901-903 range, no clash).

PIT discipline: only rolling/shift/cross-section (axis=1). corr/cov use native pandas (ops.corr/ops.cov),
ts_rank/decay_linear/ts_argmax/ts_argmin use ops. inf cleaned uniformly by runner, this module does not
clean inf and does not fillna warmup NaN.

Operator mapping convention:
- WQ101 `rank(x)`        -> ops.rank_cs(x)        (cross-section axis=1)
- WQ101 `Ts_Rank(x,d)`   -> ops.ts_rank(x,d)
- WQ101 `decay_linear`   -> ops.decay_linear(x,d) (newest bar largest weight)
- WQ101 `ts_argmax/min`  -> ops.ts_argmax/ts_argmin(x,d) (lookback offset, 0=current bar)
- WQ101 `correlation`    -> ops.corr(x,y,d)
- WQ101 `covariance`     -> ops.cov(x,y,d)
- WQ101 `ts_sum/min/max/std/mean/delta/delay/product` -> ops same name
- Fractional windows rounded to integers (original value noted in comments).
"""
import numpy as np

from alphagauntlet.wq101 import ops


# --------------------------------------------------------------------------- #
# Local helper (computes a field only, not in the hot path)
# --------------------------------------------------------------------------- #
def _advN(P, n):
    """Dollar-volume mean advN = volume.rolling(n).mean() * close (matching the contract).
    P pre-computes adv15/20/30/40/50/60/120/180; other windows are computed on the fly here."""
    key = f"adv{n}"
    if key in P:
        return P[key]
    return P["volume"].rolling(n, min_periods=n).mean() * P["close"]


# --------------------------------------------------------------------------- #
# Alpha#071
# max(ts_rank(decay_linear(corr(ts_rank(close,3), ts_rank(adv180,12), 18), 4), 16),
#     ts_rank(decay_linear((rank((low+open) - (vwap+vwap)))^2, 16), 4))
# bracket fix (fidelity review): the original Kakushadze second term is (rank(...))^2, i.e.
# rank-then-square (cross-sectional rank first, then square), not rank((...)^2). The base
# x=low+open-2*vwap is ~83% negative across the panel; the cross-sectional ordering differs before vs
# after squaring (a probe found the inner cross-sectional rank disagreeing at 97% of cells; the full p2
# path corr=-0.71 ~ near sign-flip). The formula-list ammo text writes square-inside-rank, a bracket
# error; the paper's original formula wins here.
# --------------------------------------------------------------------------- #
def alpha_071(P):
    adv180 = P["adv180"]
    p1 = ops.ts_rank(
        ops.decay_linear(
            ops.corr(ops.ts_rank(P["close"], 3), ops.ts_rank(adv180, 12), 18), 4), 16)
    base = P["low"] + P["open"] - 2.0 * P["vwap"]
    inner = ops.rank_cs(base) ** 2          # rank-then-square: cross-sectional rank first, then square
    p2 = ops.ts_rank(ops.decay_linear(inner, 16), 4)
    # max(p1, p2) element-wise
    return np.maximum(p1, p2)


# --------------------------------------------------------------------------- #
# Alpha#072
# rank(decay_linear(corr((high+low)/2, adv40, 9), 10))
#   / rank(decay_linear(corr(ts_rank(vwap,4), ts_rank(volume,19), 7), 3))
# denominator 0 guard: .replace(0, 1e-4) (matching pseudo-code)
# --------------------------------------------------------------------------- #
def alpha_072(P):
    adv40 = P["adv40"]
    mid = (P["high"] + P["low"]) / 2.0
    num = ops.rank_cs(ops.decay_linear(ops.corr(mid, adv40, 9), 10))
    den = ops.rank_cs(ops.decay_linear(
        ops.corr(ops.ts_rank(P["vwap"], 4), ops.ts_rank(P["volume"], 19), 7), 3))
    return num / den.replace(0, 1e-4)


# --------------------------------------------------------------------------- #
# Alpha#073
# -1 * max( rank(decay_linear(delta(vwap,5),3)),
#           ts_rank(decay_linear((-1*delta(0.147155*open+0.852845*low,2)
#                                  /(0.147155*open+0.852845*low)),3),17) )
# --------------------------------------------------------------------------- #
def alpha_073(P):
    p1 = ops.rank_cs(ops.decay_linear(ops.delta(P["vwap"], 5), 3))
    blend = 0.147155 * P["open"] + 0.852845 * P["low"]
    ret_blend = -1.0 * ops.delta(blend, 2) / blend.replace(0, 1e-4)
    p2 = ops.ts_rank(ops.decay_linear(ret_blend, 3), 17)
    return -1.0 * np.maximum(p1, p2)


# --------------------------------------------------------------------------- #
# Alpha#074
# (rank(corr(close, sma(adv30,37), 15))
#   < rank(corr(rank(0.0261661*high + 0.9738339*vwap), rank(volume), 11))) * -1
# --------------------------------------------------------------------------- #
def alpha_074(P):
    adv30 = P["adv30"]
    r1 = ops.rank_cs(ops.corr(P["close"], ops.ts_mean(adv30, 37), 15))
    blend = ops.rank_cs(0.0261661 * P["high"] + 0.9738339 * P["vwap"])
    r2 = ops.rank_cs(ops.corr(blend, ops.rank_cs(P["volume"]), 11))
    return (r1 < r2).astype(float) * -1.0


# --------------------------------------------------------------------------- #
# Alpha#075
# rank(corr(vwap, volume, 4)) < rank(corr(rank(low), rank(adv50), 12))
# conditional, bool -> -1/1 (pseudo-code: (cond)*2-1)
# --------------------------------------------------------------------------- #
def alpha_075(P):
    adv50 = P["adv50"]
    r1 = ops.rank_cs(ops.corr(P["vwap"], P["volume"], 4))
    r2 = ops.rank_cs(ops.corr(ops.rank_cs(P["low"]), ops.rank_cs(adv50), 12))
    return (r1 < r2).astype(float) * 2.0 - 1.0


# --------------------------------------------------------------------------- #
# Alpha#077
# min(rank(decay_linear(((high+low)/2 + high - vwap - high), 20)),
#     rank(decay_linear(corr((high+low)/2, adv40, 3), 6)))
# Note: (high+low)/2 + high - vwap - high = (high+low)/2 - vwap (kept verbatim, equivalent simplification)
# --------------------------------------------------------------------------- #
def alpha_077(P):
    adv40 = P["adv40"]
    mid = (P["high"] + P["low"]) / 2.0
    term1 = mid + P["high"] - P["vwap"] - P["high"]   # verbatim == mid - vwap
    p1 = ops.rank_cs(ops.decay_linear(term1, 20))
    p2 = ops.rank_cs(ops.decay_linear(ops.corr(mid, adv40, 3), 6))
    return np.minimum(p1, p2)


# --------------------------------------------------------------------------- #
# Alpha#078
# rank(corr(ts_sum(0.352233*low + 0.647767*vwap, 20), ts_sum(adv40,20), 7))
#   ^ rank(corr(rank(vwap), rank(volume), 6))
# --------------------------------------------------------------------------- #
def alpha_078(P):
    adv40 = P["adv40"]
    blend = 0.352233 * P["low"] + 0.647767 * P["vwap"]
    base = ops.rank_cs(ops.corr(ops.ts_sum(blend, 20), ops.ts_sum(adv40, 20), 7))
    expo = ops.rank_cs(ops.corr(ops.rank_cs(P["vwap"]), ops.rank_cs(P["volume"]), 6))
    # base^expo: base is a cross-sectional rank (>=1, positive), the power is safe; inf cleaned by runner
    return base.pow(expo)


# --------------------------------------------------------------------------- #
# Alpha#081
# (rank(log(product(rank(rank(corr(vwap, ts_sum(adv10,50), 8))^4), 15)))
#   < rank(corr(rank(vwap), rank(volume), 5))) * -1
# Note: product is a time-series 15-period product (original formula product); the pseudo-code's
#     rolling.sum is an approximation, here we faithfully use ops.product.
#     log on <=0 input gives NaN/-inf -> runner cleans inf, log(0) warmup passes through.
# --------------------------------------------------------------------------- #
def alpha_081(P):
    adv10 = _advN(P, 10)
    base = ops.rank_cs(ops.rank_cs(
        ops.corr(P["vwap"], ops.ts_sum(adv10, 50), 8))) ** 4
    prod = ops.product(base, 15)
    log_prod = ops.rank_cs(np.log(prod))
    r2 = ops.rank_cs(ops.corr(ops.rank_cs(P["vwap"]), ops.rank_cs(P["volume"]), 5))
    return (log_prod < r2).astype(float) * -1.0


# --------------------------------------------------------------------------- #
# Alpha#083
# (rank(delay((high-low)/(ts_sum(close,5)/5), 2)) * rank(rank(volume)))
#   / ((high-low)/(ts_sum(close,5)/5) / (vwap - close))
# denominator guard: ma5 and (vwap-close) use .replace(0, 1e-4) (matching pseudo-code)
# --------------------------------------------------------------------------- #
def alpha_083(P):
    ma5 = ops.ts_sum(P["close"], 5) / 5.0
    hl_ratio = (P["high"] - P["low"]) / ma5.replace(0, 1e-4)
    num = ops.rank_cs(ops.delay(hl_ratio, 2)) * ops.rank_cs(ops.rank_cs(P["volume"]))
    den = hl_ratio / (P["vwap"] - P["close"]).replace(0, 1e-4)
    return num / den.replace(0, 1e-4)


# --------------------------------------------------------------------------- #
# Alpha#084
# ts_rank(vwap - ts_max(vwap, 15), 21) ^ delta(close, 5)
# base is ts_rank ∈ (0,1]; a negative/non-integer exponent may produce inf/NaN -> runner cleans
# --------------------------------------------------------------------------- #
def alpha_084(P):
    base = ops.ts_rank(P["vwap"] - ops.ts_max(P["vwap"], 15), 21)
    expo = ops.delta(P["close"], 5)
    return base.pow(expo)


# --------------------------------------------------------------------------- #
# Alpha#085
# rank(corr(0.876703*high + 0.123297*close, adv30, 10))
#   ^ rank(corr(ts_rank((high+low)/2, 4), ts_rank(volume, 10), 7))
# --------------------------------------------------------------------------- #
def alpha_085(P):
    adv30 = P["adv30"]
    blend = 0.876703 * P["high"] + 0.123297 * P["close"]
    base = ops.rank_cs(ops.corr(blend, adv30, 10))
    mid = (P["high"] + P["low"]) / 2.0
    expo = ops.rank_cs(ops.corr(ops.ts_rank(mid, 4), ops.ts_rank(P["volume"], 10), 7))
    return base.pow(expo)


# --------------------------------------------------------------------------- #
# Alpha#086
# (ts_rank(corr(close, sma(adv20,15), 6), 20) < rank((open + close) - (vwap + open))) * -1
# Note: (open+close)-(vwap+open) = close - vwap (kept verbatim)
# --------------------------------------------------------------------------- #
def alpha_086(P):
    adv20 = P["adv20"]
    r1 = ops.ts_rank(ops.corr(P["close"], ops.ts_mean(adv20, 15), 6), 20)
    r2 = ops.rank_cs((P["open"] + P["close"]) - (P["vwap"] + P["open"]))
    return (r1 < r2).astype(float) * -1.0


# --------------------------------------------------------------------------- #
# Alpha#088
# min(rank(decay_linear(rank(open)+rank(low)-rank(high)-rank(close), 8)),
#     ts_rank(decay_linear(corr(ts_rank(close,8), ts_rank(adv60,21), 8), 7), 3))
# --------------------------------------------------------------------------- #
def alpha_088(P):
    adv60 = P["adv60"]
    ohlc = (ops.rank_cs(P["open"]) + ops.rank_cs(P["low"])
            - ops.rank_cs(P["high"]) - ops.rank_cs(P["close"]))
    p1 = ops.rank_cs(ops.decay_linear(ohlc, 8))
    p2 = ops.ts_rank(ops.decay_linear(
        ops.corr(ops.ts_rank(P["close"], 8), ops.ts_rank(adv60, 21), 8), 7), 3)
    return np.minimum(p1, p2)


# --------------------------------------------------------------------------- #
# Alpha#092
# min(ts_rank(decay_linear(((high+low)/2+close) < (low+open), 15), 19),
#     ts_rank(decay_linear(corr(rank(low), rank(adv30), 8), 7), 7))
# condition < produces bool -> float(0/1) (matching pseudo-code)
# --------------------------------------------------------------------------- #
def alpha_092(P):
    adv30 = P["adv30"]
    cond = (((P["high"] + P["low"]) / 2.0 + P["close"]) < (P["low"] + P["open"])).astype(float)
    p1 = ops.ts_rank(ops.decay_linear(cond, 15), 19)
    p2 = ops.ts_rank(ops.decay_linear(
        ops.corr(ops.rank_cs(P["low"]), ops.rank_cs(adv30), 8), 7), 7)
    return np.minimum(p1, p2)


# --------------------------------------------------------------------------- #
# Alpha#094
# (rank(vwap - ts_min(vwap, 12))
#   ^ ts_rank(corr(ts_rank(vwap,20), ts_rank(adv60,4), 18), 3)) * -1
# --------------------------------------------------------------------------- #
def alpha_094(P):
    adv60 = P["adv60"]
    base = ops.rank_cs(P["vwap"] - ops.ts_min(P["vwap"], 12))
    expo = ops.ts_rank(
        ops.corr(ops.ts_rank(P["vwap"], 20), ops.ts_rank(adv60, 4), 18), 3)
    return base.pow(expo) * -1.0


# --------------------------------------------------------------------------- #
# Alpha#095
# rank(open - ts_min(open, 12))
#   < ts_rank(rank(corr(sma((high+low)/2,19), sma(adv40,19), 13))^5, 12)
# conditional -> (cond)*2-1 (matching pseudo-code)
# --------------------------------------------------------------------------- #
def alpha_095(P):
    adv40 = P["adv40"]
    r1 = ops.rank_cs(P["open"] - ops.ts_min(P["open"], 12))
    mid = (P["high"] + P["low"]) / 2.0
    inner = ops.rank_cs(ops.corr(ops.ts_mean(mid, 19), ops.ts_mean(adv40, 19), 13)) ** 5
    r2 = ops.ts_rank(inner, 12)
    return (r1 < r2).astype(float) * 2.0 - 1.0


# --------------------------------------------------------------------------- #
# Alpha#096
# -1 * max(ts_rank(decay_linear(corr(rank(vwap), rank(volume), 4), 4), 8),
#          ts_rank(decay_linear(ts_argmax(corr(ts_rank(close,7), ts_rank(adv60,4), 4), 13), 14), 13))
# --------------------------------------------------------------------------- #
def alpha_096(P):
    adv60 = P["adv60"]
    p1 = ops.ts_rank(ops.decay_linear(
        ops.corr(ops.rank_cs(P["vwap"]), ops.rank_cs(P["volume"]), 4), 4), 8)
    raw_corr = ops.corr(ops.ts_rank(P["close"], 7), ops.ts_rank(adv60, 4), 4)
    # direction fix: use ts_argmax_fwd to match the reference-library convention; the existing ts_argmax
    # 'lookback' convention is overall sign-flipped vs the reference library / pseudo-code (a probe found
    # corr=-0.9998), which would flip this branch's signal direction.
    argmax13 = ops.ts_argmax_fwd(raw_corr, 13)
    p2 = ops.ts_rank(ops.decay_linear(argmax13, 14), 13)
    return -1.0 * np.maximum(p1, p2)


# --------------------------------------------------------------------------- #
# Alpha#098
# rank(decay_linear(corr(vwap, sma(adv5,26), 5), 7))
#  - rank(decay_linear(ts_rank(ts_argmin(corr(rank(open), rank(adv15), 21), 9), 7), 8))
# --------------------------------------------------------------------------- #
def alpha_098(P):
    adv5 = _advN(P, 5)
    adv15 = P["adv15"]
    p1 = ops.rank_cs(ops.decay_linear(
        ops.corr(P["vwap"], ops.ts_mean(adv5, 26), 5), 7))
    base_corr = ops.corr(ops.rank_cs(P["open"]), ops.rank_cs(adv15), 21)
    # direction fix: use ts_argmin_fwd to match the reference-library convention (same as #096); the
    # existing ts_argmin 'lookback' convention is overall sign-flipped vs the reference library, which
    # would flip this branch's signal direction.
    argmin9 = ops.ts_argmin_fwd(base_corr, 9)
    p2 = ops.rank_cs(ops.decay_linear(ops.ts_rank(argmin9, 7), 8))
    return p1 - p2


# --------------------------------------------------------------------------- #
# Alpha#099
# (rank(corr(ts_sum((high+low)/2, 20), ts_sum(adv60,20), 9)) < rank(corr(low, volume, 6))) * -1
# --------------------------------------------------------------------------- #
def alpha_099(P):
    adv60 = P["adv60"]
    mid = (P["high"] + P["low"]) / 2.0
    r1 = ops.rank_cs(ops.corr(ops.ts_sum(mid, 20), ops.ts_sum(adv60, 20), 9))
    r2 = ops.rank_cs(ops.corr(P["low"], P["volume"], 6))
    return (r1 < r2).astype(float) * -1.0


# --------------------------------------------------------------------------- #
# Alpha#101
# (close - open) / ((high - low) + 0.001)
# --------------------------------------------------------------------------- #
def alpha_101(P):
    return (P["close"] - P["open"]) / ((P["high"] - P["low"]) + 0.001)


# Contract: keys wq{id} style, batch4 continues wq061..wq080 (globally unique).
ALPHAS = {
    "wq061": alpha_071,   # Alpha#071
    "wq062": alpha_072,   # Alpha#072
    "wq063": alpha_073,   # Alpha#073
    "wq064": alpha_074,   # Alpha#074
    "wq065": alpha_075,   # Alpha#075
    "wq066": alpha_077,   # Alpha#077
    "wq067": alpha_078,   # Alpha#078
    "wq068": alpha_081,   # Alpha#081
    "wq069": alpha_083,   # Alpha#083
    "wq070": alpha_084,   # Alpha#084
    "wq071": alpha_085,   # Alpha#085
    "wq072": alpha_086,   # Alpha#086
    "wq073": alpha_088,   # Alpha#088
    "wq074": alpha_092,   # Alpha#092
    "wq075": alpha_094,   # Alpha#094
    "wq076": alpha_095,   # Alpha#095
    "wq077": alpha_096,   # Alpha#096
    "wq078": alpha_098,   # Alpha#098
    "wq079": alpha_099,   # Alpha#099
    "wq080": alpha_101,   # Alpha#101
}
