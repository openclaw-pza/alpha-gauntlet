#!/usr/bin/env python3
"""WQ101 batch3 — third batch of 21 A-type alphas, faithful transcription.

Formulas verbatim from the WorldQuant 101 Alphas formula list, the matching "### Alpha#NN" sections.
Implementations always use the vectorized operators in alphagauntlet/wq101/ops.py; no rolling.apply in the hot path.

Key convention (batch_api contract): globally unique, no repeats. batch1 occupies wq001.. (~20),
batch2 continues (~wq021..040), this batch3 occupies wq041..wq061 (21 alphas), smoke uses the wq9xx range.

Hard rules:
- Verbatim transcription, windows/coefficients/thresholds unchanged. Fractional windows rounded to integers (original value noted in comments).
- When pseudo-code conflicts with the original formula, the original wins, deviation noted in comments.
- Conditional alphas implemented faithfully; degeneracy left to runner guards; power-op inf cleaned uniformly by runner.
- A potentially-zero denominator uses the formula-list .replace(0, 1e-4) small-value guard (denominator only).
- No fillna of warmup NaN; no inf cleaning inside fn (runner uniformly maps +/-inf->NaN).

ts_argmax semantics note (Alpha#057 / #060):
  The WQ101 original operator ts_argmax(x, d) = "how long ago the max occurred" (lookback offset from
  the current bar), and ops.ts_argmax has exactly that semantics (0=current bar, d-1=window oldest).
  The pseudo-code uses bare np.argmax(x) returning the in-window position [0..d-1] (0=oldest), in the
  opposite direction to the original operator. **The original formula (WQ101 operator) wins**, so
  ops.ts_argmax is used. After a subsequent rank the two have opposite ordering directions; this is an
  intentional deviation from the pseudo-code.
"""
import numpy as np

from alphagauntlet.wq101 import ops


# --------------------------------------------------------------------------- #
# Internal helper
# --------------------------------------------------------------------------- #
def _cond_neg(cond, mask_nan):
    """Common tail for conditional alphas of the form (rank(A) < rank(B)) * -1.

    Per the pseudo-code `.astype(int) * -1`: cond True -> -1, False -> 0.
    Additionally uses mask_nan (cells where any rank input is NaN) to restore warmup NaN, avoiding
    warmup being filled with a spurious 0 (more PIT-honest; does not affect truncation invariance,
    since warmup is identical on both sides).
    cond / mask_nan are both same-shape wide DataFrames (bool).
    """
    out = cond.astype(float) * -1.0   # True->-1.0, False->0.0
    return out.where(~mask_nan, other=np.nan)


# --------------------------------------------------------------------------- #
# Alpha#043
# original: ts_rank(volume / adv20, 20) * ts_rank((-1 * delta(close, 7)), 8)
# --------------------------------------------------------------------------- #
def alpha_043(P):
    a = ops.ts_rank(P["volume"] / P["adv20"], 20)
    b = ops.ts_rank(-1.0 * ops.delta(P["close"], 7), 8)
    return a * b


# --------------------------------------------------------------------------- #
# Alpha#044
# original: -1 * correlation(high, rank(volume), 5)
# --------------------------------------------------------------------------- #
def alpha_044(P):
    rv = ops.rank_cs(P["volume"])          # rank(volume) cross-sectional
    return -1.0 * ops.corr(P["high"], rv, 5)


# --------------------------------------------------------------------------- #
# Alpha#045
# original: -1 * rank(sma(delay(close,5),20)) * correlation(close,volume,2)
#         * rank(correlation(sum(close,5), sum(close,20), 2))
# --------------------------------------------------------------------------- #
def alpha_045(P):
    p1 = ops.rank_cs(ops.ts_mean(ops.delay(P["close"], 5), 20))
    p2 = ops.corr(P["close"], P["volume"], 2)
    p3 = ops.rank_cs(ops.corr(ops.ts_sum(P["close"], 5), ops.ts_sum(P["close"], 20), 2))
    return -1.0 * p1 * p2 * p3


# --------------------------------------------------------------------------- #
# Alpha#046
# original: (accel > 0.25) ? -1 : ((accel < 0) ? 1 : -1*delta(close,1))
#   accel = (delay(close,20)-delay(close,10))/10 - (delay(close,10)-close)/10
# fix (fidelity review #3): the middle-branch threshold should be accel<0 (original formula), not
#   accel<-0.1 (-0.1 is Alpha#049's threshold; the formula-list text leaked it onto 046).
# --------------------------------------------------------------------------- #
def alpha_046(P):
    c = P["close"]
    accel = (ops.delay(c, 20) - ops.delay(c, 10)) / 10.0 - (ops.delay(c, 10) - c) / 10.0
    base = -1.0 * ops.delta(c, 1)
    out = base.copy()
    out = out.mask(accel < 0.0, 1.0)      # original middle branch: accel<0 -> 1 (not -0.1)
    out = out.mask(accel > 0.25, -1.0)    # accel>0.25 -> -1 (highest priority, masked last)
    return out


# --------------------------------------------------------------------------- #
# Alpha#047
# original: ((rank(1/close) * volume / adv20) * (high * rank(high-close) / (sma(high,5)/5)))
#         - rank(vwap - delay(vwap,5))
# --------------------------------------------------------------------------- #
def alpha_047(P):
    # fix (fidelity review #5/#7): the original denominator sum(high,5)/5 == mean(high,5).
    # The old code ts_mean(high,5)/5.0 divided by an extra 5, amplifying p1 5x and changing the p1/p2
    # relative weight. Removed the spurious /5.0.
    p1 = ((ops.rank_cs(1.0 / P["close"]) * P["volume"] / P["adv20"])
          * (P["high"] * ops.rank_cs(P["high"] - P["close"])
             / ops.ts_mean(P["high"], 5)))
    p2 = ops.rank_cs(P["vwap"] - ops.delay(P["vwap"], 5))
    return p1 - p2


# --------------------------------------------------------------------------- #
# Alpha#049
# original: (accel < -0.1) ? 1 : -1*delta(close,1)
# --------------------------------------------------------------------------- #
def alpha_049(P):
    c = P["close"]
    accel = (ops.delay(c, 20) - ops.delay(c, 10)) / 10.0 - (ops.delay(c, 10) - c) / 10.0
    out = (-1.0 * ops.delta(c, 1))
    out = out.mask(accel < -0.1, 1.0)
    return out


# --------------------------------------------------------------------------- #
# Alpha#050
# original: -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)
# --------------------------------------------------------------------------- #
def alpha_050(P):
    rv = ops.rank_cs(P["volume"])
    rvw = ops.rank_cs(P["vwap"])
    c = ops.corr(rv, rvw, 5)
    return -1.0 * ops.ts_max(ops.rank_cs(c), 5)


# --------------------------------------------------------------------------- #
# Alpha#051
# original: (accel < -0.05) ? 1 : -1*delta(close,1)
# --------------------------------------------------------------------------- #
def alpha_051(P):
    c = P["close"]
    accel = (ops.delay(c, 20) - ops.delay(c, 10)) / 10.0 - (ops.delay(c, 10) - c) / 10.0
    out = (-1.0 * ops.delta(c, 1))
    out = out.mask(accel < -0.05, 1.0)
    return out


# --------------------------------------------------------------------------- #
# Alpha#052
# original: ((-1 * delta(ts_min(low,5),5)) * rank((ts_sum(returns,240)-ts_sum(returns,20))/220))
#         * ts_rank(volume, 5)
# --------------------------------------------------------------------------- #
def alpha_052(P):
    low_move = ops.delta(ops.ts_min(P["low"], 5), 5)
    long_ret = (ops.ts_sum(P["returns"], 240) - ops.ts_sum(P["returns"], 20)) / 220.0
    return (-1.0 * low_move) * ops.rank_cs(long_ret) * ops.ts_rank(P["volume"], 5)


# --------------------------------------------------------------------------- #
# Alpha#053
# original: -1 * delta((close - low - (high - close)) / (close - low), 9)
# --------------------------------------------------------------------------- #
def alpha_053(P):
    cl = (P["close"] - P["low"]).replace(0, 1e-4)   # denominator small-value guard (formula list)
    candle = (P["close"] - P["low"] - (P["high"] - P["close"])) / cl
    return -1.0 * ops.delta(candle, 9)


# --------------------------------------------------------------------------- #
# Alpha#054
# original: -1 * (low - close) * open^5 / ((low - high) * close^5)
# --------------------------------------------------------------------------- #
def alpha_054(P):
    lh = (P["low"] - P["high"]).replace(0, -1e-4)   # denominator small-value guard (formula list, negative)
    return -1.0 * (P["low"] - P["close"]) * P["open"].pow(5) / (lh * P["close"].pow(5))


# --------------------------------------------------------------------------- #
# Alpha#055
# original: -1 * correlation(rank((close - ts_min(low,12)) / (ts_max(high,12) - ts_min(low,12))),
#                          rank(volume), 6)
# --------------------------------------------------------------------------- #
def alpha_055(P):
    denom = (ops.ts_max(P["high"], 12) - ops.ts_min(P["low"], 12)).replace(0, 1e-4)
    pos = ops.rank_cs((P["close"] - ops.ts_min(P["low"], 12)) / denom)
    rv = ops.rank_cs(P["volume"])
    return -1.0 * ops.corr(pos, rv, 6)


# --------------------------------------------------------------------------- #
# Alpha#057
# original: -(close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)
# Deviation: ts_argmax uses the WQ101 original operator semantics (ops.ts_argmax), not the pseudo-code's bare np.argmax (see module header).
# --------------------------------------------------------------------------- #
def alpha_057(P):
    am = ops.rank_cs(ops.ts_argmax(P["close"], 30))
    denom = ops.decay_linear(am, 2)
    return -1.0 * (P["close"] - P["vwap"]) / denom


# --------------------------------------------------------------------------- #
# Alpha#059
# original: -1 * Ts_Rank(decay_linear(correlation(vwap, volume, 4.25197), 16.2289), 8.19648)
#   (0.728317*vwap + 0.271683*vwap = vwap; IndNeutralize omitted on a 20-symbol system, see formula-list adaptation note)
#   windows: 4.25197->4, 16.2289->16, 8.19648->8 (rounded)
# --------------------------------------------------------------------------- #
def alpha_059(P):
    c = ops.corr(P["vwap"], P["volume"], 4)        # 4.25197 -> 4
    dl = ops.decay_linear(c, 16)                    # 16.2289 -> 16
    return -1.0 * ops.ts_rank(dl, 8)                # 8.19648 -> 8


# --------------------------------------------------------------------------- #
# Alpha#060
# original: -(2 * scale(rank(((close-low)-(high-close)) * volume / (high-low)))
#          - scale(rank(ts_argmax(close, 10))))
# Deviation: ts_argmax uses ops.ts_argmax (WQ101 original operator semantics), see module header.
# --------------------------------------------------------------------------- #
def alpha_060(P):
    hl = (P["high"] - P["low"]).replace(0, 1e-4)
    candle_vol = ((P["close"] - P["low"]) - (P["high"] - P["close"])) * P["volume"] / hl
    term1 = ops.scale(ops.rank_cs(candle_vol))
    term2 = ops.scale(ops.rank_cs(ops.ts_argmax(P["close"], 10)))
    return -1.0 * (2.0 * term1 - term2)


# --------------------------------------------------------------------------- #
# Alpha#061
# original: rank(vwap - ts_min(vwap, 16)) < rank(correlation(vwap, adv180, 18))
#   returns bool -> per formula list to -1/1: .astype(int)*2 - 1
# --------------------------------------------------------------------------- #
def alpha_061(P):
    rank_spread = ops.rank_cs(P["vwap"] - ops.ts_min(P["vwap"], 16))
    rank_corr = ops.rank_cs(ops.corr(P["vwap"], P["adv180"], 18))
    cond = rank_spread < rank_corr
    mask_nan = rank_spread.isna() | rank_corr.isna()
    out = cond.astype(float) * 2.0 - 1.0          # True->1, False->-1 (formula list)
    return out.where(~mask_nan, other=np.nan)


# --------------------------------------------------------------------------- #
# Alpha#062
# original: (rank(corr(vwap, sma(adv20,22), 10))
#          < rank((rank(open)+rank(open)) < (rank((high+low)/2)+rank(high)))) * -1
# --------------------------------------------------------------------------- #
def alpha_062(P):
    r_corr = ops.rank_cs(ops.corr(P["vwap"], ops.ts_mean(P["adv20"], 22), 10))
    r_open = ops.rank_cs(P["open"])
    r_mid = ops.rank_cs((P["high"] + P["low"]) / 2.0)
    r_high = ops.rank_cs(P["high"])
    inner_cond = (r_open + r_open) < (r_mid + r_high)
    inner_mask = r_open.isna() | r_mid.isna() | r_high.isna()
    inner = inner_cond.astype(float).where(~inner_mask, other=np.nan)
    r_inner = ops.rank_cs(inner)
    cond = r_corr < r_inner
    mask_nan = r_corr.isna() | r_inner.isna()
    return _cond_neg(cond, mask_nan)


# --------------------------------------------------------------------------- #
# Alpha#064
# original: (rank(corr(sma(0.178404*open + 0.821596*low, 13), sma(adv120,13), 17))
#          < rank(delta(0.178404*(high+low)/2 + 0.821596*vwap, 3.69741))) * -1
#   3.69741 -> 4 (rounded)
# --------------------------------------------------------------------------- #
def alpha_064(P):
    w1 = 0.178404 * P["open"] + 0.821596 * P["low"]
    r1 = ops.rank_cs(ops.corr(ops.ts_mean(w1, 13), ops.ts_mean(P["adv120"], 13), 17))
    w2 = 0.178404 * (P["high"] + P["low"]) / 2.0 + 0.821596 * P["vwap"]
    r2 = ops.rank_cs(ops.delta(w2, 4))            # 3.69741 -> 4
    cond = r1 < r2
    mask_nan = r1.isna() | r2.isna()
    return _cond_neg(cond, mask_nan)


# --------------------------------------------------------------------------- #
# Alpha#065
# original: (rank(corr(0.00817205*open + 0.99182795*vwap, sma(adv60,9), 6))
#          < rank(open - ts_min(open,14))) * -1
# --------------------------------------------------------------------------- #
def alpha_065(P):
    blend = 0.00817205 * P["open"] + 0.99182795 * P["vwap"]
    r1 = ops.rank_cs(ops.corr(blend, ops.ts_mean(P["adv60"], 9), 6))
    r2 = ops.rank_cs(P["open"] - ops.ts_min(P["open"], 14))
    cond = r1 < r2
    mask_nan = r1.isna() | r2.isna()
    return _cond_neg(cond, mask_nan)


# --------------------------------------------------------------------------- #
# Alpha#066
# original: (rank(decay_linear(delta(vwap, 4), 7))
#          + ts_rank(decay_linear(((low*0.96633 + low*(1-0.96633)) - vwap)
#                                 / (open - (high+low)/2), 11), 7)) * -1
#   Note: 0.96633*low + (1-0.96633)*low = low (the original is an identity, use low directly)
# --------------------------------------------------------------------------- #
def alpha_066(P):
    p1 = ops.rank_cs(ops.decay_linear(ops.delta(P["vwap"], 4), 7))
    mid = (P["high"] + P["low"]) / 2.0
    denom = (P["open"] - mid).replace(0, 1e-4)    # denominator small-value guard (formula list)
    numerator = (P["low"] - P["vwap"]) / denom    # low*0.96633 + low*0.03367 = low
    p2 = ops.ts_rank(ops.decay_linear(numerator, 11), 7)
    return (p1 + p2) * -1.0


# --------------------------------------------------------------------------- #
# Alpha#068
# original: (ts_rank(corr(rank(high), rank(adv15), 9), 14)
#          < rank(delta(0.518371*close + 0.481629*low, 1.06157))) * -1
#   1.06157 -> 1 (rounded)
# --------------------------------------------------------------------------- #
def alpha_068(P):
    rh = ops.rank_cs(P["high"])
    radv = ops.rank_cs(P["adv15"])
    r_corr = ops.ts_rank(ops.corr(rh, radv, 9), 14)
    blend = 0.518371 * P["close"] + 0.481629 * P["low"]
    r_delta = ops.rank_cs(ops.delta(blend, 1))    # 1.06157 -> 1
    cond = r_corr < r_delta
    mask_nan = r_corr.isna() | r_delta.isna()
    return _cond_neg(cond, mask_nan)


# --------------------------------------------------------------------------- #
# Keys: wq041..wq061 (21, no clash with batch1/2), value -> Alpha#NN
# --------------------------------------------------------------------------- #
ALPHAS = {
    "wq041": alpha_043,   # Alpha#043
    "wq042": alpha_044,   # Alpha#044
    "wq043": alpha_045,   # Alpha#045
    "wq044": alpha_046,   # Alpha#046
    "wq045": alpha_047,   # Alpha#047
    "wq046": alpha_049,   # Alpha#049
    "wq047": alpha_050,   # Alpha#050
    "wq048": alpha_051,   # Alpha#051
    "wq049": alpha_052,   # Alpha#052
    "wq050": alpha_053,   # Alpha#053
    "wq051": alpha_054,   # Alpha#054
    "wq052": alpha_055,   # Alpha#055
    "wq053": alpha_057,   # Alpha#057
    "wq054": alpha_059,   # Alpha#059
    "wq055": alpha_060,   # Alpha#060
    "wq056": alpha_061,   # Alpha#061
    "wq057": alpha_062,   # Alpha#062
    "wq058": alpha_064,   # Alpha#064
    "wq059": alpha_065,   # Alpha#065
    "wq060": alpha_066,   # Alpha#066
    "wq061": alpha_068,   # Alpha#068
}
