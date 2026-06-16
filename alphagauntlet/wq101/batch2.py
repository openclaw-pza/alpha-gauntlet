#!/usr/bin/env python3
"""WQ101 batch2 — Alpha#022 .. Alpha#042 (21 factors, verbatim transcription).

Formula source: the WorldQuant 101 Alphas formula list, "### Alpha#0NN" sections (original formula wins).
Operators: always the vectorized operators in alphagauntlet.wq101.ops; no rolling.apply in the hot path.

Contract (identical to batch_api):
- Each fn(P) -> same-shape wide DataFrame; P is the field-panel dict from panel_io.load_field_panel().
- Keys wq022..wq042 (1:1 with Alpha#022..Alpha#042, globally unique, not in the 9xx smoke range).
- No file reads / no execution-side imports / no inf cleaning / no fillna of warmup NaN in fn (runner uniformly maps +/-inf->NaN).

Transcription discipline (hard rules):
- Windows/coefficients/thresholds are not changed. Fractional windows are rounded to integers and noted (none in this batch).
- When pseudo-code conflicts with the original formula, the original formula wins, noted in comments.
- Bare rank(x) = cross-sectional rank. Default transcribed to ops.rank_cs(x) (ordinal, 1..N).
  Exception list (faithful to Kakushadze [0,1]-rank, switched to ops.rank_cs_pct(x) normalized to [0,1]):
  1) formula contains a literal threshold comparison (e.g. `> 0.5`) -- needs median/percentile semantics (wq027 historical convention).
  2) rank embedded in arithmetic and mixed with non-rank terms / literal constants / sign terms -- ordinal
     rank [1,N] would change the cross-sectional ordering or drown small-scale terms; Kakushadze's original
     definition rank()∈[0,1] is correct. Known affected and already switched to pct:
     wq030 ((1.0-rank)*vol ratio), wq031 (three-term sum with sign(scale(corr))∈{-1,0,1}),
     wq039 ((1-rank(decay)) nested with (1+rank(mean))).
  Only when rank is at the outermost layer or combined linearly with same-origin ranks (overall same scale /
  N cancels in a ratio, e.g. wq034/wq036/wq042) is ordinal-vs-percentile monotonically equivalent, so ordinal is kept.
- adv{N}: always P["advN"] (= volume.rolling(N).mean()*close, dollar volume; WQ101 original adv = average
  dollar volume). Pseudo-code annotations of "volume mean, not dollar" are pseudo-code approximations that
  conflict with the original formula -> the original wins, use P["advN"] (dollar), deviation noted per fn.
- Conditional alphas (ternary) implemented with np.where / .where; degeneracy left to runner guards.
"""
import numpy as np

from alphagauntlet.wq101 import ops


# --------------------------------------------------------------------------- #
# Alpha#022
# original: -1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))
# --------------------------------------------------------------------------- #
def alpha_022(P):
    hv_corr = ops.corr(P["high"], P["volume"], 5)
    return -1.0 * ops.delta(hv_corr, 5) * ops.rank_cs(ops.ts_std(P["close"], 20))


# --------------------------------------------------------------------------- #
# Alpha#023
# original: (sma(high, 20) < high) ? (-1 * delta(high, 2)) : 0
# sma = simple moving average = ts_mean. Condition false (incl. warmup sma=NaN -> comparison False) -> 0.
# --------------------------------------------------------------------------- #
def alpha_023(P):
    cond = ops.ts_mean(P["high"], 20) < P["high"]
    val = -1.0 * ops.delta(P["high"], 2)
    return val.where(cond, 0.0)


# --------------------------------------------------------------------------- #
# Alpha#024
# original: (delta(sma(close,100),100)/delay(close,100) <= 0.05)
#          ? (-1*(close-ts_min(close,100))) : (-1*delta(close,3))
# --------------------------------------------------------------------------- #
def alpha_024(P):
    ma100 = ops.ts_mean(P["close"], 100)
    cond = (ops.delta(ma100, 100) / ops.delay(P["close"], 100)) <= 0.05
    branch_true = -1.0 * (P["close"] - ops.ts_min(P["close"], 100))
    branch_false = -1.0 * ops.delta(P["close"], 3)
    return branch_true.where(cond, branch_false)


# --------------------------------------------------------------------------- #
# Alpha#025
# original: rank((-1 * returns) * adv20 * vwap * (high - close))
# Deviation: adv20 uses P["adv20"] (dollar); the pseudo-code annotation "volume mean, not dollar" is an approximation -> original wins.
# --------------------------------------------------------------------------- #
def alpha_025(P):
    inner = (-1.0 * P["returns"]) * P["adv20"] * P["vwap"] * (P["high"] - P["close"])
    return ops.rank_cs(inner)


# --------------------------------------------------------------------------- #
# Alpha#026
# original: -1 * ts_max(correlation(ts_rank(volume,5), ts_rank(high,5), 5), 3)
# --------------------------------------------------------------------------- #
def alpha_026(P):
    v_tr = ops.ts_rank(P["volume"], 5)
    h_tr = ops.ts_rank(P["high"], 5)
    c = ops.corr(v_tr, h_tr, 5)
    return -1.0 * ops.ts_max(c, 3)


# --------------------------------------------------------------------------- #
# Alpha#027
# original: (rank(sma(correlation(rank(volume), rank(vwap), 6), 2) / 2.0) > 0.5) ? -1 : 1
# Outer rank compared to literal threshold 0.5 -> must use normalized rank_cs_pct([0,1]) for median-split meaning.
# Inner rank(volume)/rank(vwap) use ordinal rank_cs (feeding corr, monotonically equivalent).
# NaN policy: when rnk is NaN (warmup/short-history symbol with insufficient corr), faithful WQ101 ternary
#   semantics keep NaN (NaN>0.5 yields NaN in WQ101 operator algebra, not False). The pseudo-code's np.where
#   folding NaN into +1 is a pseudo-code approximation that collapses the whole cross-section to a constant
#   (measured 85% sections degenerate) -> per the original semantics, keep NaN (warmup pass-through, both
#   PIT-safe and non-degenerate).
# --------------------------------------------------------------------------- #
def alpha_027(P):
    corr6 = ops.corr(ops.rank_cs(P["volume"]), ops.rank_cs(P["vwap"]), 6)
    smooth = ops.ts_mean(corr6, 2) / 2.0
    rnk = ops.rank_cs_pct(smooth)               # deviation: literal >0.5 needs normalized rank, use rank_cs_pct
    arr = rnk.to_numpy()
    out = np.where(arr > 0.5, -1.0, 1.0)        # rnk>0.5 -> -1, else +1
    out = np.where(np.isnan(arr), np.nan, out)  # rnk=NaN -> NaN pass-through (faithful ternary semantics)
    return _as_df(out, rnk)


# --------------------------------------------------------------------------- #
# Alpha#028
# original: scale(correlation(adv20, low, 5) + ((high + low) / 2) - close)
# Deviation: adv20 uses P["adv20"] (dollar), same as A025.
# --------------------------------------------------------------------------- #
def alpha_028(P):
    corr_part = ops.corr(P["adv20"], P["low"], 5)
    raw = corr_part + (P["high"] + P["low"]) / 2.0 - P["close"]
    return ops.scale(raw)


# --------------------------------------------------------------------------- #
# Alpha#029
# original (Kakushadze, parsed bracket by bracket):
#   min(product(rank(rank(scale(log(sum(ts_min(rank(rank(-1*rank(delta(close-1,5)))),2),1))))),1),5)
#     + ts_rank(delay(-1*returns,6),5)
# corrections (per the Kakushadze original, two operators restored):
#   (a) inner ts_min(.,2) (2-period rolling min), previously mis-coded as ts_sum(.,2) (sum), no algebraic
#       equivalence -> restore ts_min.
#   (b) outer min(.,5) = element-wise min with scalar 5 (not ts_min); the same formula explicitly writes
#       both ts_min and min tokens, strongly implying different operators -> use np.minimum(.,5.0).
#   sum(.,1) and product(.,1) are window-1 identities, omitted.
# Note: delta(close-1,5) = delta(close,5) (the constant -1 cancels in the difference).
# All bare ranks are ordinal rank_cs (hard-rule default).
# --------------------------------------------------------------------------- #
def alpha_029(P):
    base = -1.0 * ops.rank_cs(ops.delta(P["close"], 5))   # delta(close-1,5)=delta(close,5)
    inner3 = ops.rank_cs(ops.rank_cs(base))               # rank(rank(-1*rank(delta)))
    tmin2 = ops.ts_min(inner3, 2)                         # original ts_min(.,2) (not ts_sum)
    # sum(.,1) and product(.,1) are window-1 identities, omitted
    log_v = np.log(tmin2)                                 # tmin2<=0 -> NaN/-inf, guarded by runner +/-inf->NaN
    scaled = ops.scale(log_v)
    rr = ops.rank_cs(ops.rank_cs(scaled))
    part1 = np.minimum(rr, 5.0)                           # original min(.,5) = element-wise min with scalar 5 (not ts_min)
    part2 = ops.ts_rank(ops.delay(-1.0 * P["returns"], 6), 5)
    return part1 + part2


# --------------------------------------------------------------------------- #
# Alpha#030
# original: ((1.0 - rank(sign(delta(close,1)) + sign(delay(delta(close,1),1))
#          + sign(delay(delta(close,1),2)))) * ts_sum(volume,5)) / ts_sum(volume,20)
# correction (faithful to Kakushadze): rank() embedded in arithmetic (1.0 - rank) mixed with a volume ratio;
#   ordinal rank [1,N] would change the cross-sectional ordering -> use normalized rank_cs_pct([0,1]).
# --------------------------------------------------------------------------- #
def alpha_030(P):
    d1 = ops.delta(P["close"], 1)
    sign_sum = np.sign(d1) + np.sign(ops.delay(d1, 1)) + np.sign(ops.delay(d1, 2))
    num = (1.0 - ops.rank_cs_pct(sign_sum)) * ops.ts_sum(P["volume"], 5)
    return num / ops.ts_sum(P["volume"], 20)


# --------------------------------------------------------------------------- #
# Alpha#031
# original: rank(rank(rank(decay_linear(-1*rank(rank(delta(close,10))),10))))
#          + rank(-1*delta(close,3)) + sign(scale(correlation(adv20,low,12)))
# Deviation: adv20 uses P["adv20"] (dollar).
# correction (faithful to Kakushadze): three-term sum including sign(scale(corr))∈{-1,0,1}; ordinal rank [1,N]
#   would drown the sign term -> all nested ranks switched to normalized rank_cs_pct([0,1]) to make the three terms comparable.
# --------------------------------------------------------------------------- #
def alpha_031(P):
    dl_in = -1.0 * ops.rank_cs_pct(ops.rank_cs_pct(ops.delta(P["close"], 10)))
    dl = ops.decay_linear(dl_in, 10)
    p1 = ops.rank_cs_pct(ops.rank_cs_pct(ops.rank_cs_pct(dl)))
    p2 = ops.rank_cs_pct(-1.0 * ops.delta(P["close"], 3))
    p3 = np.sign(ops.scale(ops.corr(P["adv20"], P["low"], 12)))
    return p1 + p2 + p3


# --------------------------------------------------------------------------- #
# Alpha#032
# original: scale(sma(close, 7) - close) + 20 * scale(correlation(vwap, delay(close, 5), 230))
# --------------------------------------------------------------------------- #
def alpha_032(P):
    p1 = ops.scale(ops.ts_mean(P["close"], 7) - P["close"])
    p2 = ops.scale(ops.corr(P["vwap"], ops.delay(P["close"], 5), 230))
    return p1 + 20.0 * p2


# --------------------------------------------------------------------------- #
# Alpha#033
# original: rank(-1 + (open / close))
# --------------------------------------------------------------------------- #
def alpha_033(P):
    return ops.rank_cs(-1.0 + P["open"] / P["close"])


# --------------------------------------------------------------------------- #
# Alpha#034
# original: rank(2 - rank(stddev(returns,2)/stddev(returns,5)) - rank(delta(close,1)))
# inner rank ordinal (hard-rule default).
# --------------------------------------------------------------------------- #
def alpha_034(P):
    vol_ratio = ops.ts_std(P["returns"], 2) / ops.ts_std(P["returns"], 5)
    inner = 2.0 - ops.rank_cs(vol_ratio) - ops.rank_cs(ops.delta(P["close"], 1))
    return ops.rank_cs(inner)


# --------------------------------------------------------------------------- #
# Alpha#035
# original: ts_rank(volume,32) * (1 - ts_rank(close+high-low,16)) * (1 - ts_rank(returns,32))
# --------------------------------------------------------------------------- #
def alpha_035(P):
    return (ops.ts_rank(P["volume"], 32)
            * (1.0 - ops.ts_rank(P["close"] + P["high"] - P["low"], 16))
            * (1.0 - ops.ts_rank(P["returns"], 32)))


# --------------------------------------------------------------------------- #
# Alpha#036
# original: 2.21*rank(corr(close-open, delay(volume,1), 15)) + 0.7*rank(open-close)
#          + 0.73*rank(ts_rank(delay(-returns,6),5)) + rank(abs(corr(vwap, adv20, 6)))
#          + 0.6*rank((sma(close,200)/200-open)*(close-open))
# Deviation: adv20 uses P["adv20"] (dollar).
# correction (per the Kakushadze original): the p5 last term `sum(close,200)/200` = 200-period mean =
#       ts_mean(close,200) (price-level magnitude). The formula-list text mis-wrote it as `sma(close,200)/200`
#       (= mean divided by 200 again), i.e. dividing the mean price by 200 twice, wrong magnitude/semantics.
#       Restore ts_mean(close,200), no second division.
# --------------------------------------------------------------------------- #
def alpha_036(P):
    p1 = 2.21 * ops.rank_cs(ops.corr(P["close"] - P["open"], ops.delay(P["volume"], 1), 15))
    p2 = 0.7 * ops.rank_cs(P["open"] - P["close"])
    p3 = 0.73 * ops.rank_cs(ops.ts_rank(ops.delay(-1.0 * P["returns"], 6), 5))
    p4 = ops.rank_cs(ops.corr(P["vwap"], P["adv20"], 6).abs())
    p5 = 0.6 * ops.rank_cs((ops.ts_mean(P["close"], 200) - P["open"])
                           * (P["close"] - P["open"]))
    return p1 + p2 + p3 + p4 + p5


# --------------------------------------------------------------------------- #
# Alpha#037
# original: rank(correlation(delay(open - close, 1), close, 200)) + rank(open - close)
# --------------------------------------------------------------------------- #
def alpha_037(P):
    delay_oc = ops.delay(P["open"] - P["close"], 1)
    return (ops.rank_cs(ops.corr(delay_oc, P["close"], 200))
            + ops.rank_cs(P["open"] - P["close"]))


# --------------------------------------------------------------------------- #
# Alpha#038
# original (Kakushadze arXiv:1601.00991 Appendix A): -1 * rank(ts_rank(close, 10)) * rank(close / open)
# correction: the inner time-series rank acts on close (authoritative original), not open. The formula-list text mis-wrote ts_rank(open,10).
# --------------------------------------------------------------------------- #
def alpha_038(P):
    return (-1.0 * ops.rank_cs(ops.ts_rank(P["close"], 10))
            * ops.rank_cs(P["close"] / P["open"]))


# --------------------------------------------------------------------------- #
# Alpha#039
# original: (-1 * rank(delta(close,7) * (1 - rank(decay_linear(volume/adv20, 9)))))
#          * (1 + rank(sma(returns, 250)))
# Deviation: adv20 uses P["adv20"] (dollar). volume/adv20 = volume / (vol_mean*close).
# correction (faithful to Kakushadze): rank embedded in arithmetic (1 - rank(decay)) and (1 + rank(mean));
#   ordinal rank [1,N] changes the cross-sectional ordering -> all nested ranks switched to normalized rank_cs_pct([0,1]).
# --------------------------------------------------------------------------- #
def alpha_039(P):
    rel_vol = ops.rank_cs_pct(ops.decay_linear(P["volume"] / P["adv20"], 9))
    p1 = -1.0 * ops.rank_cs_pct(ops.delta(P["close"], 7) * (1.0 - rel_vol))
    p2 = 1.0 + ops.rank_cs_pct(ops.ts_mean(P["returns"], 250))
    return p1 * p2


# --------------------------------------------------------------------------- #
# Alpha#040
# original: -1 * rank(stddev(high, 10)) * correlation(high, volume, 10)
# --------------------------------------------------------------------------- #
def alpha_040(P):
    return (-1.0 * ops.rank_cs(ops.ts_std(P["high"], 10))
            * ops.corr(P["high"], P["volume"], 10))


# --------------------------------------------------------------------------- #
# Alpha#041
# original: (high * low)^0.5 - vwap
# --------------------------------------------------------------------------- #
def alpha_041(P):
    return (P["high"] * P["low"]).pow(0.5) - P["vwap"]


# --------------------------------------------------------------------------- #
# Alpha#042
# original: rank(vwap - close) / rank(vwap + close)
# bare rank ordinal (hard-rule default); rank_cs takes 1..N, denominator always >=1, no division by zero.
# --------------------------------------------------------------------------- #
def alpha_042(P):
    return ops.rank_cs(P["vwap"] - P["close"]) / ops.rank_cs(P["vwap"] + P["close"])


# --------------------------------------------------------------------------- #
# Utility: ndarray -> same-shape wide DataFrame (used to restore conditional alphas after np.where)
# --------------------------------------------------------------------------- #
def _as_df(arr, like_df):
    import pandas as pd
    return pd.DataFrame(arr, index=like_df.index, columns=like_df.columns)


# DROPPED: Alpha#027 (wq027) — corr(rank_cs(vwap), rank_cs(volume), 6) degenerates on a 20-symbol pool.
#   Cross-sectional price-level ranks are highly persistent (BTC vwap is always rank 20, stablecoin ranks
#   barely move), so rank_cs(vwap) is approximately a time-constant per symbol -> the rolling Pearson corr
#   has zero variance within its window -> >85% of cells NaN, 0 valid evaluation sections. After faithful
#   NaN pass-through, 100% of sections are skipped by the degenerate guard (insufficient samples), cannot be
#   faithfully implemented on this cross-section -> removed from the registry.
#   The alpha_027 function body is kept for traceability and does NOT enter ALPHAS.
ALPHAS = {
    "wq022": alpha_022,
    "wq023": alpha_023,
    "wq024": alpha_024,
    "wq025": alpha_025,
    "wq026": alpha_026,
    "wq028": alpha_028,
    "wq029": alpha_029,
    "wq030": alpha_030,
    "wq031": alpha_031,
    "wq032": alpha_032,
    "wq033": alpha_033,
    "wq034": alpha_034,
    "wq035": alpha_035,
    "wq036": alpha_036,
    "wq037": alpha_037,
    "wq038": alpha_038,
    "wq039": alpha_039,
    "wq040": alpha_040,
    "wq041": alpha_041,
    "wq042": alpha_042,
}
