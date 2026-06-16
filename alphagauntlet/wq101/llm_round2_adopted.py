#!/usr/bin/env python3
r"""LLM-generated round-2 (v2) adopted factors — hand-written implementations of the 4 gauntlet survivors.

Source (verbatim, no tuning / no window changes):
- A frozen round-2 expression list, content-addressed by sha256; the three-round elimination took 46 -> 4 survivors.
- Evaluator gold standard: these 4 functions were validated cell-for-cell against a generic DSL evaluator of
  the frozen expressions (0 absolute difference / 0 mask mismatch, outside warmup).

Contract (identical to batch7): fn(P) -> wide DataFrame(index=time, columns=symbols),
P=panel_io.load_field_panel(). Only ops/ops_ext operators; warmup NaN passes through, inf cleaned upstream.
Verbatim discipline: windows/coefficients unchanged.

Survivors (all same-signed across years, skip 0.185, incumbent |rho|<=0.16 near-orthogonal):
- r2v2_01_02 (t=+8.34): OHLC volatility estimator, intraday range vol - cross-bar body vol.
- r2v2_03_04 (t=+6.85): Roll implied spread, return-series covariance - vwap-series covariance.
- r2v2_05_03 (t=-5.50): downside co-skewness, return x squared market deviation.
- r2v2_06_01 (t=+5.94): Kyle's lambda price-impact elasticity (note: 2026 IC +0.018, recent decay, monitor live).
"""
import numpy as np

from alphagauntlet.wq101 import ops, ops_ext


def alpha_r2v2_01_02(P):
    # RANK_CS_PCT(-TS_MEAN(SIGNED_POWER(LOG($high/($low+1e-9)),2)-SIGNED_POWER(LOG($close/($open+1e-9)),2),24))
    hl = ops.signed_power(np.log(P["high"] / (P["low"] + 1e-9)), 2)
    co = ops.signed_power(np.log(P["close"] / (P["open"] + 1e-9)), 2)
    return ops.rank_cs_pct(-ops.ts_mean(hl - co, 24))


def alpha_r2v2_03_04(P):
    # RANK_CS_PCT(COV($returns,DELAY($returns,1),24)-COV(TS_PCTCHANGE($vwap,1),DELAY(TS_PCTCHANGE($vwap,1),1),24))
    r = P["returns"]
    vp = P["vwap"] / ops.delay(P["vwap"], 1) - 1.0      # TS_PCTCHANGE($vwap,1)
    a = ops.cov(r, ops.delay(r, 1), 24)
    b = ops.cov(vp, ops.delay(vp, 1), 24)
    return ops.rank_cs_pct(a - b)


def alpha_r2v2_05_03(P):
    # RANK_CS_PCT(TS_MEAN(($returns-TS_MEAN($returns,24))*SIGNED_POWER(MEDIAN_CS($returns)-TS_MEAN(MEDIAN_CS($returns),24),2),24))
    r = P["returns"]
    med = ops_ext.median_cs(r)
    term = (r - ops.ts_mean(r, 24)) * ops.signed_power(med - ops.ts_mean(med, 24), 2)
    return ops.rank_cs_pct(ops.ts_mean(term, 24))


def alpha_r2v2_06_01(P):
    # RANK_CS_PCT(-ABS(REGBETA($returns,SIGN($returns)*LOG($volume+1e-9),24)))
    x = np.sign(P["returns"]) * np.log(P["volume"] + 1e-9)
    beta = ops_ext.regbeta(P["returns"], x, 24)        # x is a wide frame -> per-window sliding regression
    return ops.rank_cs_pct(-beta.abs())


# Contract: ALPHAS = dict[str, callable], key = frozen-list name, value = fn(P)->wide.
ALPHAS = {
    "r2v2_01_02": alpha_r2v2_01_02,
    "r2v2_03_04": alpha_r2v2_03_04,
    "r2v2_05_03": alpha_r2v2_05_03,
    "r2v2_06_01": alpha_r2v2_06_01,
}
