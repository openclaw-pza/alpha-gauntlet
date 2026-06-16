#!/usr/bin/env python3
"""WQ101 batch5 — identity-correction make-up batch (3 pre-registered formulas that were overwritten by key collisions and never evaluated).

Background (key collisions between batches):
- batch2.py registered Alpha#041/#042 under keys wq041/wq042; but batch3.py also pointed wq041/wq042
  at Alpha#043/#044 -> batch2's A041/A042 were overwritten by batch3's same-named keys when the gauntlet
  reads wq101_values_batch[1-4].parquet, and never went through the three rounds.
- batch3.py registered Alpha#068 under key wq061; but batch4.py also pointed wq061 at Alpha#071 ->
  batch3's A068 (original key wq061) was likewise overwritten and never evaluated.

This batch re-registers these 3 formulas under fresh keys not clashing with wq001-wq080, and runs them
through the exact same pre-registered three rounds (R1/R2/R3), at the same thresholds, with no leniency.

Zero-copy, zero-drift: directly imports the original function objects from batch2 / batch3 (both modules
define only module-level functions + an ALPHAS dict, no side effects, safe to import). The function
bodies are byte-for-byte the same as the main run.
- batch2.alpha_041  original: (high * low)^0.5 - vwap                  (Alpha#041)
- batch2.alpha_042  original: rank(vwap - close) / rank(vwap + close)  (Alpha#042)
- batch3.alpha_068  original: (ts_rank(corr(rank(high),rank(adv15),9),14)
                              < rank(delta(0.518371*close+0.481629*low,1))) * -1  (Alpha#068)

Contract same as batchN: each fn(P) -> same-shape wide DataFrame; no file reads / fillna / inf cleaning
inside fn (runner uniformly maps +/-inf->NaN).
"""
from alphagauntlet.wq101.batch2 import alpha_041, alpha_042
from alphagauntlet.wq101.batch3 import alpha_068

# --------------------------------------------------------------------------- #
# Keys: wq081/wq082/wq083 (fresh make-up keys, no clash with wq001-wq080), value -> original Alpha function.
# --------------------------------------------------------------------------- #
ALPHAS = {
    "wq081": alpha_041,   # Alpha#041 (make-up, original key wq041 overwritten by batch3's Alpha#043)
    "wq082": alpha_042,   # Alpha#042 (make-up, original key wq042 overwritten by batch3's Alpha#044)
    "wq083": alpha_068,   # Alpha#068 (make-up, original key wq061 overwritten by batch4's Alpha#071)
}
