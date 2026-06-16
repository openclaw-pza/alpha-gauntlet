#!/usr/bin/env python3
"""WQ101 batch0 smoke — three of the simplest alphas, hand-implemented to verify operator lib + IC eval + PIT all wire up.

Formulas (verbatim from the WorldQuant 101 Alphas formula list):
    Alpha#101: (close - open) / ((high - low) + 0.001)
    Alpha#033: rank(-1 + (open / close))            # cross-sectional rank(axis=1)
    Alpha#012: sign(delta(volume, 1)) * (-1 * delta(close, 1))

ALPHAS dict contract (identical to batchN.py):
    each value is fn(P) -> wide DataFrame(index=time, columns=symbols);
    P is the field-panel dict returned by panel_io.load_field_panel().
    Keys = id (smoke uses the wq900 range to avoid colliding with batch1-4's wq001+).

Note: smoke artifacts are stored as wq101_values_smoke.parquet and do **not** take part in the
gauntlet glob (the gauntlet only globs wq101_values_batch[1-9].parquet + existing).
"""
import numpy as np

from alphagauntlet.wq101 import ops


def alpha_101(P):
    """(close - open) / ((high - low) + 0.001)"""
    return (P["close"] - P["open"]) / ((P["high"] - P["low"]) + 0.001)


def alpha_033(P):
    """rank(-1 + (open / close))  — cross-sectional rank(axis=1)"""
    return ops.rank_cs(-1.0 + P["open"] / P["close"])


def alpha_012(P):
    """sign(delta(volume, 1)) * (-1 * delta(close, 1))"""
    return np.sign(ops.delta(P["volume"], 1)) * (-1.0 * ops.delta(P["close"], 1))


# Contract: keys are wq{id}; smoke occupies the wq901-903 range (no clash with batches' wq001..)
ALPHAS = {
    "wq901": alpha_101,   # Alpha#101
    "wq902": alpha_033,   # Alpha#033
    "wq903": alpha_012,   # Alpha#012
}
