#!/usr/bin/env python3
"""Generate a small synthetic OHLCV panel for the examples.

No market data is shipped with alpha-gauntlet. These helpers fabricate a tiny,
plausible multi-instrument 1h OHLCV panel (geometric random walk with mild
cross-sectional momentum) and write one feather per symbol into a data dir,
matching the layout the factor/backtest modules expect:

    <DATA_DIR>/<SYMBOL_WITH_UNDERSCORE>-<TF>.feather
    columns: date, open, high, low, close, volume

This is illustrative data for demos and tests only -- it is not real and carries
no predictive content.
"""
import os

import numpy as np
import pandas as pd

SAMPLE_POOL = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]


def make_panel(data_dir, pool=None, bars=4000, tf="1h", seed=7):
    """Write a synthetic OHLCV feather per symbol. Returns the list of paths."""
    pool = pool or SAMPLE_POOL
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    idx = pd.date_range(start, periods=bars, freq="h")
    # a shared "market" factor + idiosyncratic drift gives mild cross-sectional structure
    market = np.cumsum(rng.normal(0, 0.004, bars))
    paths = []
    for k, sym in enumerate(pool):
        drift = rng.normal(0, 0.0002)
        idio = np.cumsum(rng.normal(drift, 0.012, bars))
        beta = 0.5 + 0.5 * rng.random()
        logp = np.log(100.0 * (k + 1)) + beta * market + idio
        close = np.exp(logp)
        # build OHLC around close with a small intrabar range
        rng_bar = np.abs(rng.normal(0, 0.006, bars)) * close
        high = close + rng_bar
        low = close - rng_bar
        open_ = np.concatenate([[close[0]], close[:-1]])
        vol = np.abs(rng.normal(1e6, 3e5, bars)) * (1 + 0.5 * np.abs(np.diff(logp, prepend=logp[0])))
        df = pd.DataFrame({"date": idx, "open": open_, "high": np.maximum(high, np.maximum(open_, close)),
                           "low": np.minimum(low, np.minimum(open_, close)), "close": close, "volume": vol})
        path = os.path.join(data_dir, sym.replace("/", "_") + f"-{tf}.feather")
        df.to_feather(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    out = os.environ.get("ALPHAGAUNTLET_DATA_DIR", "./data")
    p = make_panel(out)
    print(f"wrote {len(p)} synthetic feather files to {os.path.abspath(out)}")
