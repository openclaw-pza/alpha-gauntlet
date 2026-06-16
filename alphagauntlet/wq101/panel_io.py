#!/usr/bin/env python3
"""WQ101 panel loader — read raw OHLCV feather files and assemble the wide field panel the batchN.py operators expect.

Difference from factor_eval._load_panel:
- factor_eval returns a per-symbol DataFrame of *already-computed* factors; WQ101 alphas need the
  *raw* OHLCV plus derived fields vwap/returns/adv, in wide (time x symbol) cross-sectional format
  for the cross-sectional operators.
- This module only reads feather files; it imports no execution-side path and never touches any live system.

PIT discipline: vwap/returns/adv are all computed with shift/rolling/element-wise ops, no lookahead.
Alignment: symbols start at different times, so an outer-join union index is used; warmup/missing
cells stay NaN and are filtered cell by cell by the downstream IC valid mask (short symbols give
fewer valid cross-sections, without affecting long symbols).

Data directory:
- Reads ``<DATA_DIR>/<SYMBOL>-<tf>.feather`` where SYMBOL is the symbol with "/" replaced by "_".
- DATA_DIR is configurable: set the ALPHAGAUNTLET_DATA_DIR environment variable, otherwise it
  defaults to ``./data`` relative to the current working directory.
- Each feather file must contain a ``date`` column plus ``open/high/low/close/volume`` columns.
"""
import os

import pandas as pd

# Configurable data directory (env override; default ./data). No deployment path is hard-coded.
DATA_DIR = os.environ.get("ALPHAGAUNTLET_DATA_DIR", os.path.join(".", "data"))

# Default universe (20 symbols). Override by passing ``pool=`` to load_field_panel.
POOL = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "LINK/USDT", "AVAX/USDT", "DOT/USDT", "TRX/USDT", "LTC/USDT",
    "NEAR/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "INJ/USDT", "SUI/USDT", "FIL/USDT", "ATOM/USDT",
]

# Wide fields WQ101 alphas may reference (each value is a DataFrame[T, N_symbols]).
RAW_FIELDS = ("open", "high", "low", "close", "volume")


def _feather_path(sym, tf="1h"):
    return os.path.join(DATA_DIR, sym.replace("/", "_") + f"-{tf}.feather")


def load_field_panel(pool=None, tf="1h", n_tail=None, advs=(15, 20, 30, 40, 50, 60, 120, 180)):
    """Load the wide field panel for the batchN.py ALPHAS operator chains.

    Returns dict[str, pd.DataFrame]; each value is a wide frame (index=DatetimeIndex, columns=symbols):
        open / high / low / close / volume   raw OHLCV
        vwap     = (high + low + close) / 3   (standard approximation without tick weighting)
        returns  = close.pct_change()         (simple return, consistent with factor_eval)
        adv{N}   = volume.rolling(N).mean() * close   (dollar-volume mean)

    n_tail: keep the last n_tail+300 bars per symbol (300 warmup buffer, covering the longest
            adv180/corr windows). None = full history.
    """
    pool = pool or POOL
    raw = {f: {} for f in RAW_FIELDS}
    for sym in pool:
        fn = _feather_path(sym, tf)
        if not os.path.exists(fn):
            continue
        df = pd.read_feather(fn)
        if n_tail:
            df = df.tail(n_tail + 300).reset_index(drop=True)
        df = df.set_index("date")
        for f in RAW_FIELDS:
            raw[f][sym] = df[f].astype(float)

    if not raw["close"]:
        raise RuntimeError("load_field_panel: no symbol data was read, check DATA_DIR")

    # Assemble wide: union-index alignment (warmup/missing cells stay NaN, filtered downstream).
    fields = {}
    for f in RAW_FIELDS:
        wide = pd.DataFrame(raw[f]).sort_index()
        fields[f] = wide
    idx = fields["close"].index
    cols = fields["close"].columns

    # Derived fields (per column, PIT safe)
    fields["vwap"] = (fields["high"] + fields["low"] + fields["close"]) / 3.0
    fields["returns"] = fields["close"].pct_change()
    for n in advs:
        fields[f"adv{n}"] = fields["volume"].rolling(n, min_periods=n).mean() * fields["close"]

    # Unify index/columns (guard against derived-field column drift)
    for k in list(fields):
        fields[k] = fields[k].reindex(index=idx, columns=cols)
    return fields


def forward_return_panel(close_wide, h):
    """Identical to factor_eval._forward_return: close[t+h]/close[t] - 1. Last h rows NaN.

    close_wide is a wide close frame (index=time, columns=symbols). Returns a same-shape wide frame.
    """
    return close_wide.shift(-h) / close_wide - 1.0
