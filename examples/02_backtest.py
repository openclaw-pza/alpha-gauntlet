#!/usr/bin/env python3
"""Example 02 -- backtest a factor set with the PIT-correct backtester.

Generates a small synthetic OHLCV panel, builds the factor panel, then runs
``alphagauntlet.factor_backtest`` to simulate the factor channel (rank
instruments each bar -> open the strongest -> manage with ATR stops / targets /
optional trailing) and prints expectancy / win rate / drawdown.

Requires the factor stack (numpy/pandas + TA-Lib + PyWavelets). If TA-Lib is not
installed the example prints an install hint and exits cleanly.

Run:
    python examples/02_backtest.py
"""
import os
import tempfile

_DATA = os.path.join(tempfile.gettempdir(), "alphagauntlet_example_data")
os.environ.setdefault("ALPHAGAUNTLET_DATA_DIR", _DATA)

from _sample_data import SAMPLE_POOL, make_panel   # noqa: E402


def main():
    print(f"generating synthetic panel in {_DATA} ...")
    make_panel(_DATA, pool=SAMPLE_POOL, bars=4000)

    try:
        from alphagauntlet import factor_backtest as bt
    except ImportError as e:
        print(f"\nfactor stack not importable ({e}).")
        print("Install the optional deps:  pip install TA-Lib PyWavelets")
        print("(TA-Lib needs its native C library first; see the README.)")
        return

    print("preparing factor panel + scores ...")
    panel = bt.prep()
    print(f"panel: {len(panel['idx'])} bars x {len(panel['cols'])} instruments\n")

    # A simple, illustrative configuration of the factor channel.
    cfg = {
        "_name": "demo (factor channel, ATR 1.5/2.0, no trailing)",
        "regime_gate": False, "per_coin_bear": False,
        "adx_min": None, "rsi_max": None,
        "sl_mult": 1.5, "tp_mult": 2.0,
        "trail_enabled": False, "tp_cap": True,
        "max_positions": 3, "stake_frac": 0.2,
    }
    stats = bt.simulate(panel, cfg)

    print("=== backtest result (synthetic data -- illustrative only) ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("\nThis is fabricated data with no real edge; numbers here mean nothing")
    print("financially. Swap in your own OHLCV (see README 'Data') for real research.")


if __name__ == "__main__":
    main()
