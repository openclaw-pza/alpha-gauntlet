#!/usr/bin/env python3
"""Example 01 -- evaluate factors (Rank IC / ICIR / t-stat).

Generates a small synthetic OHLCV panel, then uses ``alphagauntlet.factor_eval``
to score every factor's cross-sectional predictive power and print a ranked
table.

Requires the factor stack (numpy/pandas + TA-Lib + PyWavelets). If TA-Lib is not
installed the example prints an install hint and exits cleanly.

Run:
    python examples/01_evaluate_factors.py

A curated set of example factors you will see in the output:
  - momentum:        mom_24h / mom_72h / mom_168h   (price rate-of-change)
  - volatility:      vol_24h / vol_72h / atr_pct    (realised vol / ATR%)
  - mean-reversion:  rev_6h (short-horizon reversal), ou_rev_press (OU pressure)
  - WQ101 alphas:    wq_a005 / wq_a020 / wq_a024     (formulaic cross-sectional)
"""
import os
import tempfile

# Point the factor stack at a temporary synthetic data dir BEFORE importing it.
_DATA = os.path.join(tempfile.gettempdir(), "alphagauntlet_example_data")
os.environ.setdefault("ALPHAGAUNTLET_DATA_DIR", _DATA)

from _sample_data import SAMPLE_POOL, make_panel   # noqa: E402


def main():
    print(f"generating synthetic panel in {_DATA} ...")
    make_panel(_DATA, pool=SAMPLE_POOL, bars=4000)

    try:
        from alphagauntlet import factor_eval
    except ImportError as e:
        print(f"\nfactor stack not importable ({e}).")
        print("Install the optional deps:  pip install TA-Lib PyWavelets")
        print("(TA-Lib needs its native C library first; see the README.)")
        return

    report = factor_eval.score_all(tf="1h", save=False)
    if not report.get("ok"):
        print("evaluation failed:", report.get("reason"))
        return

    print(f"\nfactor evaluation: {report['n_factors']} factors x {report['n_coins']} instruments")
    print(report["honesty"])
    print(f"\n{'factor':<16}{'|IC|':>8}{'ICIR':>8}{'t-stat':>9}   verdict")
    print("-" * 64)
    for name in report["ranked_factors"][:20]:
        m = report["factors"][name]["by_horizon"]["h24"]
        if m.get("mean_ic") is None:
            print(f"{name:<16}{'(insufficient sample)':>30}")
            continue
        print(f"{name:<16}{m['abs_mean_ic']:>8}{m['icir']:>8}{m['t_stat']:>9}   {m['verdict']}")

    print("\nReminder: in-sample significance is not out-of-sample edge. Run survivors")
    print("through the gauntlet (examples/03_gauntlet.py) before trusting any of them.")


if __name__ == "__main__":
    main()
