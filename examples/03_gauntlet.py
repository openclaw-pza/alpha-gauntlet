#!/usr/bin/env python3
"""Example 03 -- the anti-overfit ratchet evolution gauntlet.

This is the best starting point: it runs end-to-end with ZERO data files and
ZERO optional dependencies (only numpy, which the engine already requires).

We build a tiny *synthetic* backtest-eval callable. It does not run any real
backtester -- it fabricates trades whose quality depends on the candidate filter,
so we can watch the gauntlet:

  1. PROMOTE a genuine, robust improvement (better across every OOS segment), and
  2. REJECT a "false champion" that only looks good on the selection segments but
     collapses on the never-touched holdout segment.

Run:
    python examples/03_gauntlet.py
"""
import os
import shutil
import tempfile

import numpy as np

from alphagauntlet.evolution import GauntletConfig, RatchetEngine


def make_eval(holdout_collapse_keys=None):
    """Return a synthetic backtest-eval callable.

    The fabricated trade quality (mean profit_ratio) for the *challenger* is a
    function of which filter keys it carries. If ``holdout_collapse_keys`` is set,
    a challenger carrying those keys looks great on the selection window but its
    edge vanishes in the holdout window -- the classic false champion.
    """
    holdout_collapse_keys = set(holdout_collapse_keys or [])
    rng = np.random.default_rng(42)

    def _trades(role, chall_keys):
        # The holdout segment is 2026-01..2026-06 in the default config; the
        # selection segments are 2022..2025. A "false champion" carrying a
        # collapse key looks great in 2022-2025 but its edge vanishes in 2026.
        out = []
        for y in range(2022, 2026):                       # selection years
            for i in range(30):
                out.append(_mk(role, chall_keys, y, i, holdout=False))
        for i in range(30):                               # 2026 holdout months Jan-Jun
            out.append(_mk(role, chall_keys, 2026, i, holdout=True))
        return out

    def _mk(role, chall_keys, year, i, holdout):
        month = (1 + i % 6) if holdout else (1 + i % 12)  # holdout only spans Jan-Jun
        day = 1 + i % 27
        date = f"{year}-{month:02d}-{day:02d} 00:00:00"
        if role == "champion":
            mu, sd = 0.008, 0.05
        elif role == "baseline":
            mu, sd = 0.004, 0.05
        else:  # challenger: a tightening filter lifts mean return AND cuts variance
            mu, sd = 0.016, 0.03
            if holdout and (chall_keys & holdout_collapse_keys):
                mu, sd = 0.001, 0.06       # false champion: edge gone out of sample
        return {"profit_ratio": float(rng.normal(mu, sd)),
                "open_date": date, "close_date": date}

    def evaluate(strats, timerange):
        # The engine writes champion_filter.json / challenger_filter.json before
        # calling us; the synthetic eval reads the challenger filter and reacts.
        chall_keys = set(_read_json("challenger_filter") or {})
        result = {"strategy": {}}
        for strat in strats:
            role = ("champion" if strat.endswith("Champion")
                    else "baseline" if "Baseline" in strat else "challenger")
            trades = []
            for pair in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
                for t in _trades(role, chall_keys):
                    trades.append({**t, "pair": pair})
            result["strategy"][strat] = {"trades": trades}
        return result

    return evaluate


_STATE = {}


def _read_json(which):
    """Read the filter file the engine just wrote, so the synthetic eval can react."""
    import json
    path = _STATE.get(which)
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    workdir = tempfile.mkdtemp(prefix="alphagauntlet_demo_")
    state_dir = os.path.join(workdir, "state")
    _STATE["challenger_filter"] = os.path.join(state_dir, "challenger_filter.json")
    _STATE["champion_filter"] = os.path.join(state_dir, "champion_filter.json")

    cfg = GauntletConfig(n_total=10, n_seg=3)   # lower thresholds so the tiny synthetic set qualifies

    print("=" * 70)
    print("DEMO A: a genuine, robust improvement is PROMOTED")
    print("=" * 70)
    engine = RatchetEngine(make_eval(), state_dir=state_dir, config=cfg)
    print("init:", engine.init_genesis())
    res = engine.promote({"min_adx": 25}, evidence_summary="trend filter, robust across regimes")
    print("promote min_adx=25 ->", {k: res.get(k) for k in ("promoted", "reason_code", "epoch")})
    print("state:", {k: engine.get_state().get(k) for k in ("epoch", "champion")})

    print()
    print("=" * 70)
    print("DEMO B: a FALSE CHAMPION (great in-sample, collapses on holdout) is REJECTED")
    print("=" * 70)
    state_dir2 = os.path.join(workdir, "state2")
    _STATE["challenger_filter"] = os.path.join(state_dir2, "challenger_filter.json")
    _STATE["champion_filter"] = os.path.join(state_dir2, "champion_filter.json")
    # a challenger carrying min_vol_mult looks good on selection but dies on holdout
    engine2 = RatchetEngine(make_eval(holdout_collapse_keys={"min_vol_mult"}),
                            state_dir=state_dir2, config=cfg)
    print("init:", engine2.init_genesis())
    res2 = engine2.promote({"min_vol_mult": 2.0}, evidence_summary="looks amazing in-sample")
    print("promote min_vol_mult=2.0 ->",
          {k: res2.get(k) for k in ("promoted", "reason_code", "stage")})
    fb = res2.get("knowledge_feedback", {})
    print("rejection knowledge:", fb.get("direction"))

    print()
    print("Both outcomes are recorded in an append-only, hash-chained ledger at:")
    print("  ", os.path.join(state_dir, "evolution_ledger.jsonl"))
    print("Rejection is knowledge: the gauntlet remembers what it killed and why.")

    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
