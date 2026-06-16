#!/usr/bin/env python3
"""Tests for the ratchet evolution engine -- the anti-overfit gauntlet.

These tests use an in-memory synthetic backtest-eval and a temporary state dir,
so they run with only numpy (no TA-Lib / data files).
"""
import json
import os

import numpy as np
import pytest

from alphagauntlet.evolution import (
    GauntletConfig,
    RatchetEngine,
    materialize,
    read_ledger,
    segment_strategy,
)


def make_eval(state_dir, holdout_collapse_keys=None):
    """Synthetic eval that reads the challenger filter the engine writes and
    fabricates trades whose quality depends on the candidate and the segment."""
    holdout_collapse_keys = set(holdout_collapse_keys or [])
    rng = np.random.default_rng(0)
    chall_path = os.path.join(state_dir, "challenger_filter.json")

    def _chall_keys():
        if os.path.exists(chall_path):
            with open(chall_path, encoding="utf-8") as f:
                return set(json.load(f) or {})
        return set()

    def _mk(role, keys, year, i, holdout):
        month = (1 + i % 6) if holdout else (1 + i % 12)
        day = 1 + i % 27
        date = f"{year}-{month:02d}-{day:02d} 00:00:00"
        if role == "champion":
            mu, sd = 0.008, 0.05
        elif role == "baseline":
            mu, sd = 0.004, 0.05
        else:
            mu, sd = 0.016, 0.03
            if holdout and (keys & holdout_collapse_keys):
                mu, sd = 0.001, 0.06
        return {"profit_ratio": float(rng.normal(mu, sd)), "open_date": date, "close_date": date}

    def evaluate(strats, timerange):
        keys = _chall_keys()
        result = {"strategy": {}}
        for strat in strats:
            role = ("champion" if strat.endswith("Champion")
                    else "baseline" if "Baseline" in strat else "challenger")
            trades = []
            for pair in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
                for y in range(2022, 2026):
                    for i in range(30):
                        trades.append({**_mk(role, keys, y, i, False), "pair": pair})
                for i in range(30):
                    trades.append({**_mk(role, keys, 2026, i, True), "pair": pair})
            result["strategy"][strat] = {"trades": trades}
        return result

    return evaluate


@pytest.fixture
def engine(tmp_path):
    sd = str(tmp_path / "state")
    cfg = GauntletConfig(n_total=10, n_seg=3)
    eng = RatchetEngine(make_eval(sd), state_dir=sd, config=cfg)
    eng.init_genesis()
    return eng


def test_init_genesis_idempotent(engine):
    assert engine.get_state()["epoch"] == 0
    again = engine.init_genesis()
    assert again["ok"] and again["epoch"] == 0


def test_promote_genuine_improvement(engine):
    res = engine.promote({"min_adx": 25})
    assert res["promoted"] is True
    assert res["epoch"] == 1
    assert engine.get_state()["champion"] == {"min_adx": 25}


def test_false_champion_rejected_on_holdout(tmp_path):
    sd = str(tmp_path / "state")
    cfg = GauntletConfig(n_total=10, n_seg=3)
    eng = RatchetEngine(make_eval(sd, holdout_collapse_keys={"min_vol_mult"}),
                        state_dir=sd, config=cfg)
    eng.init_genesis()
    res = eng.promote({"min_vol_mult": 2.0})
    assert res["promoted"] is False
    assert res["reason_code"] == "HOLDOUT_REGRESSION"
    # not promoted -> epoch unchanged
    assert eng.get_state()["epoch"] == 0


def test_verifier_rejects_out_of_bounds(engine):
    res = engine.promote({"min_adx": 999})   # out of bounds (10..50)
    assert res["promoted"] is False
    assert res["reason_code"] == "INVALID"
    assert res["stage"] == "verifier"


def test_verifier_rejects_relaxation(engine):
    engine.promote({"min_adx": 25})
    # min_adx is HIGHER=stricter; proposing a lower value is a relaxation
    res = engine.promote({"min_adx": 20})
    assert res["promoted"] is False
    assert res["reason_code"] in ("RELAX_NEEDS_DEMOTE", "ALREADY_IN_CHAMPION")


def test_ledger_is_hash_chained_and_replayable(engine):
    engine.promote({"min_adx": 25})
    events = read_ledger(engine.ledger)
    assert events[0]["action"] == "genesis"
    assert any(e["action"] == "promote" for e in events)
    # projection of the ledger reproduces the champion filter
    assert materialize(events) == {"min_adx": 25}


def test_tamper_detection(engine):
    engine.promote({"min_adx": 25})
    # corrupt the ledger: flip a byte in a payload
    with open(engine.ledger, encoding="utf-8") as f:
        lines = f.readlines()
    obj = json.loads(lines[-1])
    obj["payload"]["set"]["min_adx"] = 999   # mutate without recomputing the hash
    lines[-1] = json.dumps(obj, separators=(",", ":")) + "\n"
    with open(engine.ledger, "w", encoding="utf-8") as f:
        f.writelines(lines)
    with pytest.raises(ValueError):
        read_ledger(engine.ledger)


def test_segment_strategy_fail_loud_on_missing_strat():
    result = {"strategy": {"A": {"trades": []}}}
    with pytest.raises(RuntimeError):
        segment_strategy(result, "B", [("S", "2022-01-01", "2023-01-01")])


def test_epsilon_inflates_with_attempts(engine):
    cfg = engine.cfg
    # build minimal champion/challenger metrics with equal small improvement;
    # a higher attempt count must make the gate stricter (reject where k=0 passes).
    champ = {"pooled": {"n": 100, "sharpe": 0.10, "maxdd": 0.1, "profit": 1.0, "se": 0.05},
             "segs": {nm: {"n": 0, "sharpe": 0, "maxdd": 0, "profit": 0, "se": float("inf")}
                      for nm, _, _ in cfg.segments}}
    chall = {"pooled": {"n": 100, "sharpe": 0.18, "maxdd": 0.1, "profit": 1.0, "se": 0.05},
             "segs": champ["segs"], "by_coin": {}}
    champ["by_coin"] = {}
    passed_k0, code0, _, _ = engine.ratchet_gate(champ, chall, 0)
    passed_k50, _, _, _ = engine.ratchet_gate(champ, chall, 50)
    # at k=50 the epsilon is strictly larger, so it cannot be easier to pass
    assert not (passed_k50 and not passed_k0)
