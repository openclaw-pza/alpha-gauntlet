#!/usr/bin/env python3
"""Ratchet evolution harness -- a standalone anti-overfit "gauntlet" that enforces
monotonic, forward-only promotion of strategy/filter candidates.

The core idea
-------------
Naive optimisation loops are p-hacking machines: try enough candidate filters
against the same backtest and one will look great purely by chance (a
"false champion"). This harness makes the search ratchet -- a candidate may only
become the new champion if, evaluated against the *current* champion, it is
strictly better across multiple independent out-of-sample segments, does not
regress on any guardrail metric, survives an anchored walk-forward holdout, and
clears a significance threshold that *inflates* with the number of attempts in
the same family (Bonferroni-style multiple-testing control).

Five guarantees
---------------
- G1 Ratchet: ``challenger = merge(champion, candidate)`` is jointly re-evaluated
  against the *current* champion (not a frozen baseline). Only a strict win
  promotes.
- G2 Multi-segment out-of-sample: selection segments + a forward recheck segment
  + a never-touched holdout segment, mutually non-overlapping -- a true
  forward-OOS layout.
- G3 Conjunctive multi-metric gate: pooled Sharpe (primary) + per-segment
  max-drawdown / Sharpe non-degradation guardrails + minimum trade count +
  signal-retention floor.
- G4 Tamper-evident ledger (hash chain) + atomic ``os.replace`` writes +
  single-writer lock + write-then-replay self-consistency + auto-safe rollback.
- G5 Strict-greater-than gate whose epsilon inflates by ``sqrt(1 + ln(k+1))``
  with the per-family attempt count ``k`` (multiple-testing control) + structured
  rejection knowledge + neighbourhood de-duplication.

Decoupling
----------
This module is engine-only. It is parameterised by:

- A ``BacktestEval`` callable you provide. Given a list of strategy names and a
  timerange string it returns a structured result dict (see ``BacktestResult``
  in the type hints below). The engine never shells out to any specific
  backtester; you wire your own (a vectorised simulator, an event-driven
  backtester, or any callable that returns trades in the expected shape).
- A ``GauntletConfig`` describing the allowed filter keys, tightening direction,
  numeric bounds, segment layout and gate thresholds.

State (ledger, champion cache, lock) lives under a configurable ``state_dir``
(default ``./state``). There is no deployment, no live trading, no account
access -- this is a pure research-time promotion gate over historical backtests.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

GENESIS = "0" * 64


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class GauntletConfig:
    """Declarative description of the search space and gate thresholds.

    All fields have research-sane defaults; override per project. The defaults
    encode a tightening-only filter search (a candidate may only *narrow* an
    existing constraint -- relaxation must go through an explicit demote).
    """

    # --- search space ---
    allowed: frozenset = frozenset(
        {"min_vol_mult", "exclude_hours", "min_adx", "max_rsi_entry",
         "min_atr_pct", "max_atr_pct", "confirm_bars", "min_break_atr"})
    # tightening direction: HIGHER => larger is stricter, LOWER => smaller is stricter
    stricter_higher: frozenset = frozenset(
        {"min_vol_mult", "min_adx", "min_atr_pct", "min_break_atr", "confirm_bars"})
    stricter_lower: frozenset = frozenset({"max_rsi_entry", "max_atr_pct"})
    bounds: dict = field(default_factory=lambda: {
        "min_vol_mult": (1.0, 5.0), "min_adx": (10, 50), "min_atr_pct": (0.0, 10.0),
        "max_atr_pct": (1.0, 50.0), "max_rsi_entry": (50, 95),
        "min_break_atr": (0.0, 3.0), "confirm_bars": (1, 20)})
    grid: dict = field(default_factory=lambda: {
        "min_vol_mult": 0.1, "min_adx": 1, "min_atr_pct": 0.5, "max_atr_pct": 1,
        "max_rsi_entry": 1, "min_break_atr": 0.1, "confirm_bars": 1})
    key_limit: int = 4

    # --- segment layout (non-overlapping; aligned boundaries) ---
    # selection segments (used for promotion)
    segments: list = field(default_factory=lambda: [
        ("S1", "2022-01-01", "2023-01-01"),
        ("S2", "2023-01-01", "2024-01-01"),
        ("S3", "2024-01-01", "2025-01-01"),
        ("S4", "2025-01-01", "2025-09-01")])
    bt_start: str = "20211201"          # pre-roll for indicator warm-up; only in-segment entries count
    bt_end: str = "20260601"
    recheck_range: tuple = ("20250901", "20260101")   # forward-OOS, after all selection segments
    holdout_range: tuple = ("20260101", "20260601")   # never used for selection; burned once on promote

    # --- gate thresholds ---
    n_seg: int = 15                     # min trades per segment to evaluate
    n_total: int = 60                   # global min trades
    tol_dd: float = 0.10                # per-segment maxDD guardrail: challenger <= champion x 1.10
    tol_profit: float = 0.10            # pooled profit non-regression tolerance
    retention_min: float = 0.5          # signal retention floor: challenger/champion trade count >= 0.5
    eps_floor: float = 0.05             # minimum improvement floor (per-trade Sharpe units)
    z_base: float = 1.65                # one-sided z*
    abs_maxdd_default: float = 0.30     # conservative absolute pooled maxDD ceiling
    demote_k: int = 2                   # consecutive degrade rechecks before a real demote (hysteresis)
    stall_k: int = 6                    # consecutive rejects that trigger an exploration hint
    lock_stale: int = 2700              # seconds before a lock is considered stale and stolen

    # --- backtest strategy names (whatever your BacktestEval understands) ---
    champion_strat: str = "Champion"
    challenger_strat: str = "Challenger"
    baseline_strat: str = "Baseline"    # used in recheck vs a fixed reference


# Type aliases for the user-supplied evaluation callable.
#   BacktestEval(strats: list[str], timerange: str) -> BacktestResult
# where BacktestResult is a dict shaped like:
#   {"strategy": {strat_name: {"trades": [
#         {"profit_ratio": float, "open_date": iso, "close_date": iso, "pair": str}, ...]}}}
BacktestResult = dict
BacktestEval = Callable[[list, str], BacktestResult]


# ============================================================================
# Hash utilities
# ============================================================================
def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ============================================================================
# Single-writer lock
# ============================================================================
class EngineLock:
    """``os.O_EXCL`` file lock. ``promote`` blocks and retries; periodic recheck
    uses ``try_=True`` and skips the cycle if it cannot acquire. A stale lock
    (its ``ts`` older than ``lock_stale``) is stolen (low-frequency single host,
    so ``ts`` suffices; PID liveness is left as a future hardening)."""

    def __init__(self, lockfile: str, lock_stale: int, try_: bool = False):
        self.lockfile = lockfile
        self.lock_stale = lock_stale
        self.try_ = try_
        self.fd = None

    def __enter__(self):
        deadline = time.time() + 30
        delay = 0.05
        while True:
            try:
                self.fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self.fd, _canonical({"pid": os.getpid(), "ts": time.time()}).encode())
                return self
            except FileExistsError:
                if self._steal_if_stale():
                    continue
                if self.try_ or time.time() > deadline:
                    raise TimeoutError("engine.lock held (another evolution/recheck in progress)")
                time.sleep(delay)
                delay = min(delay * 1.7, 2.0)

    def _steal_if_stale(self) -> bool:
        try:
            with open(self.lockfile, encoding="utf-8") as f:
                ts = json.load(f).get("ts", 0)
            if time.time() - ts > self.lock_stale:
                os.remove(self.lockfile)
                return True
        except (OSError, json.JSONDecodeError):
            pass
        return False

    def __exit__(self, *a):
        if self.fd is not None:
            os.close(self.fd)
            try:
                os.remove(self.lockfile)
            except OSError:
                pass


# ============================================================================
# Metrics over a backtest result
# ============================================================================
def _parse_open_date(s) -> datetime.datetime:
    """Tolerant timestamp parse to naive UTC (aligned with naive segment bounds,
    used only for bucketing/ordering). Accepts space- or 'T'-separated, with or
    without 'Z'/microseconds/timezone."""
    txt = str(s).replace("Z", "+00:00")
    try:
        dt = datetime.datetime.fromisoformat(txt)
    except ValueError:
        dt = datetime.datetime.strptime(txt[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=None)


def _seg_metrics(trades: list) -> dict:
    """Per-trade Sharpe / maxDD / compounded profit / count from a list of trades
    (each carrying ``profit_ratio`` + ``close_date``). Per-trade (non-annualised)
    Sharpe: champion and challenger share the same convention, used only for
    relative comparison."""
    n = len(trades)
    if n == 0:
        return {"n": 0, "sharpe": 0.0, "maxdd": 0.0, "profit": 0.0, "se": float("inf")}
    r = np.array([float(t["profit_ratio"]) for t in trades], dtype=float)
    mean = float(r.mean())
    sd = float(r.std(ddof=1)) if n >= 2 else 0.0
    sharpe = mean / sd if sd > 1e-12 else 0.0
    ordered = sorted(trades, key=lambda t: _parse_open_date(t["close_date"]))
    eq, peak, maxdd = 1.0, 1.0, 0.0
    for t in ordered:
        eq *= (1.0 + float(t["profit_ratio"]))
        peak = max(peak, eq)
        maxdd = max(maxdd, (peak - eq) / peak)
    profit = float(np.prod(1.0 + r) - 1.0)
    se = math.sqrt((1.0 + 0.5 * sharpe ** 2) / n) if n >= 2 else float("inf")   # Lo (2002) Sharpe SE
    return {"n": n, "sharpe": round(sharpe, 4), "maxdd": round(maxdd, 4),
            "profit": round(profit, 4), "se": se}


def segment_strategy(result: BacktestResult, strat_name: str, seg_defs: list) -> dict:
    """Slice one strategy's trades into segments + pooled + per-instrument (by_coin).

    Returns ``{segs: {name: metrics}, pooled: metrics, by_coin: {pair: metrics}}``.
    ``seg_defs`` is a list of ``(name, start_iso, end_iso)``. Bucketing is by
    ``open_date``; a trade enters the first matching segment only.

    fail-loud: a missing strategy key raises (never silently return all-zeros,
    which would let a challenger be promoted against a "ghost" champion)."""
    strat_map = result.get("strategy", {})
    if strat_name not in strat_map:
        raise RuntimeError(f"backtest result missing strategy {strat_name} (stale/failed result?)")
    trades = strat_map[strat_name].get("trades", [])
    bounds = [(nm, _parse_open_date(s + " 00:00:00"), _parse_open_date(e + " 00:00:00"))
              for nm, s, e in seg_defs]
    buckets = {nm: [] for nm, _, _ in seg_defs}
    pooled = []
    by_coin = {}
    for t in trades:
        od = _parse_open_date(t["open_date"])
        for nm, s, e in bounds:
            if s <= od < e:
                buckets[nm].append(t)
                pooled.append(t)
                by_coin.setdefault(t.get("pair", "?"), []).append(t)
                break
    return {"segs": {nm: _seg_metrics(buckets[nm]) for nm, _, _ in seg_defs},
            "pooled": _seg_metrics(pooled),
            "by_coin": {c: _seg_metrics(ts) for c, ts in by_coin.items()}}


# ============================================================================
# Ledger (append-only hash chain) + champion projection -- pure functions
# ============================================================================
def read_ledger(ledger_path: str) -> list:
    """Read the ledger, verifying the hash chain row by row (prev link + recompute
    this_hash). A broken chain raises (tamper detection). Returns the event list."""
    if not os.path.exists(ledger_path):
        return []
    events, prev = [], GENESIS
    with open(ledger_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            ev = json.loads(ln)
            body = {k: ev[k] for k in ev if k != "this_hash"}
            if ev["prev_hash"] != prev:
                raise ValueError(f"ledger chain broken @seq={ev.get('seq')}: prev_hash mismatch")
            if _sha256(_canonical(body)) != ev["this_hash"]:
                raise ValueError(f"ledger chain broken @seq={ev.get('seq')}: this_hash recompute mismatch")
            prev = ev["this_hash"]
            events.append(ev)
    return events


def materialize(events: list) -> dict:
    """Deterministic projection of the ledger -> current champion filter dict.
    Key-level event stream: promote/move push values onto a per-key stack and
    pop on ``unset``; demote pops (reverting a key to its previous promoted
    value, or removing it). Pure function -- replayable to rebuild state."""
    stacks: dict = {}
    for ev in events:
        p = ev.get("payload", {})
        if ev["action"] in ("promote", "move"):
            for k, v in p.get("set", {}).items():
                stacks.setdefault(k, []).append(v)
            for k in p.get("unset", []):
                if stacks.get(k):
                    stacks[k].pop()
        elif ev["action"] == "demote":
            for k in p.get("unset", []):
                if stacks.get(k):
                    stacks[k].pop()
    return {k: st[-1] for k, st in stacks.items() if st}


def _derive_champion(events: list) -> tuple:
    """Derive champion's derivable parts (filter / epoch / ledger head) from the
    ledger -- the single source of truth. epoch = (#state-change events) - 1
    (genesis=0; each promote/demote/move increments it)."""
    filter_ = materialize(events)
    sc = [e for e in events if e["action"] in ("genesis", "promote", "demote", "move")]
    epoch = max(0, len(sc) - 1)
    last = sc[-1] if sc else None
    head = {"seq": last["seq"], "hash": last["this_hash"]} if last else {"seq": 0, "hash": GENESIS}
    return filter_, epoch, head


# ============================================================================
# Engine
# ============================================================================
class RatchetEngine:
    """Anti-overfit ratchet evolution engine.

    Parameters
    ----------
    backtest_eval:
        Callable ``(strats: list[str], timerange: str) -> BacktestResult``. The
        engine asks it to backtest the champion and challenger strategies over a
        timerange and reads structured trades back. You decide how strategies
        read their filter from ``champion_filter.json`` / ``challenger_filter.json``
        written under ``state_dir`` before each call.
    state_dir:
        Directory for the ledger, champion cache, filters and lock. Created on
        demand. Default ``./state``.
    config:
        A :class:`GauntletConfig`. Default is research-sane.
    """

    def __init__(self, backtest_eval: BacktestEval, state_dir: str = "./state",
                 config: Optional[GauntletConfig] = None):
        self.eval = backtest_eval
        self.cfg = config or GauntletConfig()
        self.state = os.path.abspath(state_dir)
        self.ledger = os.path.join(self.state, "evolution_ledger.jsonl")
        self.champion = os.path.join(self.state, "champion.json")
        self.champion_filter = os.path.join(self.state, "champion_filter.json")
        self.challenger_filter = os.path.join(self.state, "challenger_filter.json")
        self.lockfile = os.path.join(self.state, "engine.lock")

    # ---- lock helper ----
    def _lock(self, try_=False):
        return EngineLock(self.lockfile, self.cfg.lock_stale, try_=try_)

    # ---- ledger ----
    def _read_ledger(self):
        return read_ledger(self.ledger)

    def _append_event(self, action, payload, evidence_hash=""):
        """Append an event to the hash chain. Must be called while holding the lock."""
        events = self._read_ledger()
        seq = len(events) + 1
        prev = events[-1]["this_hash"] if events else GENESIS
        body = {"seq": seq, "ts": round(time.time(), 3), "action": action,
                "payload": payload, "evidence_hash": evidence_hash, "prev_hash": prev}
        body["this_hash"] = _sha256(_canonical(body))
        os.makedirs(self.state, exist_ok=True)
        with open(self.ledger, "a", encoding="utf-8") as f:
            f.write(_canonical(body) + "\n")
        return body

    def _family_attempts(self, events, candidate):
        """G5 multiple-testing attempt count k: per-key attribution, taking the
        max -- a candidate counts a key if any historical event in the same
        dimension touched that key. Prevents gaming via decorator keys that would
        otherwise reset a frozenset-of-all-keys counter to zero each time."""
        max_k = 0
        for key in set(candidate):
            k = sum(1 for ev in events if ev["action"] in ("reject", "promote", "probe")
                    and key in (ev.get("payload", {}).get("candidate", {}) or {}))
            max_k = max(max_k, k)
        return max_k

    def _fingerprint(self, cand):
        """Candidate grid fingerprint ``{key: round(val/step)}``. Returns None if a
        non-numeric key (e.g. ``exclude_hours``) is present (skips neighbourhood
        de-dup for that candidate)."""
        fp = {}
        for k, v in cand.items():
            if k not in self.cfg.grid:
                return None
            fp[k] = round(float(v) / self.cfg.grid[k])
        return fp

    def _reject_fingerprints(self, events):
        out = []
        for ev in events:
            if ev["action"] == "reject":
                fp = self._fingerprint(ev.get("payload", {}).get("candidate", {}))
                if fp is not None:
                    out.append(fp)
        return out

    # ---- champion state ----
    def _load_champion(self):
        """READ-ONLY load (does not write). The ledger is authoritative; champion's
        filter/epoch/ledger_head are always derived from it. ``champion.json`` only
        caches non-derivable bookkeeping (promoted_at / consecutive_degrade /
        metrics). Crash recovery: if a promote crashed after appending the event
        but before writing the cache, trust the ledger (derive). Only a cache
        ledger_head absent from the chain signals tampering."""
        events = self._read_ledger()
        filter_, epoch, head = _derive_champion(events)
        extras = {"promoted_at": None, "consecutive_degrade": 0, "metrics": {}}
        cache_stale = False
        if os.path.exists(self.champion):
            with open(self.champion, encoding="utf-8") as f:
                cached = json.load(f)
            head_hash = cached.get("ledger_head", {}).get("hash")
            chain_hashes = {e["this_hash"] for e in events} | {GENESIS}
            if head_hash not in chain_hashes:
                raise ValueError("champion.json ledger_head not in current chain (replaced/tampered?)")
            if cached.get("filter", {}) == filter_:
                extras = {k: cached.get(k, extras[k]) for k in extras}
            else:
                cache_stale = True
        champ = {"epoch": epoch, "filter": filter_, "ledger_head": head,
                 "_cache_stale": cache_stale, **extras}
        return champ, events

    def _write_champion(self, champ):
        """Atomic write of champion.json (os.replace) + synced champion_filter.json."""
        os.makedirs(self.state, exist_ok=True)
        persist = {k: v for k, v in champ.items() if not k.startswith("_")}
        tmp = self.champion + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(persist, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.champion)
        tmp2 = self.champion_filter + ".tmp"
        with open(tmp2, "w", encoding="utf-8") as f:
            json.dump(champ["filter"], f, ensure_ascii=False)
        os.replace(tmp2, self.champion_filter)

    def _heal_if_stale(self, champ):
        """Crash recovery: if the cache is stale (ledger ahead of champion.json),
        write the cache back to catch up and record a recover event. Must hold
        the lock. Returns whether recovery happened."""
        if not champ.get("_cache_stale"):
            return False
        self._write_champion(champ)
        self._append_event("recover", {"epoch": champ["epoch"], "filter": champ["filter"],
                                        "detail": "crash residue: ledger ahead of cache, rebuilt cache from ledger"})
        champ["_cache_stale"] = False
        return True

    # ---- action verifier (cheap, pre-backtest) ----
    def merge_filter(self, champion, candidate):
        """challenger = merge: same key takes the stricter value (tighten), new
        keys are added, ``exclude_hours`` takes the union."""
        out = dict(champion)
        for k, v in candidate.items():
            if k not in out:
                out[k] = v
            elif k == "exclude_hours":
                out[k] = sorted(set(out[k]) | set(v))
            elif k in self.cfg.stricter_higher:
                out[k] = max(out[k], v)
            elif k in self.cfg.stricter_lower:
                out[k] = min(out[k], v)
        return out

    def verify_candidate(self, candidate, champion, events):
        """ACTION_VERIFIER hard rules (no backtest, cheap). Returns (ok, code, detail)."""
        cfg = self.cfg
        if not isinstance(candidate, dict) or not candidate:
            return False, "INVALID", "candidate empty or not a dict"
        bad = set(candidate) - cfg.allowed
        if bad:
            return False, "INVALID", f"keys {list(bad)} not in allowed backtestable filter set"
        for k, v in candidate.items():
            if k == "exclude_hours":
                if not isinstance(v, (list, tuple)) or any(
                        isinstance(h, bool) or (not isinstance(h, int)) or h < 0 or h > 23 for h in v):
                    return False, "INVALID", "exclude_hours must be a list of integers 0..23"
                candidate[k] = sorted(set(int(h) for h in v))
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return False, "INVALID", f"{k}={v!r} must be numeric (int/float)"
            if k == "confirm_bars" and (int(v) != v):
                return False, "INVALID", "confirm_bars must be an integer"
            lo, hi = cfg.bounds[k]
            if not (lo <= float(v) <= hi):
                return False, "INVALID", f"{k}={v} out of bounds [{lo},{hi}]"
        # tightening direction (same key only accepts stricter; relaxing goes through demote)
        for k, v in candidate.items():
            if k not in champion:
                continue
            cv = champion[k]
            if k == "exclude_hours":
                if not set(v) > set(cv):
                    return False, "RELAX_NEEDS_DEMOTE", "exclude_hours must be a strict superset of champion"
            elif k in cfg.stricter_higher:
                if float(v) <= float(cv):
                    return False, ("ALREADY_IN_CHAMPION" if float(v) == float(cv) else "RELAX_NEEDS_DEMOTE"), \
                        f"{k} must be > champion {cv} to tighten"
            elif k in cfg.stricter_lower:
                if float(v) >= float(cv):
                    return False, ("ALREADY_IN_CHAMPION" if float(v) == float(cv) else "RELAX_NEEDS_DEMOTE"), \
                        f"{k} must be < champion {cv} to tighten"
        challenger = self.merge_filter(champion, candidate)
        if "min_atr_pct" in challenger and "max_atr_pct" in challenger \
                and float(challenger["min_atr_pct"]) >= float(challenger["max_atr_pct"]):
            return False, "INVALID", "min_atr_pct >= max_atr_pct -> empty interval"
        if challenger == champion:
            return False, "ALREADY_IN_CHAMPION", "merge equals champion (no change)"
        if len(challenger) > cfg.key_limit:
            return False, "KEY_LIMIT", f"champion has {len(champion)} keys; exceeds limit {cfg.key_limit}"
        fp = self._fingerprint(candidate)
        if fp is not None:
            for rfp in self._reject_fingerprints(events):
                if set(rfp) == set(fp) and all(abs(rfp[k] - fp[k]) <= 1 for k in fp):
                    return False, "EXPLORED_NEIGHBORHOOD", "candidate falls in an already-rejected neighbourhood"
        return True, "OK", "passed pre-checks"

    # ---- ratchet gate (G3 + G5) ----
    def ratchet_gate(self, champ_m, chall_m, k_attempts, abs_dd=None):
        """Judge whether the challenger is strictly positive vs the champion.
        Returns (passed, reason_code, failing_dim, detail)."""
        cfg = self.cfg
        abs_dd = cfg.abs_maxdd_default if abs_dd is None else abs_dd
        cp, hp = champ_m["pooled"], chall_m["pooled"]
        if hp["n"] < cfg.n_total:
            return False, "INSUFFICIENT_TRADES", "pooled", f"challenger total trades {hp['n']}<{cfg.n_total}"
        for nm, _, _ in cfg.segments:
            if chall_m["segs"][nm]["n"] == 0 and champ_m["segs"][nm]["n"] >= cfg.n_seg:
                return False, "SIGNAL_RETENTION_LOW", nm, f"{nm} segment filtered to zero (too strict)"
        if cp["n"] > 0 and hp["n"] / cp["n"] < cfg.retention_min:
            return False, "SIGNAL_RETENTION_LOW", "pooled", \
                f"retention {hp['n']}/{cp['n']}={hp['n']/cp['n']:.2f}<{cfg.retention_min}"
        if hp["profit"] <= 0:
            return False, "DOMINATED_BY_CHAMPION", "pooled", f"challenger pooled profit {hp['profit']}<=0"
        if cp["profit"] > 0 and hp["profit"] < cp["profit"] * (1 - cfg.tol_profit):
            return False, "POOLED_PROFIT_REGRESSION", "pooled", \
                f"challenger pooled profit {hp['profit']} < champion {cp['profit']}x{1-cfg.tol_profit}"
        if hp["maxdd"] > cp["maxdd"] * (1 + cfg.tol_dd) + 1e-9:
            return False, "POOLED_DD_REGRESSION", "pooled", \
                f"challenger pooled maxDD {hp['maxdd']} > champion {cp['maxdd']}x{1+cfg.tol_dd}"
        if hp["maxdd"] > abs_dd:
            return False, "ABS_DD_BREACH", "pooled", \
                f"challenger pooled maxDD {hp['maxdd']} > absolute ceiling {abs_dd}"
        for nm, _, _ in cfg.segments:
            cs, hs = champ_m["segs"][nm], chall_m["segs"][nm]
            if cs["n"] < cfg.n_seg:
                continue
            if hs["n"] < cfg.n_seg:
                # champion had enough samples but challenger starved this segment -> reject.
                # Prevents gaming: starving a "bad regime" segment below n_seg to skip its degrade check.
                return False, "SEG_SAMPLE_COLLAPSE", nm, \
                    f"{nm}: champion {cs['n']} trades but challenger only {hs['n']}<{cfg.n_seg}"
            tol = 1.0 / math.sqrt(hs["n"])
            if hs["sharpe"] < cs["sharpe"] - tol:
                return False, "SEG_REGRESSION", nm, f"{nm} Sharpe {hs['sharpe']}<{cs['sharpe']}-{tol:.3f}"
            if hs["maxdd"] > cs["maxdd"] * (1 + cfg.tol_dd) + 1e-9:
                return False, "SEG_REGRESSION", nm, f"{nm} maxDD {hs['maxdd']}>{cs['maxdd']}x{1+cfg.tol_dd}"
        # G5: pooled Sharpe strict improvement; epsilon inflates with family attempt count
        se_diff = math.sqrt(cp["se"] ** 2 + hp["se"] ** 2) \
            if math.isfinite(cp["se"]) and math.isfinite(hp["se"]) else 1.0
        eps = max(cfg.eps_floor, cfg.z_base * se_diff) * math.sqrt(1.0 + math.log(k_attempts + 1))
        improve = hp["sharpe"] - cp["sharpe"]
        if improve < eps:
            return False, "DOMINATED_BY_CHAMPION", "pooled", \
                f"pooled Sharpe improvement {improve:.4f}<eps {eps:.4f} (k={k_attempts})"
        # multi-instrument consistency guardrail: challenger must not win on a few coins only
        cc = champ_m.get("by_coin", {}); hc = chall_m.get("by_coin", {})
        coins = [c for c in cc if cc[c]["n"] >= cfg.n_seg and c in hc]
        if coins:
            worse = sum(1 for c in coins
                        if hc[c]["sharpe"] < cc[c]["sharpe"] - 1.0 / math.sqrt(max(hc[c]["n"], 1)))
            if worse / len(coins) > 0.5:
                return False, "MULTI_COIN_DIVERGENCE", "by_coin", \
                    f"challenger degrades on {worse}/{len(coins)} instruments (>50%): not pool-wide"
        return True, "PASS", "", f"pooled Sharpe +{improve:.4f}>=eps {eps:.4f}, segments non-degrading"

    # ---- holdout confirm gate ----
    def _holdout_confirm(self, champ_ho, chall_ho):
        """True forward-OOS holdout confirm (after the ratchet gate passes): the
        challenger must not degrade on the never-touched holdout segment.
        Returns (ok, detail). Insufficient sample -> (True, warning) honest pass."""
        cp, hp = champ_ho["pooled"], chall_ho["pooled"]
        if hp["n"] < self.cfg.n_seg:
            return True, f"holdout sample insufficient ({hp['n']}<{self.cfg.n_seg}), passing with warning"
        if hp["profit"] <= 0:
            return False, f"holdout pooled profit {hp['profit']}<=0"
        tol = 1.0 / math.sqrt(max(hp["n"], 1))
        if hp["sharpe"] < cp["sharpe"] - tol:
            return False, f"holdout Sharpe {hp['sharpe']}<{cp['sharpe']}-{tol:.3f} (degrades on true future)"
        return True, f"holdout non-degrading (Sharpe {hp['sharpe']}>={cp['sharpe']}-{tol:.3f})"

    # ---- main flow: promote ----
    def promote(self, candidate, evidence_summary=""):
        """Ratchet promotion loop. Holds the lock for the whole flow. Returns a
        structured result (including rejection-knowledge feedback)."""
        cfg = self.cfg
        try:
            with self._lock():
                champ, events = self._load_champion()
                if self._heal_if_stale(champ):
                    champ, events = self._load_champion()
                champion = champ["filter"]
                ok, code, detail = self.verify_candidate(candidate, champion, events)
                if not ok:
                    ev = self._append_event("reject", {"candidate": candidate, "reason_code": code,
                                                        "stage": "verifier", "detail": detail})
                    return self._reject_result(code, "verifier", detail, candidate, champion, events, ev)
                challenger = self.merge_filter(champion, candidate)
                os.makedirs(self.state, exist_ok=True)
                with open(self.challenger_filter, "w", encoding="utf-8") as f:
                    json.dump(challenger, f, ensure_ascii=False)
                with open(self.champion_filter, "w", encoding="utf-8") as f:
                    json.dump(champion, f, ensure_ascii=False)
                strats = [cfg.champion_strat, cfg.challenger_strat]
                result = self.eval(strats, f"{cfg.bt_start}-{cfg.bt_end}")
                self._assert_strats(result, strats)
                champ_m = segment_strategy(result, cfg.champion_strat, cfg.segments)
                chall_m = segment_strategy(result, cfg.challenger_strat, cfg.segments)
                k = self._family_attempts(events, candidate)
                passed, code, fdim, detail = self.ratchet_gate(champ_m, chall_m, k)
                evidence = {"candidate": candidate, "challenger": challenger,
                            "champion_pooled": champ_m["pooled"], "challenger_pooled": chall_m["pooled"],
                            "champion_segs": champ_m["segs"], "challenger_segs": chall_m["segs"],
                            "k_attempts": k}
                ehash = _sha256(_canonical(evidence))
                if not passed:
                    ev = self._append_event("reject", {**evidence, "reason_code": code, "stage": "ratchet",
                                                        "failing_dimension": fdim, "detail": detail}, ehash)
                    return self._reject_result(code, "ratchet", detail, candidate, champion, events, ev, fdim)
                # CAS: re-read epoch to defeat TOCTOU
                champ2, events2 = self._load_champion()
                if champ2["epoch"] != champ["epoch"]:
                    ev = self._append_event("reject", {"candidate": candidate, "reason_code": "STALE_BASELINE",
                                                        "detail": "champion updated concurrently during backtest"}, ehash)
                    return self._reject_result("STALE_BASELINE", "cas", "baseline updated; re-submit",
                                               candidate, champion, events2, ev)
                # true OOS holdout burn-in confirm (segment never used for selection)
                result_ho = self.eval(strats, f"{cfg.bt_start}-{cfg.holdout_range[1]}")
                self._assert_strats(result_ho, strats)
                hr = cfg.holdout_range
                _ho_def = [("HOLDOUT", f"{hr[0][:4]}-{hr[0][4:6]}-{hr[0][6:]}",
                            f"{hr[1][:4]}-{hr[1][4:6]}-{hr[1][6:]}")]
                champ_ho = segment_strategy(result_ho, cfg.champion_strat, _ho_def)
                chall_ho = segment_strategy(result_ho, cfg.challenger_strat, _ho_def)
                ho_ok, ho_detail = self._holdout_confirm(champ_ho, chall_ho)
                if not ho_ok:
                    ev = self._append_event("reject", {**evidence, "reason_code": "HOLDOUT_REGRESSION",
                                                        "stage": "holdout", "detail": ho_detail}, ehash)
                    return self._reject_result("HOLDOUT_REGRESSION", "holdout", ho_detail,
                                               candidate, champion, events2, ev)
                changed = {k2: challenger[k2] for k2 in challenger
                           if k2 not in champion or champion[k2] != challenger[k2]}
                ev = self._append_event("promote", {"candidate": candidate, "set": changed,
                                                     "champion_segs": chall_m["segs"],
                                                     "evidence_summary": evidence_summary or "(none)"}, ehash)
                new_champ = {"epoch": champ["epoch"] + 1, "filter": challenger,
                             "ledger_head": {"seq": ev["seq"], "hash": ev["this_hash"]},
                             "promoted_at": datetime.date.today().isoformat(),
                             "consecutive_degrade": 0,
                             "metrics": {"pooled_sharpe": chall_m["pooled"]["sharpe"],
                                         "prev_sharpe": champ_m["pooled"]["sharpe"],
                                         "pooled_trades": chall_m["pooled"]["n"]}}
                replayed = materialize(self._read_ledger())
                if replayed != challenger:   # if/raise (not assert): python -O strips assert
                    raise RuntimeError(f"write-then-replay mismatch: {replayed} != challenger {challenger}")
                self._write_champion(new_champ)
                return {"ok": True, "promoted": True, "epoch": new_champ["epoch"],
                        "champion": challenger, "improvement": detail,
                        "evidence": {"champion_pooled": champ_m["pooled"],
                                     "challenger_pooled": chall_m["pooled"]},
                        "seq": ev["seq"],
                        "note": "strictly beat current champion across all segments incl. holdout"}
        except (TimeoutError, ValueError, RuntimeError, TypeError) as e:
            return {"ok": False, "promoted": False, "reason_code": "ENGINE_ERROR",
                    "detail": f"{type(e).__name__}: {e}"}

    def _assert_strats(self, result, strats):
        """Defence-in-depth: assert both strategies produced trades; fail-loud on a
        missing key (never judge on a partial result -> ghost champion)."""
        have = set(result.get("strategy", {}))
        missing = [s for s in strats if s not in have]
        if missing:
            raise RuntimeError(f"backtest result missing strategies {missing} (got {sorted(have)})")

    def _reject_result(self, code, stage, detail, candidate, champion, events, ev, fdim=""):
        """Build a rejection return + structured pitfall-avoidance knowledge
        (direction only, no exact thresholds -- prevents learning the gate shape)."""
        weakest = self._weakest_segment(events)
        return {"ok": True, "promoted": False, "reason_code": code, "stage": stage,
                "detail": detail, "failing_dimension": fdim, "seq": ev["seq"],
                "knowledge_feedback": self._knowledge_feedback(code, weakest),
                "note": "not promoted. rejection recorded in the hash-chain ledger (rejection is knowledge)."}

    def _knowledge_feedback(self, code, weakest):
        tips = {
            "DOMINATED_BY_CHAMPION": "improvement not significant: try a fresh dimension or fix the weakest segment.",
            "SEG_REGRESSION": "degrades in some regime: your candidate hurt returns there; avoid that direction.",
            "POOLED_PROFIT_REGRESSION": "total profit regressed: Sharpe up but pooled profit down (cutting exposure?).",
            "POOLED_DD_REGRESSION": "total drawdown worsened: pooled equity max drawdown exceeds tolerance.",
            "ABS_DD_BREACH": "broke the absolute drawdown ceiling regardless of relative improvement.",
            "MULTI_COIN_DIVERGENCE": "does not generalise: wins on a few instruments only, like single-name overfit.",
            "HOLDOUT_REGRESSION": "true-future segment degraded: classic overfit. switch dimension.",
            "SEG_SAMPLE_COLLAPSE": "you starved a regime below the minimum sample; loosen so it keeps enough trades.",
            "SIGNAL_RETENTION_LOW": "too strict, trades collapsed; loosen (but never relax a promoted constraint).",
            "INSUFFICIENT_TRADES": "sample too small to falsify; loosen until it produces enough trades.",
            "RELAX_NEEDS_DEMOTE": "you are relaxing a promoted constraint; forward evolution only accepts tightening.",
            "ALREADY_IN_CHAMPION": "this constraint is already in the champion (or stricter); pick another dimension.",
            "EXPLORED_NEIGHBORHOOD": "this neighbourhood was already rejected; move away or switch dimension.",
            "STALE_BASELINE": "baseline updated during your backtest; re-submit against the new champion.",
            "KEY_LIMIT": "champion filter at key limit; new keys need a replacement move (exploration mode).",
        }
        return {"direction": tips.get(code, "switch dimension or direction."),
                "champion_weakest_regime": weakest,
                "reminder": "direction only, no exact thresholds (prevents overfitting the gate)."}

    def _weakest_segment(self, events):
        """Champion's weakest segment hint, taken from the latest promote event's
        champion_segs (lowest-Sharpe segment with enough samples). None if absent."""
        last = next((e for e in reversed(events) if e["action"] == "promote"), None)
        segs = (last or {}).get("payload", {}).get("champion_segs") if last else None
        if not segs:
            return None
        valid = {nm: s for nm, s in segs.items() if isinstance(s, dict) and s.get("n", 0) >= self.cfg.n_seg}
        return min(valid, key=lambda nm: valid[nm]["sharpe"]) if valid else None

    # ---- recheck + auto-safe rollback ----
    def recheck(self):
        """Periodic recheck (auto-safe). The champion must still beat a fixed
        baseline strategy on the forward-OOS recheck segment; otherwise after
        ``demote_k`` consecutive degrading cycles, LIFO-demote the most recent
        promoted keys (roll back toward the safer state). try-lock: skip the
        cycle if it cannot acquire."""
        cfg = self.cfg
        try:
            with self._lock(try_=True):
                champ, events = self._load_champion()
                if self._heal_if_stale(champ):
                    champ, events = self._load_champion()
                if not champ["filter"]:
                    return {"ok": True, "action": "noop", "detail": "empty champion (pure baseline), nothing to recheck"}
                with open(self.champion_filter, "w", encoding="utf-8") as f:
                    json.dump(champ["filter"], f, ensure_ascii=False)
                strats = [cfg.champion_strat, cfg.baseline_strat]
                result = self.eval(strats, f"{cfg.recheck_range[0]}-{cfg.recheck_range[1]}")
                self._assert_strats(result, strats)
                rr = cfg.recheck_range
                _rc_def = [("RECHECK", f"{rr[0][:4]}-{rr[0][4:6]}-{rr[0][6:]}",
                            f"{rr[1][:4]}-{rr[1][4:6]}-{rr[1][6:]}")]
                ch = segment_strategy(result, cfg.champion_strat, _rc_def)["pooled"]
                base = segment_strategy(result, cfg.baseline_strat, _rc_def)["pooled"]
                tol = 1.0 / math.sqrt(max(ch["n"], 1))
                degraded = ch["n"] >= cfg.n_seg and ch["sharpe"] < base["sharpe"] - tol
                ehash = _sha256(_canonical({"champion": ch, "baseline": base}))
                if not degraded:
                    if champ.get("consecutive_degrade", 0):
                        champ["consecutive_degrade"] = 0
                        self._write_champion(champ)
                    self._append_event("recheck", {"degraded": False, "champion_sharpe": ch["sharpe"],
                                                    "baseline_sharpe": base["sharpe"]}, ehash)
                    return {"ok": True, "action": "hold", "champion_sharpe": ch["sharpe"],
                            "baseline_sharpe": base["sharpe"], "detail": "champion still beats baseline"}
                cnt = champ.get("consecutive_degrade", 0) + 1
                if cnt < cfg.demote_k:
                    champ["consecutive_degrade"] = cnt
                    self._write_champion(champ)
                    self._append_event("recheck", {"degraded": True, "consecutive": cnt,
                                                    "champion_sharpe": ch["sharpe"], "baseline_sharpe": base["sharpe"]}, ehash)
                    return {"ok": True, "action": "warning", "consecutive": cnt,
                            "detail": f"recheck degraded {cnt}/{cfg.demote_k} (hysteresis, below demote threshold)"}
                last_promote = next((e for e in reversed(events) if e["action"] == "promote"), None)
                if not last_promote:
                    return {"ok": True, "action": "noop", "detail": "no promote event to roll back"}
                unset_keys = list(last_promote["payload"].get("set", {}))
                ev = self._append_event("demote", {"unset": unset_keys, "reason": "recheck_degraded_auto_safe",
                                                    "champion_sharpe": ch["sharpe"], "baseline_sharpe": base["sharpe"]}, ehash)
                new_filter = materialize(self._read_ledger())
                new_champ = {"epoch": champ["epoch"] + 1, "filter": new_filter,
                             "ledger_head": {"seq": ev["seq"], "hash": ev["this_hash"]},
                             "promoted_at": champ.get("promoted_at"), "consecutive_degrade": 0,
                             "metrics": {"rolled_back_keys": unset_keys}}
                self._write_champion(new_champ)
                return {"ok": True, "action": "demote", "demoted_keys": unset_keys, "new_filter": new_filter,
                        "alert": f"auto_safe rollback: champion degraded on recheck, demoted {unset_keys}",
                        "detail": "rolled back toward the safer state"}
        except TimeoutError:
            return {"ok": False, "action": "skipped", "detail": "lock held, retry next cycle"}
        except (ValueError, RuntimeError, TypeError) as e:
            return {"ok": False, "action": "error", "detail": f"{type(e).__name__}: {e}"}

    # ---- read-only state ----
    def get_state(self):
        """Inspect current champion + epoch + recent ledger events (incl. rejection
        knowledge) before proposing, to avoid re-submitting dead candidates."""
        try:
            champ, events = self._load_champion()
            recent = [{"seq": e["seq"], "action": e["action"],
                       "candidate": e.get("payload", {}).get("candidate"),
                       "reason_code": e.get("payload", {}).get("reason_code")}
                      for e in events[-12:]]
            streak = 0
            for e in reversed(events):
                if e["action"] == "reject":
                    streak += 1
                elif e["action"] in ("promote", "demote", "move", "genesis"):
                    break
            out = {"ok": True, "epoch": champ["epoch"], "champion": champ["filter"],
                   "promoted_at": champ.get("promoted_at"),
                   "ledger_events": len(events), "consecutive_rejects": streak, "recent": recent,
                   "rejected_recently": [r for r in recent if r["action"] == "reject"]}
            if streak >= self.cfg.stall_k:
                out["exploration_hint"] = (
                    f"{streak} consecutive rejects (>={self.cfg.stall_k}). Stop fishing the same dimension -- "
                    "switch to a fresh dimension or fix the champion's weakest segment. A strict gate "
                    "rejecting often is guarding against false champions, not a fault.")
            return out
        except (ValueError, RuntimeError) as e:
            return {"ok": False, "detail": str(e)}

    def init_genesis(self):
        """Initialise: empty champion + genesis ledger (idempotent)."""
        os.makedirs(self.state, exist_ok=True)
        with self._lock():
            if not os.path.exists(self.ledger):
                self._append_event("genesis", {"note": "ratchet evolution engine initialised, champion=empty"})
            champ, events = self._load_champion()
            self._write_champion(champ)
            return {"ok": True, "epoch": champ["epoch"], "ledger_events": len(events)}


# ============================================================================
# NOTE on intentionally-dropped deployment glue
# ----------------------------------------------------------------------------
# The internal version of this engine also materialised a champion projection
# into a live skill document and pushed it to a remote runner with a two-phase
# commit + hash reconcile. That deployment layer is intentionally NOT published:
# it was tightly coupled to one private deployment topology and carries no
# research value. The published engine ends at "champion state is updated in the
# ledger + cache"; wiring the promoted filter into your own live system is left
# to you, downstream of this gate.
# ============================================================================
