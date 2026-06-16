#!/usr/bin/env python3
"""WQ101 PIT truncation-invariance test — generic, for any factor in any batch.

Principle (point-in-time conservation): a lookahead-free factor's value at time t depends only on data
at [<=t]. So "chop off the tail N bars then recompute" and "compute on the full history" must be
element-for-element equal over the region both cover (the earlier region). If a factor uses future data
(center=True / shift(-k) / full-sample statistics), chopping the tail changes the earlier values -> diff!=0.

Test construction:
    full   = load_field_panel(full)            -> compute factor -> take the factor wide
    trunc  = load_field_panel(drop last 200)   -> compute factor -> take the factor wide
  comparison region = trunc's index, then drop trunc's last 300 rows (lookback buffer: after chopping, the
            last few bars' rolling windows are inherently incomplete, not lookahead, so excluded from comparison).
  requirement: full and trunc are element-for-element equal over that region (np.array_equal, NaN positions must match).

CLI:
    python -m alphagauntlet.wq101.pit_check --batch N --sample 3
        --batch N : import batchN.py (N=0 -> batch0_smoke)
        --sample K: randomly sample K factors to test (default test all); 0 or omitted = test all
        --mod NAME: directly specify the module name (overrides --batch)
"""
import argparse
import random

import numpy as np

from alphagauntlet.wq101 import panel_io, runner

TAIL_CHOP = 200       # number of tail rows chopped
LOOKBACK_BUF = 300    # extra tail rows excluded from the comparison region (covers the longest adv180/corr windows)


def _equal_with_nan(a, b, rtol=1e-9, atol=1e-12):
    """Element-for-element equal (NaN positions must match too). a, b are aligned ndarrays.

    Uses np.isclose with tolerance rather than exact == : reduction operators like decay_linear that use
    np.tensordot/prod have a BLAS sum order that depends on array total length, so full vs trunc (different
    lengths) produce ULP-level differences in the last bit (16th significant digit); exact equality would
    misjudge a clean factor as FAIL (false positive). rtol=1e-9/atol=1e-12 is far below any real leak
    magnitude (shift-leak diff magnitude >> tolerance), absorbing ULP noise without missing lookahead leaks.
    NaN positions must still match element-for-element.
    """
    both_nan = np.isnan(a) & np.isnan(b)
    close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
    ok = both_nan | close
    return bool(np.all(ok)), int(np.sum(~ok))


def check_batch(mod_name, sample_k=0, seed=0):
    """Run truncation-invariance for all (or sampled) ALPHAS of mod_name. Returns (ok, details)."""
    print("[pit] loading full field panel ...")
    fields_full = panel_io.load_field_panel(n_tail=None)
    wides_full = runner.compute_batch(mod_name, fields_full)

    print(f"[pit] loading truncated (drop last {TAIL_CHOP}) field panel ...")
    # n_tail won't do (it keeps the tail); we want to drop the tail -> manually iloc-chop the full panel and recompute
    fields_trunc = _truncate_fields(fields_full, TAIL_CHOP)
    wides_trunc = _recompute_from_truncated(mod_name, fields_trunc)

    names = list(wides_full)
    if sample_k and sample_k > 0 and sample_k < len(names):
        random.seed(seed)
        names = random.sample(names, sample_k)

    print(f"[pit] testing {len(names)} factors: {', '.join(names)}")
    details = {}
    all_ok = True
    for name in names:
        full = wides_full[name]
        trunc = wides_trunc[name]
        # comparison region = trunc's index minus the first LOOKBACK_BUF rows (rolling warmup at the series start).
        # Must cut the head, not the tail: shift(-k) lookahead leaks only perturb trunc's last k rows (boundary
        # effect); cutting the tail [:-LOOKBACK_BUF] would discard the leak rows along with legitimate tail rows
        # -> blind to all bounded leaks with k<=300; cutting the head [LOOKBACK_BUF:] keeps the tail, the leak
        # rows fall inside the comparison region and are faithfully caught.
        cmp_idx = trunc.index[LOOKBACK_BUF:] if len(trunc.index) > LOOKBACK_BUF else trunc.index[:0]
        cols = full.columns
        a = full.reindex(index=cmp_idx, columns=cols).to_numpy(dtype=float)
        b = trunc.reindex(index=cmp_idx, columns=cols).to_numpy(dtype=float)
        ok, ndiff = _equal_with_nan(a, b)
        details[name] = {"ok": ok, "n_diff": ndiff, "n_cells": int(a.size),
                         "cmp_rows": int(len(cmp_idx))}
        all_ok = all_ok and ok
        flag = "PASS" if ok else f"FAIL(diff={ndiff})"
        print(f"  {name:<10} {flag}  (cells={a.size}, rows={len(cmp_idx)})")
    return all_ok, details


def _truncate_fields(fields_full, chop):
    """Drop the last chop rows from the full field panel — derived fields must be re-derived from the *raw*
    OHLCV, otherwise returns/adv rolling windows at the chop point would not change (they are inherently
    PIT-safe), but for rigor still recompute from raw."""
    # take the raw OHLCV after chopping the tail, then re-run load_field_panel's derivation logic
    idx = fields_full["close"].index
    keep_idx = idx[:-chop] if chop > 0 and len(idx) > chop else idx
    raw = {f: fields_full[f].loc[keep_idx] for f in panel_io.RAW_FIELDS}
    out = dict(raw)
    out["vwap"] = (out["high"] + out["low"] + out["close"]) / 3.0
    out["returns"] = out["close"].pct_change()
    for k in fields_full:
        if k.startswith("adv"):
            n = int(k[3:])
            out[k] = out["volume"].rolling(n, min_periods=n).mean() * out["close"]
    return out


def _recompute_from_truncated(mod_name, fields_trunc):
    return runner.compute_batch(mod_name, fields_trunc)


def main():
    ap = argparse.ArgumentParser(description="WQ101 PIT truncation-invariance test")
    ap.add_argument("--batch", type=int, default=0, help="batchN (N=0 -> batch0_smoke)")
    ap.add_argument("--mod", type=str, default=None, help="directly specify the module name (overrides --batch)")
    ap.add_argument("--sample", type=int, default=0, help="number of factors to sample (0=test all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mod:
        mod = args.mod
    elif args.batch == 0:
        mod = "batch0_smoke"
    else:
        mod = f"batch{args.batch}"

    ok, details = check_batch(mod, sample_k=args.sample, seed=args.seed)
    print(f"\n[pit] overall verdict: {'ALL PASS' if ok else 'FAIL'}  ({mod})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
