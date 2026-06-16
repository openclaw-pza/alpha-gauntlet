#!/usr/bin/env python3
"""WQ101 runner — compute the factor panel / export existing factors / run the three-round elimination gauntlet.

CLI:
    python -m alphagauntlet.wq101.runner --batch N
        import batchN.py's ALPHAS, compute each factor wide panel (uniformly replace +/-inf->NaN after),
        store <OUT_DIR>/wq101_values_batchN.parquet (column names wq{id}), print per-factor full-sample IC/t.

    python -m alphagauntlet.wq101.runner --batch-mod batch0_smoke --tag smoke
        generic entry: specify the module name (default batch{N}) and the artifact tag (default batch{N}), for smoke reuse.

    python -m alphagauntlet.wq101.runner --existing
        store the panel values of the current scorecard factors (>=16 raw + composites) to <OUT_DIR>/wq101_existing_values.parquet.

    python -m alphagauntlet.wq101.runner --gauntlet
        load wq101_values_batch[1-9].parquet + existing, run the pre-registered three-round elimination,
        store the result in <REPORT_DIR>/wq101_gauntlet.md + <OUT_DIR>/wq101_gauntlet_results.json.

IC/t math reuses ic_eval (verbatim transcription from factor_eval), no self-invented simplification.
Degenerate guard: skip a section with valid symbols<8 or cross-sectional zero variance; a factor with >50% timestamps skipped is marked DEGENERATE.

Output directories are configurable via the ALPHAGAUNTLET_OUT_DIR / ALPHAGAUNTLET_REPORT_DIR environment
variables (defaults ./out and ./reports relative to the current working directory).
"""
import argparse
import glob
import importlib
import json
import os

import numpy as np
import pandas as pd

from alphagauntlet.wq101 import ic_eval, panel_io

OUT_DIR = os.environ.get("ALPHAGAUNTLET_OUT_DIR", os.path.join(".", "out"))
REPORT_DIR = os.environ.get("ALPHAGAUNTLET_REPORT_DIR", os.path.join(".", "reports"))
PKG = "alphagauntlet.wq101"

# ---- gauntlet pre-registered parameters (hard-coded, no CLI knobs) ----
G_R1_T = 3.5          # R1 full-sample |t_neff| threshold
G_R2_RATE = 0.70      # R2 yearly same-sign rate threshold
G_R2_MINPTS = 100     # R2 minimum sampling points per year
G_R3_RHO = 0.70       # R3 correlation elimination threshold


# --------------------------------------------------------------------------- #
# Factor panel computation
# --------------------------------------------------------------------------- #
def _wide_index_columns(fields):
    return fields["close"].index, fields["close"].columns


def compute_batch(mod_name, fields):
    """Import the specified module's ALPHAS, compute each wide panel. Uniformly replace +/-inf->NaN after.

    Returns dict[col_name -> wide DataFrame]. All factors reindexed to a uniform (time x symbols).
    """
    mod = importlib.import_module(f"{PKG}.{mod_name}")
    alphas = getattr(mod, "ALPHAS", None)
    if not isinstance(alphas, dict) or not alphas:
        raise RuntimeError(f"{mod_name}.ALPHAS missing or empty")
    idx, cols = _wide_index_columns(fields)
    out = {}
    for name, fn in alphas.items():
        val = fn(fields)
        if not isinstance(val, pd.DataFrame):
            raise RuntimeError(f"factor {name} returned a non-DataFrame: {type(val)}")
        val = val.reindex(index=idx, columns=cols)
        val = val.replace([np.inf, -np.inf], np.nan)   # inf policy: uniformly clear +/-inf after compute
        out[name] = val
    return out


def _stack_to_long(factor_wides):
    """Stack {name: wide} into a (date, symbol) MultiIndex long table for parquet (col=factor name).

    parquet is awkward for many equal-shape wide frames, so flatten into a (date, symbol) x factors long
    table; on read back, set_index+unstack restores the wide form (load_values_parquet).

    pandas 3.0's DataFrame.stack no longer keeps all-NaN rows, dropping the warmup segment and breaking
    PIT/IC alignment. So this uses a numpy full-grid flatten (each date repeated N_symbols times, symbols
    tiled), guaranteeing every (date, symbol) cell exists with NaN preserved. All factors share the same
    (date, symbol) grid.
    """
    names = [n for n in factor_wides]
    ref = factor_wides[names[0]]
    dates = ref.index
    syms = list(ref.columns)
    nd, ns = len(dates), len(syms)
    date_col = np.repeat(dates.to_numpy(), ns)
    sym_col = np.tile(np.array(syms, dtype=object), nd)
    data = {"date": date_col, "symbol": sym_col}
    for name in names:
        wide = factor_wides[name].reindex(index=dates, columns=syms)
        data[name] = wide.to_numpy(dtype=float).ravel()   # row-major: aligns with repeat/tile
    long_df = pd.DataFrame(data)
    return long_df


def load_values_parquet(path):
    """Read back the long table stored by _stack_to_long, restoring {name: wide DataFrame}."""
    long_df = pd.read_parquet(path)
    long_df = long_df.set_index(["date", "symbol"])
    out = {}
    for name in long_df.columns:
        out[name] = long_df[name].unstack("symbol")
    return out


# --------------------------------------------------------------------------- #
# Existing factor export
# --------------------------------------------------------------------------- #
def export_existing(fields_close_template, n_tail=None):
    """Use factor_eval's per-symbol compute_factors + _inject_composites to get the scorecard factors, transpose to wide, store.

    At least 16 raw (FACTOR_NAMES) + 3 composites. close is also stored (so the gauntlet computes forward
    return aligned to the same close as the batch factors).
    """
    from alphagauntlet import factor_eval as fe
    panel = fe._load_panel(tf="1h", n_tail=n_tail)   # read-only reference, does not modify existing modules
    fe._inject_composites(panel)
    names = list(fe.FACTOR_NAMES) + list(fe.COMPOSITES)
    out = {}
    for f in names:
        out[f] = pd.DataFrame({s: panel[s][f] for s in panel})
    # close stored separately (column name __close__ reserved; the gauntlet uses it for forward return)
    out["__close__"] = pd.DataFrame({s: panel[s]["close"] for s in panel})
    return out, names


# --------------------------------------------------------------------------- #
# IC preview
# --------------------------------------------------------------------------- #
def preview_ic(factor_wides, close_wide, h=ic_eval.DEFAULT_H):
    """Print a full-sample IC/t preview per factor. Returns {name: eval_dict}."""
    res = {}
    print(f"\n  full-sample IC/t preview (h={h}, non-overlapping sampling, n_eff-shrunk t):")
    print(f"  {'factor':<10} {'n':>5} {'IC':>9} {'t':>8} {'skip%':>6}  verdict")
    for name, wide in factor_wides.items():
        if name == "__close__":
            continue
        ev = ic_eval.evaluate(wide, close_wide, h)
        res[name] = ev
        ic = ev.get("mean_ic")
        t = ev.get("t_stat")
        skip = ev.get("skip_frac")
        ic_s = f"{ic:+.4f}" if ic is not None else "   n/a"
        t_s = f"{t:+.2f}" if t is not None else "  n/a"
        sk_s = f"{skip*100:.0f}%" if skip is not None else " n/a"
        print(f"  {name:<10} {ev.get('n_periods', 0):>5} {ic_s:>9} {t_s:>8} {sk_s:>6}  {ev.get('verdict')}")
    return res


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def _close_from_fields(fields):
    return fields["close"]


def cmd_batch(mod_name, tag, n_tail):
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[batch] loading field panel (n_tail={n_tail}) ...")
    fields = panel_io.load_field_panel(n_tail=n_tail)
    print(f"[batch] computing {mod_name}.ALPHAS ...")
    wides = compute_batch(mod_name, fields)
    close_wide = _close_from_fields(fields)
    out_path = os.path.join(OUT_DIR, f"wq101_values_{tag}.parquet")
    long_df = _stack_to_long(wides)
    long_df.to_parquet(out_path)
    print(f"[batch] stored: {out_path}  ({len(wides)} factors, {long_df.shape[0]} long-table rows)")
    preview_ic(wides, close_wide)
    return out_path


def cmd_existing(n_tail):
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[existing] exporting current scorecard factors (n_tail={n_tail}) ...")
    out, names = export_existing(None, n_tail=n_tail)
    out_path = os.path.join(OUT_DIR, "wq101_existing_values.parquet")
    long_df = _stack_to_long(out)
    long_df.to_parquet(out_path)
    n_raw = len([n for n in names if n not in ("mom_combo", "vol_combo", "meta_combo")])
    print(f"[existing] stored: {out_path}  (raw factors {n_raw} + composites {len(names)-n_raw} + close)")
    close_wide = out["__close__"]
    preview_ic({k: v for k, v in out.items() if k != "__close__"}, close_wide)
    return out_path


# --------------------------------------------------------------------------- #
# Gauntlet three-round elimination
# --------------------------------------------------------------------------- #
def _sign(x):
    return 0 if x is None else (1 if x > 0 else (-1 if x < 0 else 0))


def _half_window_ics(fac_wide, close_wide, h):
    """Split the sampling time axis in half, returns (ic_full, ic_first, ic_second, t_full, sec).
    Reuses ic_eval.section_ic_series to get the per-section IC series then halves by sampling time.
    """
    sec = ic_eval.section_ic_series(fac_wide, close_wide, h)
    ics = sec["ics"]
    n = len(ics)
    if n < ic_eval.N_PERIODS_MIN:
        return None
    half = n // 2
    ic_full = float(ics.mean())
    ic_first = float(ics[:half].mean()) if half >= 1 else float("nan")
    ic_second = float(ics[half:].mean()) if (n - half) >= 1 else float("nan")
    _, _, _, _, t_full = ic_eval._t_from_ics(ics)
    return {"ic_full": ic_full, "ic_first": ic_first, "ic_second": ic_second,
            "t_full": float(t_full), "ics": ics, "ics_index": sec["ics_index"],
            "n": n, "skip_frac": sec["skip_frac"]}


def _yearly_signrate(ics, ics_index, ic_full_sign):
    """Per-calendar-year (only years with >=G_R2_MINPTS sampling points) IC same-sign rate vs the full sample."""
    s = pd.Series(ics, index=pd.DatetimeIndex(ics_index))
    years = s.groupby(s.index.year)
    same, total = 0, 0
    detail = {}
    for yr, grp in years:
        if len(grp) < G_R2_MINPTS:
            continue
        yr_ic = float(grp.mean())
        is_same = (_sign(yr_ic) == ic_full_sign) and ic_full_sign != 0
        detail[int(yr)] = {"n": int(len(grp)), "ic": round(yr_ic, 4), "same": bool(is_same)}
        total += 1
        same += int(is_same)
    rate = (same / total) if total else 0.0
    return rate, total, detail


def _spearman_aligned(a_wide, b_wide, close_wide, h):
    """Two factors' "cross-sectional rank then concatenate by sampling time" Spearman.

    Implementation: each does a cross-sectional pct-rank (axis=1) on non-overlapping sampling sections,
    flattened into same-length vectors (same sampling time + symbol aligned, keeping only cells finite in
    both), then Spearman. Returns |rho| or None.
    """
    idx = a_wide.index.intersection(b_wide.index).intersection(close_wide.index)
    a = a_wide.loc[idx].iloc[::h]
    b = b_wide.loc[idx].iloc[::h]
    ar = a.rank(axis=1)
    br = b.rank(axis=1)
    af = ar.to_numpy().ravel()
    bf = br.to_numpy().ravel()
    m = np.isfinite(af) & np.isfinite(bf)
    if m.sum() < 50:
        return None
    af, bf = af[m], bf[m]
    # already ranks (cross-sectional rank); rank again globally to guarantee Spearman (Pearson on ranks = Spearman)
    afr = pd.Series(af).rank().to_numpy()
    bfr = pd.Series(bf).rank().to_numpy()
    if afr.std() < 1e-9 or bfr.std() < 1e-9:
        return None
    rho = float(np.corrcoef(afr, bfr)[0, 1])
    return abs(rho)


def cmd_gauntlet():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    h = ic_eval.DEFAULT_H

    # 1) load candidates (batch1-9) + existing
    cand = {}
    batch_paths = sorted(glob.glob(os.path.join(OUT_DIR, "wq101_values_batch[1-9].parquet")))
    for p in batch_paths:
        cand.update(load_values_parquet(p))
    if not batch_paths:
        msg = "[gauntlet] no wq101_values_batch[1-9].parquet found, no candidates to eliminate. Run --batch 1.. first."
        print(msg)
        _write_gauntlet_report(msg, {}, [], [], [], {})
        return

    existing_path = os.path.join(OUT_DIR, "wq101_existing_values.parquet")
    if not os.path.exists(existing_path):
        msg = "[gauntlet] missing wq101_existing_values.parquet, run --existing first."
        print(msg)
        _write_gauntlet_report(msg, {}, [], [], [], {})
        return
    existing = load_values_parquet(existing_path)
    close_wide = existing.pop("__close__")
    incumbents = {k: v for k, v in existing.items()}

    results = {"params": {"R1_t": G_R1_T, "R2_rate": G_R2_RATE,
                          "R2_minpts": G_R2_MINPTS, "R3_rho": G_R3_RHO, "h": h},
               "n_candidates": len(cand), "n_incumbents": len(incumbents),
               "per_factor": {}, "R1_survivors": [], "R2_survivors": [],
               "R3_final": [], "R3_killed": {}}

    # ---- R1: full-sample |t|>=3.5 and first/second/full same-signed ----
    r1_survivors = []
    for name, wide in cand.items():
        hw = _half_window_ics(wide, close_wide, h)
        if hw is None:
            results["per_factor"][name] = {"stage": "drop", "reason": "insufficient samples", "t_full": None}
            continue
        sf, ss, sful = _sign(hw["ic_first"]), _sign(hw["ic_second"]), _sign(hw["ic_full"])
        cond_t = abs(hw["t_full"]) >= G_R1_T
        cond_sign = (sf == ss == sful) and sful != 0
        passed = bool(cond_t and cond_sign)
        results["per_factor"][name] = {
            "t_full": round(hw["t_full"], 2), "ic_full": round(hw["ic_full"], 4),
            "ic_first": round(hw["ic_first"], 4), "ic_second": round(hw["ic_second"], 4),
            "sign_consistent": cond_sign, "R1_pass": passed, "n": hw["n"],
            "skip_frac": round(hw["skip_frac"], 3),
        }
        if passed:
            r1_survivors.append(name)
    results["R1_survivors"] = r1_survivors

    # ---- R2: yearly same-sign rate >=70% ----
    r2_survivors = []
    for name in r1_survivors:
        hw = _half_window_ics(cand[name], close_wide, h)
        rate, ny, detail = _yearly_signrate(hw["ics"], hw["ics_index"], _sign(hw["ic_full"]))
        results["per_factor"][name]["R2_yearly"] = detail
        results["per_factor"][name]["R2_rate"] = round(rate, 3)
        results["per_factor"][name]["R2_years"] = ny
        passed = bool(ny >= 1 and rate >= G_R2_RATE)
        results["per_factor"][name]["R2_pass"] = passed
        if passed:
            r2_survivors.append(name)
    results["R2_survivors"] = r2_survivors

    # ---- R3: correlation elimination ----
    # 3a vs the existing factors: any |rho|>=0.7 -> candidate eliminated (incumbent wins)
    survivor_t = {}
    for name in r2_survivors:
        survivor_t[name] = abs(results["per_factor"][name]["t_full"])
    alive = list(r2_survivors)
    killed = {}
    for name in list(alive):
        max_rho_inc = 0.0
        worst_inc = None
        for inc_name, inc_wide in incumbents.items():
            rho = _spearman_aligned(cand[name], inc_wide, close_wide, h)
            if rho is not None and rho > max_rho_inc:
                max_rho_inc, worst_inc = rho, inc_name
        if max_rho_inc >= G_R3_RHO:
            killed[name] = {"vs": worst_inc, "rho": round(max_rho_inc, 3), "type": "incumbent"}
            alive.remove(name)
        results["per_factor"][name]["R3_max_rho_incumbent"] = round(max_rho_inc, 3)
        results["per_factor"][name]["R3_worst_incumbent"] = worst_inc

    # 3b survivors vs each other: |rho|>=0.7 -> keep the higher |t|
    alive_sorted = sorted(alive, key=lambda n: survivor_t[n], reverse=True)
    final = []
    for name in alive_sorted:
        clash = None
        for kept in final:
            rho = _spearman_aligned(cand[name], cand[kept], close_wide, h)
            if rho is not None and rho >= G_R3_RHO:
                clash = (kept, rho)
                break
        if clash is None:
            final.append(name)
        else:
            kept, rho = clash
            killed[name] = {"vs": kept, "rho": round(rho, 3), "type": "peer (|t| lower)"}
    results["R3_final"] = final
    results["R3_killed"] = killed

    # store
    json_path = os.path.join(OUT_DIR, "wq101_gauntlet_results.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2, default=str)
    print(f"[gauntlet] JSON: {json_path}")
    _write_gauntlet_report(None, results, r1_survivors, r2_survivors, final, killed)
    return results


def _write_gauntlet_report(early_msg, results, r1, r2, final, killed):
    lines = ["# WQ101 three-round elimination gauntlet report", ""]
    if early_msg:
        lines += [early_msg, ""]
        path = os.path.join(REPORT_DIR, "wq101_gauntlet.md")
        with open(path, "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines))
        print(f"[gauntlet] report: {path}")
        return
    p = results["params"]
    lines += [
        f"- main horizon h={p['h']}, non-overlapping sampling, t shrunk via n_eff (lag-1 autocorrelation) (verbatim from factor_eval)",
        f"- degenerate guard: skip a section with valid symbols<{ic_eval.MIN_COINS_SECTION} or cross-sectional zero variance; >50% skipped marked DEGENERATE",
        f"- candidates {results['n_candidates']}, incumbents {results['n_incumbents']}",
        "",
        f"## R1 (full-sample |t|>={p['R1_t']} and first/second/full-sample IC same-signed)",
        f"survivors {len(r1)}: {', '.join(r1) if r1 else '(none)'}",
        "",
        f"## R2 (years with >={p['R2_minpts']} points, IC same-sign rate >={p['R2_rate']:.0%})",
        f"survivors {len(r2)}: {', '.join(r2) if r2 else '(none)'}",
        "",
        f"## R3 (correlation elimination |rho|>={p['R3_rho']})",
        f"final selected {len(final)}: {', '.join(final) if final else '(none)'}",
        "",
        "elimination detail:",
    ]
    if killed:
        for name, info in killed.items():
            lines.append(f"- {name}: clashed with {info['vs']} (rho={info['rho']}, {info['type']})")
    else:
        lines.append("- (none)")
    lines += ["", "## per-factor diagnostics", "",
              "| factor | n | IC | t | first-half IC | second-half IC | same-sign | R1 | R2 rate | R3 max rho |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, d in results["per_factor"].items():
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, d.get("n", "-"), d.get("ic_full", "-"), d.get("t_full", "-"),
            d.get("ic_first", "-"), d.get("ic_second", "-"),
            "Y" if d.get("sign_consistent") else "N",
            "Y" if d.get("R1_pass") else "N",
            d.get("R2_rate", "-"), d.get("R3_max_rho_incumbent", "-")))
    path = os.path.join(REPORT_DIR, "wq101_gauntlet.md")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    print(f"[gauntlet] report: {path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="WQ101 runner")
    ap.add_argument("--batch", type=int, help="compute batchN.py's ALPHAS and store")
    ap.add_argument("--batch-mod", type=str, default=None,
                    help="specify the module name (default batch{N}), for batch0_smoke etc. reuse")
    ap.add_argument("--tag", type=str, default=None,
                    help="artifact tag (default batch{N}); determines wq101_values_{tag}.parquet")
    ap.add_argument("--existing", action="store_true", help="export the current scorecard factors")
    ap.add_argument("--gauntlet", action="store_true", help="run the three-round elimination")
    ap.add_argument("--n-tail", type=int, default=None,
                    help="keep the last N bars per symbol (debug speedup); default full history")
    args = ap.parse_args()

    if args.batch is not None or args.batch_mod is not None:
        if args.batch_mod is not None:
            mod = args.batch_mod
            tag = args.tag or args.batch_mod
        else:
            mod = f"batch{args.batch}"
            tag = args.tag or f"batch{args.batch}"
        cmd_batch(mod, tag, args.n_tail)
    elif args.existing:
        cmd_existing(args.n_tail)
    elif args.gauntlet:
        cmd_gauntlet()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
