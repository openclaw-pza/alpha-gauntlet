#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PIT-correct factor-channel backtester.

Replays a weighted cross-sectional factor channel (entry / exit) over history
so you can sweep execution parameters and look for positive expectancy.

Fidelity:
* Scoring uses the same factor panel as :mod:`alphagauntlet.factor_eval`
  (``_load_panel`` + the four ``_inject_*`` composites) and the same weighting
  formula as :mod:`alphagauntlet.scoring`.
* Fees are a flat 0.1% per side.
* The bear-regime / ADX / RSI / ATR series are reconstructed causally with
  TA-Lib from historical bars — no look-ahead.

``cfg`` options:
    regime_gate    : True = do not open new positions on bars where the
                     benchmark (BTC) is in a bear regime (market-wide gate).
    per_coin_bear  : True = do not open a coin while it is in its own bear regime.
    adx_min        : None or float; require the coin's ADX >= this to enter.
    adx_max        : None or float; require the coin's ADX <= this to enter
                     (only open low-ADX / ranging coins, mean-reversion home turf).
    rsi_max        : None or float; require the coin's RSI <= this (avoid chasing).
    sl_mult/tp_mult: stop-loss / take-profit ATR multiples.
    trail_enabled  : bool; trail_trigger_r: open profit (in R) that arms the trail;
                     trail_atr: trail distance = peak - trail_atr * ATR.
    tp_cap         : True = fixed take-profit; False = trailing only (let winners run).
    max_positions  : max concurrent positions; stake_frac: fraction of equity per position.
    score_type     : "meanrev" to score on the mean-reversion signal instead of factors.
    hold_bars      : optional time-stop; close at market once held this many bars.
"""
import numpy as np
import pandas as pd
import talib

from alphagauntlet import factor_eval as fe
from alphagauntlet import scoring as ls

FEE = 0.001
INIT_CASH = 100.0
WARMUP = 250   # deepest lookback (EMA200 / factors)


def _wide(panel, col):
    return pd.DataFrame({s: panel[s][col] for s in panel if col in panel[s]})


def prep(n_tail=None):
    """One-shot preparation: historical weighted factor score[t, coin] plus
    indicators and the bear regime. Returns a dict of wide DataFrames."""
    print(f"[bt] loading factor panel (_load_panel + 4 inject, n_tail={n_tail}) ...")
    panel = fe._load_panel(tf="1h", n_tail=n_tail)
    fe._inject_composites(panel)
    fe._inject_wq101(panel, tf="1h", n_tail=n_tail)
    fe._inject_llm1(panel, tf="1h", n_tail=n_tail)
    fe._inject_r2(panel, tf="1h", n_tail=n_tail)
    syms = list(panel)
    close = _wide(panel, "close").sort_index()
    idx, cols = close.index, close.columns

    # Learning weights (the persisted factor-weight document).
    wdoc = ls.gauntlet_weights()
    if not wdoc:
        raise RuntimeError("factor weights unavailable")
    weights = wdoc
    print(f"[bt] weighted factors {len(weights)}: {sorted(weights)}")

    # Per-bar weighted pct-rank score (same formula as scoring:
    # sum(w * (pct - 50) / 50) / sum(|w_present|)).
    score_num = pd.DataFrame(0.0, index=idx, columns=cols)
    wsum = pd.DataFrame(0.0, index=idx, columns=cols)
    for f, w in weights.items():
        wf = _wide(panel, f)
        if wf.empty:
            print(f"[bt] !! factor {f} has no column, skipping"); continue
        wf = wf.reindex(index=idx, columns=cols)
        pct = wf.rank(axis=1, pct=True) * 100.0
        contrib = w * (pct - 50.0) / 50.0
        present = wf.notna()
        score_num = score_num.add(contrib.where(present, 0.0), fill_value=0.0)
        wsum = wsum.add(present.astype(float) * abs(w), fill_value=0.0)
    score = score_num / wsum.replace(0.0, np.nan)

    # Mean-reversion score (for ranging regimes): -z_score
    # (z = (close - MA24) / STD24); buys the most oversold coins.
    NMR = 24
    _ma = close.rolling(NMR, min_periods=NMR).mean()
    _sd = close.rolling(NMR, min_periods=NMR).std()
    mr_score = -((close - _ma) / (_sd + 1e-9))

    # Raw high/low come from the field panel (the factor panel carries close +
    # factors only, not raw OHLCV).
    from alphagauntlet.wq101 import panel_io as _pio
    fields = _pio.load_field_panel(n_tail=n_tail)
    high = fields["high"].reindex(index=idx, columns=cols)
    low = fields["low"].reindex(index=idx, columns=cols)
    # ATR reconstructed from the atr_pct factor (atr_pct = ATR/close*100, same
    # source as the scorecard); RSI from the rsi_14 factor.
    atr = (_wide(panel, "atr_pct").reindex(index=idx, columns=cols) / 100.0) * close
    rsi = _wide(panel, "rsi_14").reindex(index=idx, columns=cols)
    adx = pd.DataFrame(index=idx, columns=cols, dtype=float)
    bear = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for s in cols:
        c = close[s].to_numpy(float); h = high[s].to_numpy(float); lo = low[s].to_numpy(float)
        if np.isfinite(c).sum() < WARMUP:
            continue
        adx[s] = talib.ADX(h, lo, c, 14)
        e200 = talib.EMA(c, 200)
        e200s = pd.Series(e200, index=idx)
        rising = e200s > e200s.shift(25)
        bear[s] = ((close[s] < e200s) | (~rising)).astype(float)

    btc = "BTC/USDT"
    bear_btc = bear[btc] if btc in bear.columns else pd.Series(0.0, index=idx)
    return {"idx": idx, "cols": list(cols), "close": close, "high": high, "low": low,
            "atr": atr, "adx": adx, "rsi": rsi, "bear": bear, "bear_btc": bear_btc,
            "score": score, "mr_score": mr_score}


def simulate(P, cfg):
    idx = P["idx"]; cols = P["cols"]
    close, high, low = P["close"], P["high"], P["low"]
    atr, adx, rsi, bear, bear_btc, score = P["atr"], P["adx"], P["rsi"], P["bear"], P["bear_btc"], P["score"]
    sl_mult = cfg["sl_mult"]; tp_mult = cfg["tp_mult"]
    maxpos = cfg.get("max_positions", 5); stake_frac = cfg.get("stake_frac", 0.19)
    adx_min = cfg.get("adx_min"); rsi_max = cfg.get("rsi_max"); adx_max = cfg.get("adx_max")
    trail = cfg.get("trail_enabled", False); trig_r = cfg.get("trail_trigger_r", 1.5); trail_atr = cfg.get("trail_atr", 1.0)
    tp_cap = cfg.get("tp_cap", True); hold_bars = cfg.get("hold_bars")
    score_key = "mr_score" if cfg.get("score_type") == "meanrev" else "score"

    cash = INIT_CASH
    pos = {}     # sym -> dict(entry, qty, sl, tp, r0, peak, regime_bear_at_open)
    trades = []
    n = len(idx)
    arr = {k: P[k] for k in ("close", "high", "low", "atr", "adx", "rsi", "bear", "score", "mr_score")}
    npz = {k: arr[k].to_numpy(float) for k in arr}
    cidx = {s: i for i, s in enumerate(cols)}
    bear_btc_np = bear_btc.to_numpy(float)
    eq_curve = []

    def _equity(t):
        v = cash
        for sy, pp in pos.items():
            cpx = npz["close"][t, cidx[sy]]
            v += pp["qty"] * (cpx if np.isfinite(cpx) else pp["entry"])
        return v

    for t in range(WARMUP, n):
        # 1) manage open positions (use this bar's high/low for triggers; a
        #    position opened on t-1 can be triggered on this bar).
        for sym in list(pos):
            j = cidx[sym]
            hi = npz["high"][t, j]; lo = npz["low"][t, j]
            if not (np.isfinite(hi) and np.isfinite(lo)):
                continue
            p = pos[sym]
            # time-stop: close at market once held hold_bars, before SL/TP.
            if hold_bars and (t - p["t_open"]) >= hold_bars:
                cpx = npz["close"][t, j]
                if np.isfinite(cpx):
                    proceeds = p["qty"] * cpx * (1 - FEE)
                    trades.append({"sym": sym, "pnl": proceeds - p["stake"], "pnl_pct": (proceeds - p["stake"]) / p["stake"],
                                   "reason": "horizon", "bear_open": p["bear_open"], "t_open": p["t_open"], "t_close": t})
                    cash += proceeds; del pos[sym]
                continue
            # trailing: once open profit reaches trig_r * R, ratchet the stop up.
            if trail:
                a = npz["atr"][t, j]
                if np.isfinite(a):
                    rr = (hi - p["entry"]) / p["r0"] if p["r0"] > 0 else 0.0
                    if rr >= trig_r:
                        p["sl"] = max(p["sl"], hi - trail_atr * a)
            hit = None; px = None
            if lo <= p["sl"]:                      # stop-loss first (conservative)
                hit = "stop_loss"; px = p["sl"]
            elif tp_cap and p["tp"] is not None and hi >= p["tp"]:
                hit = "take_profit"; px = p["tp"]
            if hit:
                proceeds = p["qty"] * px * (1 - FEE)
                pnl = proceeds - p["stake"]
                trades.append({"sym": sym, "pnl": pnl, "pnl_pct": pnl / p["stake"],
                               "r": pnl / (p["stake"] * (p["entry"] - p["sl0"]) / p["entry"]) if p["entry"] > p["sl0"] else np.nan,
                               "reason": hit, "bear_open": p["bear_open"], "t_open": p["t_open"], "t_close": t})
                cash += proceeds
                del pos[sym]

        # 2) regime gate + opens (wrapped in if, no early continue, so every bar
        #    samples equity).
        gated = bool(cfg.get("regime_gate")) and bear_btc_np[t] == 1.0
        if not gated and len(pos) < maxpos:
            cand = []
            for s in cols:
                if s in pos:
                    continue
                j = cidx[s]
                sc = npz[score_key][t, j]; px = npz["close"][t, j]; a = npz["atr"][t, j]
                if not (np.isfinite(sc) and np.isfinite(px) and np.isfinite(a) and a > 0):
                    continue
                if sc <= 0:                            # score > 0 quality gate
                    continue
                if cfg.get("per_coin_bear") and npz["bear"][t, j] == 1.0:
                    continue
                if adx_min is not None and not (np.isfinite(npz["adx"][t, j]) and npz["adx"][t, j] >= adx_min):
                    continue
                if adx_max is not None and not (np.isfinite(npz["adx"][t, j]) and npz["adx"][t, j] <= adx_max):
                    continue  # only open low-ADX ranging coins (mean-reversion home turf)
                if rsi_max is not None and not (np.isfinite(npz["rsi"][t, j]) and npz["rsi"][t, j] <= rsi_max):
                    continue
                cand.append((sc, s, px, a))
            cand.sort(reverse=True)
            stake = stake_frac * _equity(t)            # stake an equity fraction (avoids fixed-size drift)
            for sc, s, px, a in cand:
                if len(pos) >= maxpos or cash < stake or stake < 1.0:
                    break
                sl = px - sl_mult * a
                tp = (px + tp_mult * a) if tp_cap else None
                if sl >= px:
                    continue
                qty = stake * (1 - FEE) / px
                cash -= stake
                pos[s] = {"entry": px, "qty": qty, "stake": stake, "sl": sl, "sl0": sl, "tp": tp,
                          "r0": px - sl, "bear_open": bear_btc_np[t] == 1.0, "t_open": t}
        if t % 24 == 0:
            eq_curve.append(_equity(t))

    # force-close remainder (at last close)
    for sym in list(pos):
        j = cidx[sym]; px = npz["close"][n - 1, j]
        if np.isfinite(px):
            p = pos[sym]; proceeds = p["qty"] * px * (1 - FEE); pnl = proceeds - p["stake"]
            trades.append({"sym": sym, "pnl": pnl, "pnl_pct": pnl / p["stake"], "r": np.nan,
                           "reason": "eod", "bear_open": p["bear_open"], "t_open": p["t_open"], "t_close": n - 1})
            cash += proceeds
    eq_curve.append(cash)
    return _stats(trades, eq_curve, cfg, idx)


def _stats(trades, eq_curve, cfg, idx=None):
    final_eq = eq_curve[-1] if eq_curve else INIT_CASH
    peak, mdd = -1e9, 0.0
    for e in eq_curve:
        peak = max(peak, e)
        if peak > 0:
            mdd = max(mdd, (peak - e) / peak)
    n = len(trades)
    if n == 0:
        return {"name": cfg.get("_name", ""), "n": 0, "final_eq": round(final_eq, 2), "note": "no trades"}
    pcts = np.array([t["pnl_pct"] for t in trades])    # per-trade return (% of stake), comparable
    wins = pcts > 0
    nw = int(wins.sum()); wr = nw / n
    avg_w = float(pcts[wins].mean()) if nw else 0.0
    avg_l = float(pcts[~wins].mean()) if (n - nw) else 0.0
    bear_tr = [t for t in trades if t["bear_open"]]
    bull_tr = [t for t in trades if not t["bear_open"]]
    def _wr(ts):
        return (round(sum(1 for x in ts if x["pnl"] > 0) / len(ts) * 100, 1),
                round(sum(x["pnl_pct"] for x in ts) * 100, 1)) if ts else (0, 0)
    bwr, bpnl = _wr(bear_tr); ulwr, ulpnl = _wr(bull_tr)
    per_year = {}
    if idx is not None:
        import pandas as _pd
        ybuf = {}
        for tr in trades:
            y = int(_pd.Timestamp(idx[tr["t_close"]]).year)
            ybuf.setdefault(y, []).append(tr["pnl_pct"])
        per_year = {y: round(float(np.mean(v)) * 100, 3) for y, v in sorted(ybuf.items())}
    return {"name": cfg.get("_name", ""), "n": n, "winrate": round(wr * 100, 1), "per_year_exp_pct": per_year,
            "final_eq": round(final_eq, 2), "ret_pct": round((final_eq / INIT_CASH - 1) * 100, 1),
            "max_dd": round(mdd * 100, 1), "exp_pct": round(float(pcts.mean()) * 100, 3),
            "payoff": round(avg_w / abs(avg_l), 2) if avg_l else None,
            "breakeven_wr": round(abs(avg_l) / (avg_w + abs(avg_l)) * 100, 1) if (avg_w and avg_l) else None,
            "bear_n": len(bear_tr), "bear_wr": bwr, "bear_pnlpct": bpnl,
            "bull_n": len(bull_tr), "bull_wr": ulwr, "bull_pnlpct": ulpnl}


if __name__ == "__main__":
    P = prep()
    print(f"[bt] panel {len(P['idx'])} bars x {len(P['cols'])} coins\n")
    # Baseline = a plain long-only config (regime gate off / no entry filters /
    # sl 1.5 tp 2.0 / no trailing).
    base = {"_name": "BASELINE", "regime_gate": False, "per_coin_bear": False,
            "adx_min": None, "rsi_max": None, "sl_mult": 1.5, "tp_mult": 2.0,
            "trail_enabled": False, "tp_cap": True, "max_positions": 5, "stake_frac": 0.19}
    r = simulate(P, base)
    print("=== baseline run ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
