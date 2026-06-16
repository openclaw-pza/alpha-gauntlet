#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic public OHLCV downloader (ccxt).

Fetches **public** candlestick data from any ccxt-supported exchange and writes
it to the local data directory in the layout the alphagauntlet package expects:

    <DATA_DIR>/<SYMBOL_with_slash_as_underscore>-<TIMEFRAME>.feather

e.g. ``./data/BTC_USDT-1h.feather``

This script is intentionally limited to **public market data**. It never uses an
API key, secret, password, or account endpoint, and it places no orders. It only
calls the public ``fetch_ohlcv`` endpoint with paging.

Examples:
    # Download 1h candles for a few symbols from Binance into ./data
    python scripts/fetch_data.py --exchange binance --symbols BTC/USDT ETH/USDT --tf 1h

    # Two years of 4h candles, write CSV instead of feather, custom data dir
    python scripts/fetch_data.py -e kraken -s BTC/USDT --tf 4h --since 2024-01-01 \
        --format csv --data-dir ./mydata

    # Route through a SOCKS proxy if your network needs one
    python scripts/fetch_data.py -e okx -s BTC/USDT --tf 1h --proxy socks5://127.0.0.1:1080
"""
import argparse
import os
import sys
import time
from datetime import datetime, timezone

try:
    import ccxt
except ImportError:
    sys.exit("ccxt is required: pip install ccxt")

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required: pip install pandas")


# Default data directory; override with --data-dir or the DATA_DIR env var.
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", os.path.join(".", "data"))

COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _parse_since(since):
    """Accept an ISO date (YYYY-MM-DD) or epoch-ms int, return epoch-ms or None."""
    if since is None:
        return None
    s = str(since).strip()
    if s.isdigit():
        return int(s)
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"could not parse --since value: {since!r}")


def make_exchange(name, proxy=None, timeout_ms=20000):
    """Build a ccxt exchange instance for PUBLIC data only (no credentials)."""
    if not hasattr(ccxt, name):
        sys.exit(f"unknown exchange: {name!r} (see `python -c \"import ccxt; print(ccxt.exchanges)\"`)")
    cfg = {"enableRateLimit": True, "timeout": timeout_ms}
    if proxy:
        # ccxt accepts socks/http proxies; map to the right field.
        if proxy.startswith("socks"):
            cfg["socksProxy"] = proxy
        else:
            cfg["httpProxy"] = proxy
    return getattr(ccxt, name)(cfg)


def fetch_ohlcv(ex, symbol, tf, since_ms=None, limit=1000, max_bars=None):
    """Page through public fetch_ohlcv until now (or max_bars). Returns a DataFrame."""
    all_rows = []
    seen_last = None
    tf_ms = ex.parse_timeframe(tf) * 1000
    cursor = since_ms
    while True:
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe=tf, since=cursor, limit=limit)
        except ccxt.BaseError as e:
            print(f"  [warn] {symbol} {tf}: {type(e).__name__}: {e}", file=sys.stderr)
            break
        if not batch:
            break
        # Drop any overlap with the previous page.
        if seen_last is not None:
            batch = [r for r in batch if r[0] > seen_last]
            if not batch:
                break
        all_rows.extend(batch)
        seen_last = batch[-1][0]
        cursor = seen_last + tf_ms
        if max_bars and len(all_rows) >= max_bars:
            all_rows = all_rows[:max_bars]
            break
        if len(batch) < limit:
            break  # reached the most recent candle
        time.sleep((ex.rateLimit or 200) / 1000.0)
    if not all_rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    return df


def write_df(df, symbol, tf, data_dir, fmt="feather"):
    os.makedirs(data_dir, exist_ok=True)
    base = symbol.replace("/", "_") + f"-{tf}"
    if fmt == "csv":
        path = os.path.join(data_dir, base + ".csv")
        df.to_csv(path, index=False)
    else:
        path = os.path.join(data_dir, base + ".feather")
        df.reset_index(drop=True).to_feather(path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download public OHLCV via ccxt.")
    ap.add_argument("-e", "--exchange", default="binance",
                    help="ccxt exchange id (default: binance)")
    ap.add_argument("-s", "--symbols", nargs="+", required=True,
                    help="symbols, e.g. BTC/USDT ETH/USDT")
    ap.add_argument("--tf", "--timeframe", dest="tf", default="1h",
                    help="timeframe, e.g. 1h 4h 1d (default: 1h)")
    ap.add_argument("--since", default=None,
                    help="start date YYYY-MM-DD or epoch-ms (default: exchange max history)")
    ap.add_argument("--max-bars", type=int, default=None,
                    help="cap total bars per symbol")
    ap.add_argument("--limit", type=int, default=1000,
                    help="bars per request page (default: 1000)")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="output directory (default: ./data or $DATA_DIR)")
    ap.add_argument("--format", choices=("feather", "csv"), default="feather",
                    help="output format (default: feather)")
    ap.add_argument("--proxy", default=None,
                    help="optional proxy, e.g. socks5://127.0.0.1:1080 or http://...")
    args = ap.parse_args(argv)

    ex = make_exchange(args.exchange, proxy=args.proxy)
    since_ms = _parse_since(args.since)
    print(f"[fetch] exchange={args.exchange} tf={args.tf} -> {args.data_dir}")
    written = []
    for sym in args.symbols:
        print(f"[fetch] {sym} ...")
        df = fetch_ohlcv(ex, sym, args.tf, since_ms=since_ms,
                         limit=args.limit, max_bars=args.max_bars)
        if df.empty:
            print(f"  [skip] {sym}: no data")
            continue
        path = write_df(df, sym, args.tf, args.data_dir, fmt=args.format)
        written.append(path)
        print(f"  [ok] {len(df)} bars -> {path}")
    print(f"[fetch] done, {len(written)} file(s) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
