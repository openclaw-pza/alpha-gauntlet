# Contributing

Thanks for your interest in alpha-gauntlet.

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

TA-Lib needs its native C library installed first (see the README install note).

## Before you open a PR

- Run the formatters/linters: `pre-commit run --all-files` (ruff + black).
- Run the tests: `pytest`.
- Keep functions small and PIT-correct: any factor or backtest code must not use
  future information. If you add a factor, add a point-in-time check.
- New factors should go through the gauntlet methodology
  ([`docs/methodology.md`](docs/methodology.md)) -- don't add a factor to the
  scored set on the strength of one in-sample t-stat.

## Scope and ground rules

- This is a **research framework, not financial advice** and not a live trading
  system. PRs that add live order execution, exchange account handling, or
  anything that turns this into a trading bot are out of scope.
- Do not commit market data, API keys, or any secrets. The `.gitignore` already
  excludes `data/`, `state/`, `*.feather` and `*.parquet`.

## Reporting issues

Open a GitHub issue with a minimal reproducer. For factor/backtest bugs, include
the data shape and a short script.
