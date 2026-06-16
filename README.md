# alpha-gauntlet

A factor-research framework for cross-sectional quantitative strategies, built
around one opinionated idea: **most factor discovery is overfitting, so the
framework's job is to make survival expensive.**

It gives you four things:

1. **Factor evaluation** -- cross-sectional Rank IC, ICIR and (autocorrelation-aware)
   t-stats over a small-universe panel, with honest small-sample caveats baked in.
2. **A PIT-correct backtester** -- a point-in-time factor-channel simulator that
   scores instruments each bar, opens/closes on ATR stops/targets/trailing, and
   reports per-year expectancy, win rate, payoff and drawdown.
3. **A factor library** -- momentum / volatility / mean-reversion / microstructure
   factors, a time-frequency (wavelet) + OU-reversion family, and a faithful
   transcription of WorldQuant's "101 Formulaic Alphas" plus blind-generated
   custom alphas.
4. **An anti-overfit ratchet evolution engine** -- the centrepiece. A
   tamper-evident, monotonic promotion gate that only lets a candidate become
   the new champion if it *strictly* beats the current one across multiple
   independent out-of-sample segments, survives a never-touched holdout, and
   clears a significance threshold that tightens with the number of attempts.

> **Research framework, not financial advice.** Nothing here is a trading system,
> a signal service, or a profit guarantee. The whole point of the gauntlet is to
> show how often "great" factors are statistical mirages. In-sample significance
> is not out-of-sample edge, and out-of-sample backtest edge is not live edge.
> Use at your own risk.

---

## Why "gauntlet"?

If you optimise a filter against one backtest and try enough variants, one will
look brilliant by pure chance -- a *false champion*. The defining methodology of
this project is to run every candidate through a gauntlet designed to kill false
champions:

1. **Blind generation** -- candidates are generated without touching the IC /
   historical data they will be judged against.
2. **Pre-registration + hash-freeze** -- the candidate set and its significance
   threshold are frozen and hashed *before* evaluation, so the bar cannot move.
3. **Bonferroni-anchored t-threshold** -- the survival threshold is set for the
   number of candidates tested, not one.
4. **Multi-round survival** -- full-sample significance, then year-by-year sign
   stability, then low correlation to incumbents.
5. **Decay-lens kill-shot** -- a re-skin check (is it just a non-linear copy of an
   existing factor?) and a recent-decay check (has its edge already faded?).
6. **Walk-forward ratchet** -- promotion only on a strict, monotonic improvement
   over the current champion across non-overlapping OOS segments + a holdout,
   recorded in an append-only hash-chain ledger that can be replayed and audited.

See [`docs/methodology.md`](docs/methodology.md) for the full write-up.

---

## Install

Requires Python >= 3.10.

```bash
pip install -e .
```

TA-Lib is a C library; install the native lib first (e.g. `brew install ta-lib`,
`apt-get install ta-lib`, or a prebuilt wheel) before `pip install TA-Lib`.

The evolution engine itself only needs numpy and is importable without TA-Lib /
PyWavelets.

## Quickstart

```python
from alphagauntlet.evolution import RatchetEngine, GauntletConfig

# You supply a backtest-eval callable: (strats, timerange) -> result dict.
def my_backtest(strats, timerange):
    ...  # run your simulator, return {"strategy": {name: {"trades": [...]}}}

engine = RatchetEngine(my_backtest, state_dir="./state")
engine.init_genesis()

# Propose a candidate filter; the gauntlet decides whether to promote it.
result = engine.promote({"min_adx": 25}, evidence_summary="trend filter")
print(result["promoted"], result.get("reason_code"))
```

For factor work:

```python
from alphagauntlet import factor_eval

report = factor_eval.score_all(tf="1h", save=False)   # needs a data panel; see examples/
```

## Examples

- [`examples/01_evaluate_factors.py`](examples/01_evaluate_factors.py) -- compute a
  handful of factors on sample data and score IC / ICIR / t-stat.
- [`examples/02_backtest.py`](examples/02_backtest.py) -- run the factor backtester on a
  factor set.
- [`examples/03_gauntlet.py`](examples/03_gauntlet.py) -- watch the ratchet evolution
  gauntlet promote a genuine improvement and reject a false champion, with no
  external backtester (a synthetic eval is included).

`03_gauntlet.py` is the best place to start: it runs end-to-end with zero data
and zero optional dependencies.

## Data

Factor/backtest modules read OHLCV from a configurable data directory (env
`ALPHAGAUNTLET_DATA_DIR`, default `./data`), one `SYMBOL-TIMEFRAME.feather` file
per instrument with columns `date, open, high, low, close, volume`. No data is
shipped; bring your own.

## Layout

```
alphagauntlet/
  evolution.py        ratchet evolution engine (anti-overfit gauntlet)
  factor_eval.py      cross-sectional factor evaluation
  wavelet_factors.py  wavelet + OU-reversion factors
  scoring.py          panel scoring from validated weights
  factor_backtest.py  PIT-correct factor-channel backtester
  regime.py           Hurst / regime detection
  wq101/              WorldQuant-101 + custom formulaic alpha library
examples/             runnable end-to-end demos
tests/                pytest suite
docs/                 methodology + factor reference
```

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party attributions.
