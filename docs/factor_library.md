# Factor library reference

A brief reference to the factors shipped with alpha-gauntlet. All factors are
computed from OHLCV only (no external/alternative data) and are point-in-time:
the value at time `t` uses only information available up to and including `t`.

Factors are evaluated cross-sectionally: at each timestamp, the factor ranks the
instruments in the universe, and that ranking is correlated with forward returns
(Rank IC). Directions below are *intuitions*; the actual sign is decided
empirically by the IC, never assumed.

> Small-universe honesty: with a universe of ~6-30 instruments, a single-period
> IC is extremely noisy. Only multi-period ICIR / t-stats are meaningful, and
> even then in-sample significance is not out-of-sample edge.

## Curated example factors

A minimal, representative set to start with (each maps to one or more shipped
factors):

| Type | Example | Shipped factor(s) | Idea |
|---|---|---|---|
| Momentum | ROC / mom | `mom_24h`, `mom_72h`, `mom_168h` | Past winners keep winning (the strongest empirical crypto factor). |
| Volatility | ATR% / realised vol | `atr_pct`, `vol_24h`, `vol_72h` | Volatility regime; low- vs high-vol premium (sign empirical). |
| Mean-reversion | short reversal / OU | `rev_6h`, `ou_rev_press` | Very short momentum tends to reverse; OU reversion pressure toward a rolling mean. |
| WQ101 | alpha005 | `wq_a005` | Open vs 10-period VWAP mean deviation, rank-gated by `|close - vwap|`. |
| WQ101 | alpha020 | `wq_a020` | Open-gap strength: product of three ranks of open vs prior H/C/L (negated). |
| WQ101 | alpha024 | `wq_a024` | Conditional reversal: regress to a 100-period low when a long trend is shallow, else 3-period reversal. |

## Built-in time-domain factors

| Factor | Family | Description |
|---|---|---|
| `mom_24h` / `mom_72h` / `mom_168h` | momentum | Price rate-of-change over 24 / 72 / 168 bars. |
| `rev_6h` | mean-reversion | Negated 6-bar return (short-horizon reversal). |
| `vol_24h` / `vol_72h` | volatility | Rolling std of returns over 24 / 72 bars. |
| `atr_pct` | volatility | ATR(14) as a percentage of price (volatility regime). |
| `liq_ratio` | liquidity | Recent mean volume / longer-baseline mean volume (volume ramp). |
| `rsi_14` | oscillator | RSI(14), overbought/oversold. |
| `dist_ema200` | trend | Distance of price from EMA(200) (trend position). |
| `ema_slope` | trend | (EMA20 - EMA50) / EMA50 (trend direction). |
| `vol_price` | confirmation | Momentum x volume change (price-volume agreement). |
| `amihud_illiq_168h` | microstructure | Amihud illiquidity: `\|ret\| / dollar-volume`, 168-bar mean, log-compressed. |
| `downside_var_ratio_168h` | microstructure | Downside semivariance / total variance (sell-pressure release). |
| `tail_ratio_168h` | distribution | P95 / \|P05\| of returns (right-tail fatness / lottery-like). |

## Time-frequency + OU family (`wavelet_factors`)

A wavelet (stationary wavelet transform) energy family plus an OU-reversion
factor, computed strictly as **end-of-window scalars** to avoid lookahead (a
single SWT over the trailing window, reading only the coefficient aligned with
the window end -- the window physically contains no future bars).

- `ou_rev_press` -- OU-reversion pressure: AR(1) reversion of de-meaned log price
  toward a rolling mean. The one directional member of the family that scores as
  an independent dimension.
- Wavelet energy-ratio columns (`wav_*`) are produced for **diagnostic** use
  (e.g. spectral entropy as a cross-sectional noise monitor) and are not part of
  the scored set, because empirically they behaved as non-linear re-skins of
  time-domain factors -- a textbook re-skin kill (see methodology stage 5).

## Formulaic alpha library (`wq101`)

A faithful transcription of formulas from WorldQuant's "101 Formulaic Alphas"
(Kakushadze, 2016; arXiv:1601.00991), plus blind-generated custom alphas that
survived the gauntlet. These are cross-sectional formula factors: they require a
wide OHLCV panel (`panel_io.load_field_panel`) and operate over the whole
universe at once.

- Operators live in `ops.py` / `ops_ext.py`; a small DSL evaluator in `dsl.py`
  cross-checks hand-written implementations against parsed formulas.
- `pit_check.py` provides point-in-time verification helpers; `reskin_screen.py`
  implements the re-skin correlation screen; `ic_eval.py` scores panel alphas.
- Adopted WQ101 survivors include `wq_a005`, `wq_a020`, `wq_a024`, `wq_a077`.

> The number and identity of *adopted* (scored) factors are an output of the
> gauntlet, not a fixed list -- factors enter the scored set only after surviving
> the full pipeline in `docs/methodology.md`. Treat the names above as examples.

## Adding a factor

1. Implement it point-in-time (no future leakage). Add a PIT check.
2. Evaluate it with `factor_eval` to get IC / ICIR / t-stat.
3. Do **not** trust a single in-sample t-stat. Run it through the gauntlet
   (pre-register, anchor the threshold to the number tried, check sign stability,
   correlation to incumbents, re-skin, recent decay).
4. Only ratchet-promoted survivors should influence anything downstream.
