"""alpha-gauntlet -- a factor-research framework with an anti-overfit ratchet.

Public surface
--------------
- ``alphagauntlet.factor_eval``      cross-sectional factor evaluation (Rank IC / ICIR / t-stat)
- ``alphagauntlet.wavelet_factors``  time-frequency (wavelet) + OU-reversion factors
- ``alphagauntlet.scoring``          panel scoring from validated factor weights
- ``alphagauntlet.factor_backtest``  PIT-correct factor-channel backtester
- ``alphagauntlet.regime``           Hurst / trend-vs-chop regime detection
- ``alphagauntlet.evolution``        anti-overfit ratchet evolution engine
- ``alphagauntlet.wq101``            WorldQuant-101 + custom formulaic alpha library

The evolution engine has no heavy dependencies and is always importable. The
factor/backtest modules require numpy/pandas/scipy and (for several factors)
TA-Lib + PyWavelets; import them directly when you need them so a missing
optional dependency does not break ``import alphagauntlet``.
"""
__version__ = "0.1.0"

__all__ = [
    "evolution",
    "factor_eval",
    "wavelet_factors",
    "scoring",
    "factor_backtest",
    "regime",
    "wq101",
]
