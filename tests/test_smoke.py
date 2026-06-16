#!/usr/bin/env python3
"""Smoke tests: the package imports and the lightweight engine surface is present."""
import importlib

import pytest


def test_package_imports():
    import alphagauntlet
    assert alphagauntlet.__version__


def test_evolution_always_importable():
    # The engine depends only on numpy and must import without TA-Lib/PyWavelets.
    from alphagauntlet.evolution import GauntletConfig, RatchetEngine
    cfg = GauntletConfig()
    assert cfg.key_limit >= 1
    assert callable(RatchetEngine.promote)


@pytest.mark.parametrize("mod", ["factor_eval", "wavelet_factors", "scoring",
                                 "factor_backtest", "regime"])
def test_factor_modules_import_or_skip(mod):
    # These need TA-Lib/PyWavelets; skip cleanly if the optional deps are absent.
    try:
        importlib.import_module(f"alphagauntlet.{mod}")
    except ImportError as e:
        pytest.skip(f"optional dependency missing for {mod}: {e}")
