#!/usr/bin/env python3
"""WQ101 operator library (operators) — fully vectorized, sized for ~48000 rows x 20 cols.

Design discipline:
- No python-lambda rolling.apply in the hot path (48k x 20 would be painfully slow).
  Every rolling-window operator is vectorized with numpy.lib.stride_tricks.sliding_window_view,
  or uses native pandas/Cython (rolling.corr / rolling.cov / rolling.sum, etc.).
- PIT (point-in-time) discipline: only rolling / shift / cross-section (axis=1). No center=True,
  no lookahead. sliding_window_view anchors each window at its last element (right-aligned),
  matching pandas rolling — the window [t-d+1 .. t] lands at time t and contains no future bar.

Input contract:
- Operators take wide DataFrames: index=DatetimeIndex (time), columns=symbols (cross-section).
- Time-series operators (ts_*, decay_linear, product) roll along axis=0 (time), per column.
- Cross-sectional operators (rank_cs, scale) operate along axis=1 (across symbols at one time).
- NaN policy: warmup NaN passes through; never fillna/0/9999 (0 pollutes stats and gets ranked
  to an extreme; 9999 collapses the cross-section into a tied block). A time-series window that
  contains a NaN yields NaN; downstream IC valid masks filter cell by cell.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# bottleneck is not assumed present; everything runs on numpy. Probe kept so a future
# environment with bottleneck can auto-accelerate.
try:  # pragma: no cover - environment probe
    import bottleneck as _bn
    HAS_BOTTLENECK = True
except Exception:  # noqa: BLE001
    _bn = None
    HAS_BOTTLENECK = False


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _as_2d(df):
    """Return (values, index, columns). values is a float64 2D ndarray."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("ops expects a wide DataFrame input")
    return df.to_numpy(dtype=float), df.index, df.columns


def _windows_axis0(a, d):
    """Right-aligned sliding window along axis=0. Returns a (rows-d+1, ncols, d) view;
    the first d-1 rows have no window.

    sliding_window_view(a, d, axis=0) -> shape (rows-d+1, ncols, d); the last axis is the
    window. Window i covers original rows [i .. i+d-1], i.e. "with i+d-1 as the last element".
    """
    if d < 1:
        raise ValueError("window length d must be >= 1")
    return sliding_window_view(a, window_shape=d, axis=0)


def _empty_like(a):
    out = np.full_like(a, np.nan, dtype=float)
    return out


# --------------------------------------------------------------------------- #
# Time-series rolling operators (axis=0)
# --------------------------------------------------------------------------- #
def ts_rank(df, d):
    """Percentile rank [0, 1] of the last element among the past d values.

    Definition (uniform across columns, monotone consistency over absolute scale):
        ts_rank = (#{win < last} + 0.5 * #{win == last}) / d
    i.e. the "midpoint rank percentile" of the last element. Strictly-less fraction plus
    half the tie fraction, range (0, 1]. All-equal window = 0.5. A window containing NaN
    (or a NaN last element) yields NaN.

    Note: bottleneck.move_rank outputs [-1, 1]; if used it must be rescaled to (r+1)/2 to
    match this definition. Here we implement directly with numpy sliding_window_view.
    """
    a, idx, cols = _as_2d(df)
    out = _empty_like(a)
    if a.shape[0] < d:
        return pd.DataFrame(out, index=idx, columns=cols)
    win = _windows_axis0(a, d)                  # (R-d+1, ncols, d)
    last = win[:, :, -1]                        # last element (R-d+1, ncols)
    finite = np.isfinite(win)
    last_finite = np.isfinite(last)
    less = np.sum((win < last[:, :, None]) & finite, axis=2)
    eq = np.sum((win == last[:, :, None]) & finite, axis=2)
    valid_cnt = np.sum(finite, axis=2)          # count of finite values in window
    with np.errstate(invalid="ignore", divide="ignore"):
        rank = (less + 0.5 * eq) / valid_cnt
    # Any window containing NaN -> NaN (strict warmup NaN pass-through, no partial windows).
    full = valid_cnt == d
    rank = np.where(full & last_finite, rank, np.nan)
    out[d - 1:, :] = rank
    return pd.DataFrame(out, index=idx, columns=cols)


def decay_linear(df, d):
    """d-period linearly weighted moving average, weights [1, 2, ..., d] / sum,
    **newest bar gets the largest weight**.

    The sliding_window_view last axis window is [t-d+1 .. t]; the last slot (index -1) is the
    newest element t, which must carry the largest weight -> the weight vector np.arange(1, d+1)
    has its largest weight d at the end, so the ordering aligns naturally.
    (This is the easiest place to get backwards — the tests pin it: for a monotonically
    increasing series, decay must exceed the simple mean.) A window with NaN -> NaN.
    """
    a, idx, cols = _as_2d(df)
    out = _empty_like(a)
    if a.shape[0] < d:
        return pd.DataFrame(out, index=idx, columns=cols)
    w = np.arange(1, d + 1, dtype=float)
    w /= w.sum()                                # last w[-1] = d/sum is largest -> newest bar largest weight
    win = _windows_axis0(a, d)                  # (R-d+1, ncols, d)
    val = np.tensordot(win, w, axes=([2], [0]))  # dot product -> (R-d+1, ncols)
    # A window with NaN propagates to NaN via tensordot; mask again to guard boundaries.
    has_nan = np.any(~np.isfinite(win), axis=2)
    val = np.where(has_nan, np.nan, val)
    out[d - 1:, :] = val
    return pd.DataFrame(out, index=idx, columns=cols)


def ts_argmax(df, d):
    """Lookback offset of the max in window [t-d+1 .. t] from the current bar
    (0 = max at current bar, d-1 = at window's oldest).

    Implementation: argmax gives the in-window position [0..d-1], converted to "lookback from
    the last element" = (d-1) - pos. Matches WQ101 ts_argmax semantics (how long ago the max
    occurred). A window with NaN -> NaN.
    """
    return _ts_arg(df, d, np.argmax, want_max=True)


def ts_argmin(df, d):
    """Lookback offset of the min in window (0 = at current bar). A window with NaN -> NaN."""
    return _ts_arg(df, d, np.argmin, want_max=False)


def _ts_arg(df, d, argfn, want_max):
    a, idx, cols = _as_2d(df)
    out = _empty_like(a)
    if a.shape[0] < d:
        return pd.DataFrame(out, index=idx, columns=cols)
    win = _windows_axis0(a, d)                  # (R-d+1, ncols, d)
    has_nan = np.any(~np.isfinite(win), axis=2)
    pos = argfn(win, axis=2)                     # [0..d-1], 0 = oldest in window
    lookback = (d - 1) - pos                     # 0 = newest bar, d-1 = oldest
    lookback = np.where(has_nan, np.nan, lookback.astype(float))
    out[d - 1:, :] = lookback
    return pd.DataFrame(out, index=idx, columns=cols)


def ts_sum(df, d):
    """d-period rolling sum (pandas Cython; a window with NaN -> NaN)."""
    return df.rolling(d, min_periods=d).sum()


def ts_min(df, d):
    return df.rolling(d, min_periods=d).min()


def ts_max(df, d):
    return df.rolling(d, min_periods=d).max()


def ts_std(df, d):
    return df.rolling(d, min_periods=d).std()


def ts_mean(df, d):
    return df.rolling(d, min_periods=d).mean()


def delta(df, d):
    """x[t] - x[t-d]."""
    return df - df.shift(d)


def delay(df, d):
    """x[t-d] (lag by d periods)."""
    return df.shift(d)


def product(df, d):
    """Window product of d elements (vectorized; a window with NaN -> NaN)."""
    a, idx, cols = _as_2d(df)
    out = _empty_like(a)
    if a.shape[0] < d:
        return pd.DataFrame(out, index=idx, columns=cols)
    win = _windows_axis0(a, d)
    has_nan = np.any(~np.isfinite(win), axis=2)
    prod = np.prod(win, axis=2)
    prod = np.where(has_nan, np.nan, prod)
    out[d - 1:, :] = prod
    return pd.DataFrame(out, index=idx, columns=cols)


# --------------------------------------------------------------------------- #
# Rolling correlation / covariance (native pandas Cython, faster and steadier than hand-rolled)
# --------------------------------------------------------------------------- #
def corr(x, y, d):
    """Per-column (per-symbol) rolling d-period Pearson correlation. x, y same-shape wide DataFrames."""
    return x.rolling(d, min_periods=d).corr(y)


def cov(x, y, d):
    """Per-column rolling d-period covariance."""
    return x.rolling(d, min_periods=d).cov(y)


# --------------------------------------------------------------------------- #
# Cross-sectional operators (axis=1)
# --------------------------------------------------------------------------- #
def rank_cs(df):
    """Cross-sectional rank (axis=1), ranking across symbols at one time. NaN does not
    participate in ranking (pandas default) and stays NaN."""
    return df.rank(axis=1)


def rank_cs_pct(df):
    """Cross-sectional percentile rank [0, 1] (axis=1, pct=True)."""
    return df.rank(axis=1, pct=True)


def scale(df, a=1.0):
    """Cross-sectional normalization: x * a / sum(|x|) (axis=1). Denominator 0 -> whole row NaN.

    WQ101 scale normalizes each cross-section's L1 norm to a (default 1). NaN is skipped in the
    abs-sum (pandas skipna, no pollution), but an all-NaN row gives a NaN denominator -> NaN.
    """
    denom = df.abs().sum(axis=1, skipna=True)
    denom = denom.replace(0.0, np.nan)           # denominator 0 -> NaN (WQ101 degenerate guard)
    return df.mul(a, axis=0).div(denom, axis=0)


def signed_power(df, e):
    """sign(x) * |x|^e. Sign-preserving power, avoids NaN from a negative base raised to an even power."""
    return np.sign(df) * df.abs().pow(e)


def product_cs(df):
    """(Placeholder alias to avoid confusion with the time-series product.) No cross-sectional
    product is needed, so it is not implemented."""
    raise NotImplementedError("WQ101 product is a time-series product, use product(df, d)")


# --------------------------------------------------------------------------- #
# Forward (reference-library) arg operators — does not touch the existing ts_argmax/ts_argmin
# --------------------------------------------------------------------------- #
# The existing ts_argmax/ts_argmin use a 'lookback offset' convention (0=current bar, d-1=oldest),
# pinned by the tests as this project's standard, used across all batches. The most widely cited
# reference library convention uses '1=oldest, d=newest'; the two are direction-consistent (they
# differ only by a constant shift, absorbed by downstream ts_rank/decay), and overall sign-flipped
# vs this library's 'lookback' convention. The _fwd variants below match the reference-library
# direction, for the few alphas that need to align with it; the rest of the batches keep the
# existing 'lookback' operators.
def ts_argmax_fwd(df, d):
    """Forward arg convention: max at oldest bar -> 1, at newest bar -> d.

    Derived from the existing ts_argmax lookback offset lb=(d-1)-pos: pos+1 = d - lb.
    Overall sign-flipped vs the existing ts_argmax (constant d shift + negation), NaN passes through.
    """
    return d - ts_argmax(df, d)


def ts_argmin_fwd(df, d):
    """Forward arg convention: min at oldest bar -> 1, at newest bar -> d.

    Derived from the existing ts_argmin lookback offset lb: pos+1 = d - lb. Overall sign-flipped
    vs the existing ts_argmin.
    """
    return d - ts_argmin(df, d)
