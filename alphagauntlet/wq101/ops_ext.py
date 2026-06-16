#!/usr/bin/env python3
# Ported from QuantaAlpha (https://github.com/QuantaAlpha/QuantaAlpha), MIT License, Copyright (c) Ziyi Tang et al.
"""WQ101 extended operator library (ops_ext) — QuantaAlpha function_lib semantics ported to wide DataFrames.

License: QuantaAlpha's pyproject.toml classifiers declare "License :: OSI Approved :: MIT License".
The MIT license permits free copy/modify/distribute provided the source and original license are
noted in the file header, so this file uses a "port + attribution" model. Each operator's docstring
notes the source function_lib.py line range plus semantic notes.

Design discipline (consistent with ops.py):
- No python-lambda rolling.apply in the hot path (48k x 20 would be painfully slow). The original
  QuantaAlpha makes heavy use of groupby('instrument').transform(rolling.apply(..., raw=False/True));
  in the wide format here it is rewritten with sliding_window_view (fully vectorized) or native
  pandas/numpy (rolling.median/quantile/sum).
- PIT discipline: only rolling / shift / cross-section (axis=1). No center=True, no lookahead.
  sliding_window_view's last axis window is right-aligned (anchored at the last element), matching pandas rolling.
- NaN policy: warmup NaN passes through, never fillna/0/9999. A window with NaN in a time-series
  higher-moment operator yields NaN (unless that operator's semantics are explicitly nan-omit; see docstrings).

Direction conventions (easiest to get backwards, pinned per operator):
- Cross-sectional operators (SKEW/KURT/MEDIAN/ZSCORE): axis=1, across symbols at one time. Orthogonal to the time-series ts_*.
- Time-series operators (ts_kurt/ts_skew/ts_median/percentile/ts_quantile/wma/ts_mad/regbeta/regresi):
  axis=0, per column (per symbol) rolling along time.
- The original QuantaAlpha uses groupby('datetime')=cross-section, groupby('instrument')=time-series;
  the port maps these to axis=1 / axis=0 respectively.

min_periods conventions (faithful to original semantics):
- Higher moments (skew needs >=3, kurt needs >=4): NaN at that point if the window has too few valid values.
- COUNT/SUMIF/percentile/ts_quantile/ts_median/ts_mad/wma: original min_periods=1 (lax), but per the
  pitfall note COUNT should be strict-window min_periods=p; the rest keep the original lax semantics
  and note it in the docstring.
- REGBETA/REGRESI: strict min_periods=p, no partial-window regression (partial-window slope is unstable).
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# --------------------------------------------------------------------------- #
# Internal helpers (same shape as ops.py; ops is not imported, to keep this file self-contained)
# --------------------------------------------------------------------------- #
def _as_2d(df):
    """Return (values, index, columns). values is a float64 2D ndarray."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("ops_ext expects a wide DataFrame input")
    return df.to_numpy(dtype=float), df.index, df.columns


def _windows_axis0(a, d):
    """Right-aligned sliding window along axis=0. Returns a (rows-d+1, ncols, d) view; first d-1 rows have no window.

    sliding_window_view(a, d, axis=0) -> window i covers original rows [i .. i+d-1], last element = i+d-1.
    """
    if d < 1:
        raise ValueError("window length d must be >= 1")
    return sliding_window_view(a, window_shape=d, axis=0)


def _empty_like(a):
    return np.full_like(a, np.nan, dtype=float)


def _nan_skew_kurt_lastaxis(arr, min_valid, want):
    """nan-omit skewness/kurtosis along the last axis, fully vectorized (no python-per-window loop).

    Numerically equivalent to scipy.stats.{skew,kurtosis}(bias=True, nan_policy='omit'):
        m_k = mean((x - mean)^k)   (biased central moment; each slice uses its non-NaN count n as denominator)
        skew      = m3 / m2^1.5
        kurtosis  = m4 / m2^2 - 3  (Fisher excess kurtosis)
    A slice with fewer than min_valid valid values -> NaN. m2==0 (constant slice) -> NaN (matches scipy 0/0).

    arr: any shape, aggregated over the last axis. Returns arr.shape[:-1].
    This is the key to replacing per-window scipy calls: it collapses ~737k scalar scipy calls into a few whole-array numpy ops.
    """
    finite = np.isfinite(arr)
    n = finite.sum(axis=-1).astype(float)            # valid count per slice (0 -> division below gives NaN)
    x = np.where(finite, arr, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = x.sum(axis=-1) / n
        xc = np.where(finite, arr - mean[..., None], 0.0)   # demean; set NaN positions to 0 so they don't enter the moment
        m2 = (xc ** 2).sum(axis=-1) / n
        if want == "skew":
            m3 = (xc ** 3).sum(axis=-1) / n
            res = m3 / np.power(m2, 1.5)
        else:  # kurt (Fisher)
            m4 = (xc ** 4).sum(axis=-1) / n
            res = m4 / (m2 ** 2) - 3.0
    res = np.asarray(res, dtype=float)
    res[n < min_valid] = np.nan
    res[~np.isfinite(m2) | (m2 == 0.0)] = np.nan     # degenerate denominator (constant/invalid) -> NaN
    return res


# =========================================================================== #
# Cross-sectional operators (axis=1, across symbols at one time)
# =========================================================================== #
def skew_cs(df):
    """Cross-sectional skewness: skewness of the factor distribution across all symbols at one time.

    Source: QuantaAlpha function_lib.py L48-52 SKEW (originally groupby('datetime') cross-section).
    Semantics: scipy.stats.skew(row.dropna(), bias=True), nan_policy equivalent to omit; a cross-section
    with < 3 non-NaN values -> NaN (skewness is meaningless for <3 samples). Opposite direction to the time-series ts_skew.
    Implementation: fully vectorized (_nan_skew_kurt_lastaxis nan-omit central moment along axis=1), no
    per-row scipy call; numerically equivalent to scipy.stats.skew(bias=True) (pinned by tests). The
    scalar result is broadcast across that row's columns by convention.
    """
    a, idx, cols = _as_2d(df)
    row_skew = _nan_skew_kurt_lastaxis(a, min_valid=3, want="skew")   # (nrow,)
    return pd.DataFrame(np.repeat(row_skew[:, None], a.shape[1], axis=1), index=idx, columns=cols)


def kurt_cs(df):
    """Cross-sectional kurtosis: Fisher excess kurtosis (normal=0) across all symbols at one time.

    Source: QuantaAlpha function_lib.py L54-61 KURT (originally groupby('datetime') cross-section).
    Semantics: scipy.stats.kurtosis(row.dropna(), fisher=True, bias=True); a cross-section with < 4
    non-NaN values -> NaN. Note: the original repo L58-60 has an uncalled dead-code calc_kurt, not ported.
    Implementation: fully vectorized (_nan_skew_kurt_lastaxis, want='kurt'), no per-row scipy call.
    """
    a, idx, cols = _as_2d(df)
    row_kurt = _nan_skew_kurt_lastaxis(a, min_valid=4, want="kurt")
    return pd.DataFrame(np.repeat(row_kurt[:, None], a.shape[1], axis=1), index=idx, columns=cols)


def median_cs(df):
    """Cross-sectional median: median across symbols at one time (df.median(axis=1), skipna).

    Source: QuantaAlpha function_lib.py L73-76 MEDIAN (originally groupby('datetime') cross-section).
    Semantics: each cross-section takes the median of all non-NaN symbols, scalar broadcast across the row's
    columns. Different direction from the time-series ts_median.
    """
    med = df.median(axis=1, skipna=True)
    # Broadcast back to wide (same value per symbol); an all-NaN row -> NaN for that row.
    return pd.DataFrame(
        np.repeat(med.to_numpy()[:, None], df.shape[1], axis=1),
        index=df.index, columns=df.columns,
    )


def zscore_cs(df):
    """Cross-sectional Z-score standardization: (x - cross-section mean) / cross-section std (across symbols at one time).

    Source: QuantaAlpha function_lib.py L583-588 ZSCORE (originally groupby('datetime') cross-section).
    Semantics: axis=1; ddof=1 (pandas std default, matching the original groupby.std()). A continuous
    alternative to RANK, friendlier to linear models. A cross-section with std=0 (all equal) -> inf/NaN
    for that row (pandas division rule), not guarded for parity with the original; downstream IC valid
    masks filter it. Different direction from the time-series TS_ZSCORE.
    """
    mean = df.mean(axis=1, skipna=True)
    std = df.std(axis=1, skipna=True)            # ddof=1
    return df.sub(mean, axis=0).div(std, axis=0)


# =========================================================================== #
# Time-series rolling operators (axis=0, per column)
# =========================================================================== #
def ts_kurt(df, p):
    """Time-series rolling kurtosis: Fisher excess kurtosis of each symbol over the past p bars.

    Source: QuantaAlpha function_lib.py L79-88 TS_KURT. The original uses rolling.apply(raw=False) feeding
    a Series to per-window scipy; 47k x 20 is painfully slow (measured 280s+). This version uses
    sliding_window_view + vectorized central moments, no python-per-window loop, numerically equivalent to
    scipy.stats.kurtosis(fisher=True, bias=True).
    Semantics: computed only when the window has >= 4 valid values, else NaN (original min_periods=min(4,p)
    plus a len>=4 guard). Fisher excess kurtosis (normal=0). A window with NaN -> computed over the window's
    non-NaN values (nan-omit, matching the original nan_policy='omit'), but still NaN if valid values < 4.
    """
    return _ts_moment(df, p, want="kurt", min_valid=4)


def ts_skew(df, p):
    """Time-series rolling skewness: distribution skewness of each symbol over the past p bars.

    Source: QuantaAlpha function_lib.py L90-99 TS_SKEW. The original uses rolling.apply(raw=False) per-window
    scipy (poor performance, measured 260s+); this version uses sliding_window_view + vectorized central
    moments, numerically equivalent to scipy.stats.skew(bias=True).
    Semantics: computed only when the window has >= 3 valid values, else NaN. nan-omit (uses the window's non-NaN values).
    """
    return _ts_moment(df, p, want="skew", min_valid=3)


def _ts_moment(df, p, want, min_valid):
    """Common implementation for time-series rolling higher moments: vectorized nan-omit central moment over the sliding-window last axis.

    Key performance fix: no per-window scipy call (the original ~737k scalar calls -> 280s); instead
    _nan_skew_kurt_lastaxis computes the moment over the entire (R-p+1, ncols, p) window tensor along the
    last axis in one pass. nan-omit semantics: each window computes the moment over its internal non-NaN
    values; valid values < min_valid -> NaN.
    """
    a, idx, cols = _as_2d(df)
    out = _empty_like(a)
    nrow = a.shape[0]
    if nrow < p:
        return pd.DataFrame(out, index=idx, columns=cols)
    win = _windows_axis0(a, p)                     # (R-p+1, ncols, p)
    res = _nan_skew_kurt_lastaxis(win, min_valid=min_valid, want=want)   # (R-p+1, ncols)
    out[p - 1:, :] = res
    return pd.DataFrame(out, index=idx, columns=cols)


def ts_median(df, p):
    """Time-series rolling median: median of each symbol over the past p bars.

    Source: QuantaAlpha function_lib.py L121-124 TS_MEDIAN. Native pandas rolling.median (Cython).
    Semantics: original min_periods=1 (lax, partial windows during warmup). This version keeps
    min_periods=1 to match. Different direction from the cross-sectional median_cs (this is time-series, axis=0).
    """
    return df.rolling(p, min_periods=1).median()


def percentile(df, p, q):
    """Time-series rolling quantile: q-quantile of each symbol over the past p bars (q in [0,1]).

    Source: QuantaAlpha function_lib.py L126-136 PERCENTILE. The original degenerates to a full-history
    quantile when p=None (lookahead!); this version implements only the rolling variant (p required) and
    does not inherit the lookahead branch.
    Semantics: original min_periods=1. q must be in [0,1]. Native pandas rolling.quantile.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"percentile q must be in [0,1], got {q}")
    return df.rolling(p, min_periods=1).quantile(q)


def ts_quantile(df, p, q):
    """Time-series rolling arbitrary quantile, args (df, p, q), q in [0,1].

    Source: QuantaAlpha function_lib.py L610-619 TS_QUANTILE. The original has a p/q auto-swap side effect
    (if p is float and q is int, they get swapped); this clean-room version drops that side effect and
    requires strict types.
    Semantics: effectively the same as percentile (both rolling.quantile), but kept under QuantaAlpha's two
    distinct names for DSL-call compatibility. Original min_periods=1.
    """
    p = int(p)
    q = float(q)
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"ts_quantile q must be in [0,1], got {q}")
    if p < 1:
        raise ValueError(f"ts_quantile window p must be >= 1, got {p}")
    return df.rolling(p, min_periods=1).quantile(q)


def wma(df, p):
    """Exponentially weighted moving average: weights 0.9^i (i=0 is the newest bar, largest weight=1).

    Source: QuantaAlpha function_lib.py L291-300 WMA. The original weights=[0.9**i for i in range(p)][::-1],
    i.e. [0.9^(p-1), ..., 0.9^1, 0.9^0=1] (old->new), with the newest bar (last element) carrying the
    largest weight (=1). Note this differs from DECAYLINEAR's linear weights [1..p] (WMA is geometric decay).
    Implementation: clean-room sliding_window_view + weight-vector dot product, avoiding the original
    rolling.apply. Weight vector w[i]=0.9^(p-1-i) (i=0..p-1, old->new), with w[-1]=0.9^0=1 largest, aligning
    naturally with sliding_window_view's last axis (old->new, last=newest). Normalized by sum(w).
    Semantics: original min_periods=1 (warmup uses weights[:len(window)] over partial windows). This version
    uses strict warmup: a window with NaN or fewer than p elements -> NaN (warmup pass-through), consistent
    with ops.decay_linear, to avoid partial-window weight-normalization discrepancies.
    """
    a, idx, cols = _as_2d(df)
    out = _empty_like(a)
    if a.shape[0] < p:
        return pd.DataFrame(out, index=idx, columns=cols)
    # weights: old->new = 0.9^(p-1), ..., 0.9^0; last is largest
    w = np.power(0.9, np.arange(p - 1, -1, -1, dtype=float))   # shape (p,), w[-1]=1
    w = w / w.sum()
    win = _windows_axis0(a, p)                     # (R-p+1, ncols, p)
    val = np.tensordot(win, w, axes=([2], [0]))    # (R-p+1, ncols)
    has_nan = np.any(~np.isfinite(win), axis=2)
    val = np.where(has_nan, np.nan, val)
    out[p - 1:, :] = val
    return pd.DataFrame(out, index=idx, columns=cols)


def ts_mad(df, p):
    """Time-series rolling median absolute deviation: MAD = median(|x - median(x)|) over a p-period window.

    Source: QuantaAlpha function_lib.py L597-607 TS_MAD. The original uses a rolling.apply(raw=True) lambda,
    slow on 48k x 20; this version uses sliding_window_view (median over the last axis, then absolute
    deviation, then median). More robust to outliers than ts_std.
    Semantics: original min_periods=1 (warmup uses partial windows, np.median valid for partial windows).
    This version uses strict warmup: a window with NaN or fewer than p elements -> NaN (consistent with the
    higher-moment operators, avoiding confusion from np.median returning NaN on NaN).
    """
    a, idx, cols = _as_2d(df)
    out = _empty_like(a)
    if a.shape[0] < p:
        return pd.DataFrame(out, index=idx, columns=cols)
    win = _windows_axis0(a, p)                     # (R-p+1, ncols, p)
    has_nan = np.any(~np.isfinite(win), axis=2)
    med = np.median(win, axis=2, keepdims=True)    # (R-p+1, ncols, 1)
    mad = np.median(np.abs(win - med), axis=2)     # (R-p+1, ncols)
    mad = np.where(has_nan, np.nan, mad)
    out[p - 1:, :] = mad
    return pd.DataFrame(out, index=idx, columns=cols)


# =========================================================================== #
# Days-since operators (HIGHDAY / LOWDAY) — 1..p convention
# =========================================================================== #
def highday(df, p):
    """Days since the highest value: distance from the bar holding the window max to the current bar, returns 1..p.

    Source: QuantaAlpha function_lib.py L349-357 HIGHDAY. The original highday = len(window) - argmax, no -1,
    so a max at the newest bar -> p - (p-1) = 1; a max at the oldest bar -> p - 0 = p.
    Important: this convention (1..p, newest=1) differs by 1 from ops.ts_argmax's 'lookback offset' (0=current bar).
    Clean-room implementation: sliding_window_view + argmax(last axis) -> pos (0=oldest, p-1=newest),
    highday = p - pos (newest pos=p-1 -> 1; oldest pos=0 -> p). A window with NaN -> NaN.
    Tie handling: np.argmax takes the first (oldest) max position, matching the original np.argmax.
    """
    return _day_since(df, p, np.argmax)


def lowday(df, p):
    """Days since the lowest value: distance from the bar holding the window min to the current bar, returns 1..p.

    Source: QuantaAlpha function_lib.py L359-367 LOWDAY. The original lowday = len(window) - argmin. A min at
    the newest bar -> 1; at the oldest bar -> p. Clean-room: lowday = p - argmin(last axis). Note the 1-offset
    difference from ops.ts_argmin's 0-offset lookback convention. A window with NaN -> NaN.
    """
    return _day_since(df, p, np.argmin)


def _day_since(df, p, argfn):
    a, idx, cols = _as_2d(df)
    out = _empty_like(a)
    if a.shape[0] < p:
        return pd.DataFrame(out, index=idx, columns=cols)
    win = _windows_axis0(a, p)                     # (R-p+1, ncols, p), last axis 0=oldest, p-1=newest
    has_nan = np.any(~np.isfinite(win), axis=2)
    pos = argfn(win, axis=2)                        # 0=oldest, p-1=newest (ties take the first=oldest)
    day = (p - pos).astype(float)                  # newest(pos=p-1)->1, oldest(pos=0)->p
    day = np.where(has_nan, np.nan, day)
    out[p - 1:, :] = day
    return pd.DataFrame(out, index=idx, columns=cols)


# =========================================================================== #
# Conditional operators (COUNT / SUMIF / FILTER) — input boolean/multipliable DataFrame
# =========================================================================== #
def count(cond, p):
    """Conditional count: number of True occurrences over the past p bars (equivalent to ts_sum(cond, p)).

    Source: QuantaAlpha function_lib.py L302-307 COUNT. The original min_periods=1; but per the pitfall note,
    conditional counting should be strict-window min_periods=p (a warmup window with fewer than p bars should
    not give a partial count). This version uses min_periods=p (strict), deviating from the original lax
    min_periods=1; rationale in the docstring.
    Semantics: cond is a boolean DataFrame (produced by GT/LT comparison operators), True=1/False=0,
    rolling.sum. A window with NaN (a boolean frame should have no NaN in theory, but for robustness) -> NaN at that point.
    """
    return cond.astype(float).rolling(p, min_periods=p).sum()


def sumif(df, p, cond):
    """Conditional rolling sum: sum of df values where cond=True over the past p bars = ts_sum(df*cond, p).

    Source: QuantaAlpha function_lib.py L309-314 SUMIF. Essentially (df*cond).rolling(p).sum().
    Semantics: original min_periods=1. This version uses min_periods=p (strict window, consistent with count,
    avoiding partial sums during warmup). Aligns index/columns before the conditional multiply (reindex_like to
    prevent misalignment). cond is a boolean frame.
    """
    cond_aligned = cond.reindex_like(df).astype(float)
    return df.mul(cond_aligned).rolling(p, min_periods=p).sum()


def filter_(df, cond):
    """Conditional filter: keep the original value where cond=True, set 0 where cond=False (df.mul(cond)).

    Source: QuantaAlpha function_lib.py L316-321 FILTER. Unlike WHERE's three branches, FILTER has only two:
    True->original value / False->0. In a DSL it is commonly used to mask abnormal price ranges.
    Semantics: cond is a boolean frame, multiplied element-wise after aligning index/columns. NaN*bool still
    passes through NaN. The trailing underscore avoids the python builtin filter.
    """
    cond_aligned = cond.reindex_like(df).astype(float)
    return df.mul(cond_aligned)


# =========================================================================== #
# Element-wise / utility
# =========================================================================== #
def sign(df):
    """Element-wise sign function: np.sign(x), returns -1/0/+1 (NaN passes through).

    Source: QuantaAlpha function_lib.py L272-275 SIGN. A one-line np.sign. ops has signed_power but no bare
    SIGN; in DSLs sign($close-$open) to judge candle direction is common.
    """
    return pd.DataFrame(np.sign(df.to_numpy(dtype=float)), index=df.index, columns=df.columns)


def sequence(n):
    """Sequence generator: returns a length-n float vector [1, 2, ..., n] (np.arange(1,n+1)).

    Source: QuantaAlpha function_lib.py L370-375 SEQUENCE (originally np.linspace(1,n,n)). Not a DataFrame
    operator but a numpy utility, used as the x-axis for time-series linear regression
    REGBETA(SEQUENCE(p), $close, p).
    Note: the original uses np.linspace(...,dtype=float32); this version uses float64 (arange) for regression
    precision, numerically equivalent.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"sequence n must be >= 1, got {n}")
    return np.arange(1, n + 1, dtype=float)


# =========================================================================== #
# Time-series rolling OLS (REGBETA / REGRESI) — most complex, strict min_periods=p
# =========================================================================== #
def _prep_reg_x(x, p, nrow, ncol):
    """Regularize the regression independent variable x into a window sequence.

    Supports two forms of x:
    1. A SEQUENCE vector (1D ndarray, length p): every window shares the same fixed x (matching the
       original df2=Series(df2[:p])). Returns ('fixed', x_vec) -- x_vec shape (p,).
    2. A wide DataFrame (same shape as y): per column per window take x's sliding window (matching the
       original branch where x is also a DataFrame). Returns ('wide', x_values) -- x_values shape (nrow, ncol).
    """
    if isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != p:
            raise ValueError(f"REG: fixed x vector length ({x.shape[0]}) must == window p ({p})")
        return "fixed", x
    if isinstance(x, pd.DataFrame):
        xv = x.to_numpy(dtype=float)
        if xv.shape != (nrow, ncol):
            raise ValueError(f"REG: wide x shape {xv.shape} must match y {(nrow, ncol)}")
        return "wide", xv
    raise TypeError("REG x must be a 1D ndarray (SEQUENCE) or a wide DataFrame")


def _ols_beta_resid(Y, X):
    """OLS y ~ x + 1 over a batch of windows simultaneously, returns (beta, last_resid).

    Y: (M, p) one y window per row; X: (M, p) the matching x windows (may be broadcast from a fixed vector).
    Uses the closed-form (covariance) solution rather than per-window lstsq loops, to vectorize the whole batch:
        beta = cov(x,y)/var(x), intercept = mean(y) - beta*mean(x)
        last_resid = y[:,-1] - (beta*x[:,-1] + intercept)
    Mathematically equivalent to the original np.linalg.lstsq([x,1], y) OLS slope/residual (single-regressor
    OLS closed-form is unique). var(x)=0 (x degenerate to a constant, e.g. all-NaN filled) -> beta=NaN
    (filtered downstream).
    """
    p = Y.shape[1]
    xm = X.mean(axis=1, keepdims=True)
    ym = Y.mean(axis=1, keepdims=True)
    xc = X - xm
    yc = Y - ym
    sxx = np.sum(xc * xc, axis=1)                  # (M,)
    sxy = np.sum(xc * yc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = sxy / sxx
    intercept = ym[:, 0] - beta * xm[:, 0]
    last_resid = Y[:, -1] - (beta * X[:, -1] + intercept)
    return beta, last_resid


def _rolling_reg(y_df, x, p, want):
    """REGBETA/REGRESI common skeleton. want in {'beta','resid'}. Strict min_periods=p, a window with NaN->NaN."""
    a, idx, cols = _as_2d(y_df)
    nrow, ncol = a.shape
    out = _empty_like(a)
    if nrow < p:
        return pd.DataFrame(out, index=idx, columns=cols)
    mode, xprep = _prep_reg_x(x, p, nrow, ncol)
    ywin = _windows_axis0(a, p)                     # (R-p+1, ncol, p)
    R = ywin.shape[0]
    if mode == "fixed":
        # fixed x vector, shared by every window; broadcast to (R, ncol, p)
        xwin = np.broadcast_to(xprep, (R, ncol, p))
    else:
        xwin = _windows_axis0(xprep, p)            # (R-p+1, ncol, p)
    # A y or x window containing NaN -> NaN (strict warmup, no partial/fillna)
    bad = np.any(~np.isfinite(ywin), axis=2) | np.any(~np.isfinite(xwin), axis=2)  # (R, ncol)
    Yf = ywin.reshape(R * ncol, p)
    Xf = xwin.reshape(R * ncol, p)
    beta, resid = _ols_beta_resid(Yf, Xf)
    val = (beta if want == "beta" else resid).reshape(R, ncol)
    val = np.where(bad, np.nan, val)
    out[p - 1:, :] = val
    return pd.DataFrame(out, index=idx, columns=cols)


def regbeta(y, x, p):
    """Time-series rolling OLS slope: slope beta of y on x over a p-period window.

    Source: QuantaAlpha function_lib.py L387-451 (calculate_beta + rolling_beta + REGBETA). The original uses
    joblib parallelism + dual mode (when x is ndarray/Series, a fixed p-sample x shared by all windows; when x
    is a DataFrame, per-instrument alignment). This clean-room version serves only wide DataFrames, with a
    closed-form OLS vectorized over the whole batch, no joblib.
    Args: y=wide DataFrame; x=1D vector returned by SEQUENCE(p) (fixed x, trend regression) or a wide DataFrame
    same shape as y (per-window x sliding window); p=window length.
    PIT: strict min_periods=p, no partial-window regression (unstable slope). A window with NaN -> NaN.
    Right-aligned window (last element = current bar).
    Use: regbeta(sequence(p) <-> close, p) gives a time-series linear trend slope (momentum/reversal).
    """
    return _rolling_reg(y, x, p, want="beta")


def regresi(y, x, p):
    """Time-series rolling OLS residual: y - y_hat at the end of a p-period window (y_hat=beta*x+intercept).

    Source: QuantaAlpha function_lib.py L455-538 (calculate_residuals + rolling_residuals + REGRESI). Same
    structure as REGBETA but outputs the end-of-window residual instead of the slope. The original has a
    MultiIndex alignment branch; this clean-room version focuses on wide (x as wide or a SEQUENCE vector),
    outputting the window's last residual.
    Semantics: residual = y[t] - (beta*x[t] + intercept), beta/intercept from this p-period window's OLS.
    Strict min_periods=p, a window with NaN -> NaN. Commonly used for detrended reversal signals
    regresi(close, sequence(p), p).
    """
    return _rolling_reg(y, x, p, want="resid")
