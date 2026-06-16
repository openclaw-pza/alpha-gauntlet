#!/usr/bin/env python3
"""Time-frequency / statistical-arbitrage factor family — wavelet (SWT) energy factors + OU reversion pressure.

Why this family:
- The original time-domain factors are all time-domain statistics (momentum/volatility/RSI/ATR/EMA), with
  zero frequency-domain / time-frequency content.
- Crypto 1h data has multi-period resonance and heavy wick noise; a time-frequency decomposition can separate
  "trend vs noise vs a specific band rhythm", providing a new dimension low-correlated with time-domain factors.

Anti-lookahead (the lifeblood of this system):
- The only safe paradigm = **end-of-window scalar**: factor[t] runs a single SWT over the historical window
  close[t-W+1:t+1] and takes only the coefficient aligned with the window's end t. The window physically
  contains no future bar, and the SWT's internal periodic extension can only use in-window history -> it cannot
  leak future data (verified empirically: setting close[t+1:] to a sentinel leaves the same-window factor
  bit-identical).
- Never run one SWT over the whole series then slice (the bidirectional convolution + boundary extension would
  leak close[t+1..] into coef[t] — exactly the lookahead failure mode).
- Insufficient warmup / a gap in the window -> NaN, **never fillna/9999** (0 pollutes rolling statistics and gets
  ranked to an extreme; 9999 collapses the cross-section into a tied block). Warmup NaN is left to the IC
  evaluator's valid mask to drop cell by cell (consistent with hurst_rs returning None for len<64).

SWT configuration (validated in this environment's venv):
- Mother wavelet sym4: near-symmetric, less phase distortion than db4, friendly to "current bar alignment";
  support length 8, narrow boundary-contamination region.
- level=4: on 1h bars each level covers D1≈2-4h (wicks/microstructure noise) / D2≈4-8h / D3≈8-16h
  (funding 8h + intraday waves) / D4≈16-32h / A4≈>32h (>1-day trend), covering crypto's "few hours to over a day".
- window=256 (=2^8): SWT level=4 requires the window length divisible by 2^4=16 (W=168/200 raise ValueError;
  256/192 work); 256≈10.7 days, long enough for A4 (>32h scale) to mean something, short enough to stay locally
  time-varying, in the same order as the system's ema200/hurst n=480.
- norm=True: Parseval energy conservation (measured ratio=1.0000), required for energy-ratio factors to be meaningful.

OU family (pure pandas, not bound by the SWT divisibility constraint, hence win=168 weekly window aligned with mom_168h):
- Log price + rolling-mean demean for an AR(1) (a raw non-stationary price gives spurious regression; BTC≈100k vs
  DOGE≈0.1 magnitude differences make the cross-section incomparable).
- Only ou_rev_press (directional) enters scoring; ou_mr_speed (theta, no direction) is moved out of scoring and
  used only as a regime gate (its h1 directional signal turned out to be a theta∝1/vol low-volatility proxy,
  homogeneous with vol_24h, not an independent alpha).

Honest null statement: a wavelet energy ratio's random-walk null is not centered at 0 / not uniform (Brownian
1/f^2 spectrum naturally favors low-frequency energy), so the absolute values are not readable; only used for
cross-sectional relative ranking. True direction / strength is determined empirically by the IC sign + t-stat, not preset.
"""
import numpy as np
import pandas as pd

try:
    import pywt
except ImportError:  # pragma: no cover - degrade: without pywt the wavelet columns are all NaN, the scorer drops them naturally
    pywt = None

# ============ configuration (locked; re-run the PIT gold standard before changing) ============
WAVELET = "sym4"
LEVEL = 4
WIN = 256          # wavelet window, must be divisible by 2^LEVEL=16
OU_WIN = 168       # OU window (weekly window, pure pandas, not bound by divisibility)
MID_LAG = 12       # mid-band momentum lookback
SLOPE_K = 24       # A4 end-of-window regression length

WAVELET_COLS = ["wav_lf_energy_ratio", "wav_hf_lf_ratio", "wav_mid_band_mom",
                "wav_spectral_entropy", "wav_hf_energy_ratio", "wav_lf_slope"]
OU_COLS = ["ou_rev_press", "ou_mr_speed"]
ALL_COLS = WAVELET_COLS + OU_COLS                          # all produced columns (8)

# Columns that enter the scorecard (factor_eval.FACTOR_NAMES), decided after a three-way adversarial study:
#  - All 5 wav_* columns are moved out of scoring: at h24 all have t<2; root cause = the end-of-window scalar is
#    a nonlinear reskin of time-domain factors (mid_band_mom ~ mom_168h ρ0.59 / dist_ema200 ρ0.62), and at N=20 the
#    cross-sectional energy-spectrum shapes are highly similar, so the SNR is inherently insufficient. Three
#    improvement lines (VisuShrink denoising / A4-reconstruction momentum / spectral-entropy regime conditioning)
#    were all dropped after independent adversarial review (denoising gave 0 improvement across 12 cells; A4
#    momentum flipped sign in a later half-period and was ρ0.77 redundant; the regime-conditional IC failed all 12
#    Welch tests, same fate as the time-domain hurst/ADX). Cross-sectional alpha is a dead end here.
#  - wav_* columns are still produced as diagnostics only (spectral entropy = section noise-level monitor, zero-cost
#    observational value, not in the selection score).
#  - ou_rev_press is kept in scoring: the only independent dimension (|ρ|<=0.07/0.024 vs the wav/vol families),
#    with h1/h4/h24 t all >3.
#  - wav_lf_slope / ou_mr_speed keep their earlier verdict (redundant / low-volatility-proxy).
REGIME_DIAGNOSTIC_COLS = ["wav_lf_slope", "ou_mr_speed",
                          "wav_lf_energy_ratio", "wav_hf_lf_ratio", "wav_mid_band_mom",
                          "wav_spectral_entropy", "wav_hf_energy_ratio"]
SCORED_COLS = [c for c in ALL_COLS if c not in REGIME_DIAGNOSTIC_COLS]   # only ou_rev_press enters scoring

EPS = 1e-12


# ============ single-window SWT -> 6 scalars ============
# Key: the energy family runs on [log returns] (stationary), not on log price.
# Measured: log price is I(1) non-stationary; the low-frequency approximation A4 captures the price level/trend,
#       and the energy-ratio details span 6 orders of magnitude -> lf≈0.9999/hf≈1e-6/entropy≈0 degenerate to
#       constants (cross-sectional std two orders smaller than the returns version, no discriminative power as a factor).
# The band-pass momentum / trend slope still run on [log price] (they want price-level structure, not energy distribution).
def _swt_readouts(lr_win, lp_win):
    """lr_win=WIN log returns (stationary, feeds the energy family), lp_win=WIN log prices (feeds band-pass momentum / trend slope).
    Both have their right end = the current bar t; inside the function there is no access to the future. Returns a tuple, ordered like WAVELET_COLS."""
    # ---- energy family (log returns) ----
    cr = pywt.swt(lr_win, WAVELET, level=LEVEL, trim_approx=False, norm=True)
    # cr = [(cA4,cD4),(cA3,cD3),(cA2,cD2),(cA1,cD1)] high level (coarse) -> low level (fine), each length=WIN
    eA = float(np.dot(cr[0][0], cr[0][0]))
    eD4 = float(np.dot(cr[0][1], cr[0][1])); eD3 = float(np.dot(cr[1][1], cr[1][1]))
    eD2 = float(np.dot(cr[2][1], cr[2][1])); eD1 = float(np.dot(cr[3][1], cr[3][1]))
    total = eA + eD1 + eD2 + eD3 + eD4 + EPS
    lf_energy_ratio = eA / total                                   # low-frequency (persistent/drift) energy share
    hf_lf_ratio = float(np.log((eD1 + eD2 + EPS) / (eA + EPS)))     # high-freq oscillation vs low-freq (frequency-domain hurst)
    hf_energy_ratio = (eD1 + eD2) / total                          # high-freq noise / wick energy share
    e = np.array([eA, eD4, eD3, eD2, eD1])
    p = e / (e.sum() + EPS); p = p[p > 0]
    spectral_entropy = float(-np.sum(p * np.log(p)) / np.log(5))   # spectral entropy (energy-distribution uniformity)

    # in-symbol volatility (log-return std) standardization base: directional factors divide by it, to avoid degenerating into a volatility/tier proxy
    sd = float(np.std(lr_win)) + EPS

    # ---- band-pass momentum / trend slope (log price) ----
    cp = pywt.swt(lp_win, WAVELET, level=LEVEL, trim_approx=False, norm=True)
    cA4 = cp[0][0]
    # A4 end-of-window linear regression slope (near-zero-phase denoised trend slope, in volatility units)
    y = cA4[-SLOPE_K:]
    xx = np.arange(SLOPE_K, dtype=float)
    xc = xx - xx.mean(); yc = y - y.mean()
    den = float(np.dot(xc, xc))
    lf_slope = (float(np.dot(xc, yc) / den) / sd) if den > EPS else np.nan
    # band-pass momentum from reconstructing the mid-frequency D3 sub-band alone (detrended, denoised band momentum)
    zero = np.zeros_like(cA4)
    rec_in = [(zero, zero)] * LEVEL
    rec_in[1] = (zero, cp[1][1])                                   # keep only the D3 layer (coeffs[1])
    band = pywt.iswt(rec_in, WAVELET, norm=True)
    mid_band_mom = float(band[-1] - band[-1 - MID_LAG]) / sd

    return (lf_energy_ratio, hf_lf_ratio, mid_band_mom,
            spectral_entropy, hf_energy_ratio, lf_slope)


def _rolling_wavelet_loop(close):
    """[reference implementation] per-bar rolling wavelet factors (end-of-window scalar, anti-lookahead), clear but slow (~25s/47k bars).
    Production uses the vectorized _rolling_wavelet; this stays as a readable baseline + equivalence-test reference.
    Warmup (first WIN bars, needing WIN+1 prices to compute WIN returns) and in-window non-positive/gaps -> NaN. No pywt -> all NaN."""
    c = np.asarray(close, dtype=float)
    n = len(c)
    cols = {name: np.full(n, np.nan) for name in WAVELET_COLS}
    if pywt is None or n < WIN + 1:
        return cols
    logc = np.log(np.where(c > 0, c, np.nan))
    for t in range(WIN, n):                                        # needs logc[t-WIN .. t], WIN+1 values total
        seg = logc[t - WIN:t + 1]                                  # WIN+1 log prices, right end = t
        if seg.shape[0] != WIN + 1 or not np.all(np.isfinite(seg)):
            continue
        lr = np.diff(seg)                                         # WIN log returns, right end = t
        lp = seg[1:]                                              # WIN log prices, right end = t
        try:
            vals = _swt_readouts(lr, lp)
        except Exception:  # noqa: BLE001 a single-window failure does not affect the rest, leave NaN
            continue
        for name, v in zip(WAVELET_COLS, vals):
            cols[name][t] = v
    return cols


def _rolling_wavelet(close, chunk=8000):
    """[production implementation] vectorized rolling wavelet factors: sliding_window_view + batched SWT/iSWT (axis=1).
    Point-for-point equivalent to _rolling_wavelet_loop (tested), ~15x faster (~1-2s/47k bars). Chunked to control memory (default 8000 windows/chunk).
    Anti-lookahead unchanged: each factor[t] still uses only the window logc[t-WIN:t+1] (sliding_window_view introduces no future)."""
    c = np.asarray(close, dtype=float)
    n = len(c)
    cols = {name: np.full(n, np.nan) for name in WAVELET_COLS}
    if pywt is None or n < WIN + 1:
        return cols
    logc = np.log(np.where(c > 0, c, np.nan))                     # per-point log, introduces no lookahead
    nwin = n - WIN                                                # number of windows; window k -> bar t=k+WIN
    sw = np.lib.stride_tricks.sliding_window_view(logc, WIN + 1)  # (nwin, WIN+1) view, sw[k]=logc[k:k+WIN+1]
    xc = np.arange(SLOPE_K, dtype=float); xc = xc - xc.mean()
    den_slope = float(np.dot(xc, xc))
    for s in range(0, nwin, chunk):
        e = min(s + chunk, nwin)
        wins = sw[s:e]                                            # (m, WIN+1)
        finite = np.isfinite(wins).all(axis=1)                   # drop any window with a gap/non-positive value entirely
        if not finite.any():
            continue
        wf = wins[finite]                                        # (mf, WIN+1)
        lr = np.diff(wf, axis=1)                                 # (mf, WIN) log returns, right end = t
        lp = wf[:, 1:]                                           # (mf, WIN) log prices, right end = t
        # energy family (log-return SWT)
        cr = pywt.swt(lr, WAVELET, level=LEVEL, trim_approx=False, norm=True, axis=1)
        eA = np.sum(cr[0][0] ** 2, axis=1)
        eD4 = np.sum(cr[0][1] ** 2, axis=1); eD3 = np.sum(cr[1][1] ** 2, axis=1)
        eD2 = np.sum(cr[2][1] ** 2, axis=1); eD1 = np.sum(cr[3][1] ** 2, axis=1)
        total = eA + eD1 + eD2 + eD3 + eD4 + EPS
        lf_er = eA / total
        hf_lf = np.log((eD1 + eD2 + EPS) / (eA + EPS))
        hf_er = (eD1 + eD2) / total
        emat = np.stack([eA, eD4, eD3, eD2, eD1], axis=1)        # (mf,5)
        pm = emat / (emat.sum(axis=1, keepdims=True) + EPS)
        logpm = np.zeros_like(pm)
        np.log(pm, out=logpm, where=pm > 0)                      # log only where p>0, avoiding log(0) RuntimeWarning
        ent = -np.sum(pm * logpm, axis=1) / np.log(5)
        sd = np.std(lr, axis=1) + EPS                            # in-symbol volatility (return std, ddof=0)
        # price family (log-price SWT): A4 slope + D3 band-pass momentum
        cp = pywt.swt(lp, WAVELET, level=LEVEL, trim_approx=False, norm=True, axis=1)
        cA4 = cp[0][0]                                           # (mf, WIN)
        yk = cA4[:, -SLOPE_K:]
        ycm = yk - yk.mean(axis=1, keepdims=True)
        slope = (ycm * xc).sum(axis=1) / den_slope / sd
        zero = np.zeros_like(cA4)
        rec = [(zero, zero)] * LEVEL
        rec[1] = (zero, cp[1][1])                               # keep only D3
        band = np.asarray(pywt.iswt(rec, WAVELET, norm=True, axis=1))   # (mf, WIN)
        midmom = (band[:, -1] - band[:, -1 - MID_LAG]) / sd
        # scatter back to the global bar index t=k+WIN
        idx = np.arange(s, e)[finite] + WIN
        for name, vals in zip(WAVELET_COLS, [lf_er, hf_lf, midmom, ent, hf_er, slope]):
            cols[name][idx] = vals
    return cols


# ============ OU reversion pressure (pure pandas, right-aligned rolling is inherently lookahead-free) ============
def _ou_theta(close, win=OU_WIN):
    """Mean-reversion speed theta=-ln(1+b) of an AR(1) on the [rolling-mean-demeaned] log price (mu is the rolling
    window mean, not an in-window constant mean; this estimator's theta is still monotone in the true reversion speed
    phi, used for relative speed ranking, not as an absolute OU parameter).
    b∈(-1,0)->finite positive theta (mean reverting); b>=0 (trending)->clamp 0 (continuous mapping, never inject 9999
    to create a tied block); b<=-1 (oscillatory divergence)->NaN. Returns (theta:Series, logp:Series, mu:Series, sd:Series)
    so the two factors can share b.
    Honesty: under a small-sample Dickey-Fuller bias, theta is systematically overestimated and HL underestimated by ~44%
    (measured in this environment), so only relative speed is compared, not the absolute HL."""
    c = pd.Series(np.asarray(close, dtype=float))
    logp = np.log(c.where(c > 0))
    mp = max(8, int(win * 0.8))
    mu = logp.rolling(win, min_periods=mp).mean()
    sd = logp.rolling(win, min_periods=mp).std()
    y = logp - mu                                                 # rolling-mean demean (mu=rolling window mean, stationarizing)
    dy = y.diff()
    yl = y.shift(1)
    cov = dy.rolling(win, min_periods=mp).cov(yl)
    var = yl.rolling(win, min_periods=mp).var()
    b = cov / var.where(var > 1e-18)
    one_plus = 1.0 + b
    theta = pd.Series(np.where(one_plus > 0, -np.log(one_plus.clip(lower=1e-9)), np.nan),
                      index=c.index)
    theta = theta.where(b > -1.0)                                 # b<=-1 divergent end -> NaN (don't clamp to a fake super-fast reversion)
    theta = theta.clip(lower=0.0)                                 # b>=0 trending state -> 0 (continuous regime mapping)
    return theta, logp, mu, sd


def _ou_factors(close, win=OU_WIN):
    """Returns (ou_rev_press, ou_mr_speed) two ndarrays.
    ou_rev_press = -tanh(demeaned z-score) * sqrt(soft-standardized theta): expensive and fast-reverting -> expected
    pullback (negative); cheap and fast-reverting -> rebound (positive).
    ou_mr_speed = theta itself (regime-only, no direction, Rank IC expectation ≈0, a gate candidate not a ranking alpha)."""
    theta, logp, mu, sd = _ou_theta(close, win=win)
    mp = max(8, int(win * 0.8))
    dev_z = (logp - mu) / sd.where(sd > 1e-12)
    theta_med = theta.rolling(win, min_periods=mp).median()
    theta_norm = (theta / (theta_med + 1e-9)).clip(0, 5)          # soft standardization, trending state auto-mutes
    ou_rev_press = -np.tanh(dev_z) * np.sqrt(theta_norm)          # warmup/trend/gap all naturally NaN, no fillna
    return ou_rev_press.to_numpy(), theta.to_numpy()


# ============ public entry ============
def wavelet_ou_factors(df):
    """Compute all time-frequency / OU factors for a single-symbol OHLCV DataFrame. Returns DataFrame(index=df.index, 8 columns).
    Called by factor_eval.compute_factors; the produced columns automatically enter the Rank IC check. Anti-lookahead: all end-of-window / right-aligned rolling."""
    close = df["close"].astype(float).to_numpy()
    out = pd.DataFrame(index=df.index)
    wav = _rolling_wavelet(close)
    for name in WAVELET_COLS:
        out[name] = wav[name]
    rev_press, mr_speed = _ou_factors(close)
    out["ou_rev_press"] = rev_press
    out["ou_mr_speed"] = mr_speed
    return out


FACTOR_DESC = {
    "wav_lf_energy_ratio": "wavelet low-frequency A4 energy share (return domain; high share = returns dominated by persistence/drift, little high-freq noise)",
    "wav_hf_lf_ratio": "high-freq (D1+D2)/low-freq (A4) return-energy log ratio (oscillation vs trend, frequency-domain cross-check of hurst)",
    "wav_mid_band_mom": "mid-frequency D3 (~8-16h) band-pass momentum (log-price detrended denoised band momentum, return-volatility standardized)",
    "wav_spectral_entropy": "wavelet spectral entropy (return sub-band energy distribution Shannon entropy / ln5, high = no dominant period/chaotic, low = a clear rhythm)",
    "wav_hf_energy_ratio": "high-freq detail (D1+D2, ~2-8h) return-energy share (microstructure noise / wick intensity)",
    "wav_lf_slope": "wavelet A4 end-of-window regression slope (denoised trend slope; |ρ|≈0.8 redundant with wav_mid_band_mom, moved out of scoring as a diagnostic)",
    "ou_rev_press": "OU reversion pressure (-tanh(demeaned z)*sqrt(theta); expensive and fast-reverting -> expected down / cheap and fast-reverting -> expected up; main field h1/h4)",
    "ou_mr_speed": "OU mean-reversion speed theta (regime-only; measured h1 directional signal is actually a theta∝1/vol low-volatility proxy, homogeneous with vol_24h, moved out of scoring as a gate)",
}


if __name__ == "__main__":
    # self-test: run once on a real BTC 1h feather, inspect end-of-window values and warmup NaN
    import os
    data_dir = os.environ.get("ALPHAGAUNTLET_DATA_DIR", os.path.join(".", "data"))
    fn = os.path.join(data_dir, "BTC_USDT-1h.feather")
    d = pd.read_feather(fn).tail(1200).reset_index(drop=True)
    res = wavelet_ou_factors(d)
    print("pywt:", None if pywt is None else pywt.__version__, "| rows:", len(res))
    print("warmup NaN counts (should be ≈WIN-1=255 for wav, OU_WIN for ou):")
    print(res.isna().sum().to_string())
    print("\nlast 3 bars:")
    print(res.tail(3).to_string())
