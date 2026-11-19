"""Probabilistic RUL metrics as defined by de Pater and Mitici (2022), "Novel metrics to evaluate
probabilistic remaining useful life prognostics with applications to turbofan engines",
PHM Society European Conference 7(1), 96-109."""
import numpy as np

from .config import ALPHA_GRID


def crps(draws, y_true):
    """CRPS of the empirical distribution of `draws` (N, M) against `y_true` (N,), in days.

    The empirical CDF is a step function, so the integral is a sum over the intervals between
    sorted draws, each split at the true value, plus the two tails.
    """
    x = np.sort(np.asarray(draws, dtype=np.float64), axis=1)
    y = np.asarray(y_true, dtype=np.float64)[:, None]
    m = x.shape[1]
    total = np.clip(y[:, 0] - x[:, -1], 0, None) + np.clip(x[:, 0] - y[:, 0], 0, None)
    if m > 1:
        low, high = x[:, :-1], x[:, 1:]
        cdf = np.arange(1, m, dtype=np.float64) / m
        below = np.clip(np.minimum(high, y) - low, 0, None)
        above = np.clip(high - np.maximum(low, y), 0, None)
        total = total + (cdf ** 2 * below + (cdf - 1.0) ** 2 * above).sum(axis=1)
    return float(total.mean())


def credible_bounds(draws, alphas):
    """Central credible intervals of width alpha: the (0.5 - alpha/2) and (0.5 + alpha/2) quantiles."""
    alphas = np.atleast_1d(np.asarray(alphas, dtype=np.float64))
    draws = np.asarray(draws, dtype=np.float64)
    low = np.quantile(draws, np.clip(0.5 - 0.5 * alphas, 0.0, 1.0), axis=1)
    high = np.quantile(draws, np.clip(0.5 + 0.5 * alphas, 0.0, 1.0), axis=1)
    return low, high


def coverage(y_true, draws, alphas=ALPHA_GRID):
    """Share of true RULs inside the alpha credible interval."""
    scalar = np.isscalar(alphas)
    low, high = credible_bounds(draws, alphas)
    y = np.asarray(y_true, dtype=np.float64)[None, :]
    values = ((y >= low) & (y <= high)).mean(axis=1)
    return float(values[0]) if scalar else values


def mean_width(draws, alphas=ALPHA_GRID):
    """Sharpness: the mean width of the alpha credible intervals."""
    scalar = np.isscalar(alphas)
    low, high = credible_bounds(draws, alphas)
    values = (high - low).mean(axis=1)
    return float(values[0]) if scalar else values


def reliability_score(alphas, alpha_coverage):
    """Areas between the reliability curve and the diagonal: below it the intervals are too
    narrow (RS_under), above it too wide (RS_over)."""
    gap = np.asarray(alpha_coverage, dtype=np.float64) - np.asarray(alphas, dtype=np.float64)
    under = float(np.trapezoid(np.where(gap <= 0.0, -gap, 0.0), alphas))
    over = float(np.trapezoid(np.where(gap > 0.0, gap, 0.0), alphas))
    return {'RS_under': under, 'RS_over': over, 'RS_total': under + over}


def score(y_true, mean, draws):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(mean, 0.0, None)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {'n': len(y_true),
            'crps': crps(draws, y_true),
            'mae': float(np.abs(y_true - y_pred).mean()),
            'rmse': float(np.sqrt(((y_true - y_pred) ** 2).mean())),
            'r2': 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            'cov50': coverage(y_true, draws, 0.5),
            'cov95': coverage(y_true, draws, 0.95),
            'width95': mean_width(draws, 0.95),
            **reliability_score(ALPHA_GRID, coverage(y_true, draws, ALPHA_GRID))}
