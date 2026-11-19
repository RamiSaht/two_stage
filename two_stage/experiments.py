"""The training runs behind the stored results in results/stage2."""
import zlib

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import evaluation, metrics, stage2
from .stage2 import N_PASSES, Spec


def fit_case(model, config, case, repeat):
    frames, failures = stage2.load_frames(config['resample_freq'], config['stat_set'])
    train_events, test_events = stage2.cases(frames, failures)[case - 1]
    fitted = stage2.fit(Spec.of(model), frames, train_events, config['params'], int(config['window']),
                        config['epochs'], stage2.seeds(case - 1, repeat), config['stat_set'])
    return fitted, frames, train_events, test_events


def predict_event(fitted, frames, event, n_passes=N_PASSES):
    samples = stage2.windows(frames, event, fitted.columns, fitted.scaler, fitted.window)
    mean, _, draws = stage2.predict(fitted, samples, n_passes)
    return {'t': np.array([s[3] for s in samples]), 'y': np.array([s[1] for s in samples], dtype=np.float64),
            'mean': mean, 'draws': draws}


# ------------------------------------------------------------------------------------------
# Cases 1 to 9, five repeats per model
# ------------------------------------------------------------------------------------------

def case_results(tuned, path, models=stage2.MODELS, repeats=stage2.N_REPEATS, cases=range(1, stage2.N_CASES + 1)):
    """Scores of every model on every case and repeat, appended to `path` after each model. Models that
    are already in `path` are skipped, so an interrupted run continues where it stopped."""
    done = set(pd.read_csv(path)['model']) if path.exists() else set()
    jobs = [(model, repeat, case) for model in models if model not in done
            for repeat in range(repeats) for case in cases]
    rows = []
    for i, (model, repeat, case) in enumerate(tqdm(jobs)):
        config, spec = tuned[model], Spec.of(model)
        fitted, frames, _, test_events = fit_case(model, config, case, repeat)
        for event in test_events:
            p = predict_event(fitted, frames, event)
            rows.append({'model': model, 'backbone': spec.backbone, 'method': spec.method, 'repeat': repeat,
                         'case': case, 'event': event, 'component': frames[event][1]['component'],
                         'freq': config['resample_freq'], 'window': int(config['window']),
                         **metrics.score(p['y'], p['mean'], p['draws'])})
        if i + 1 == len(jobs) or jobs[i + 1][0] != model:
            pd.DataFrame(rows).to_csv(path, mode='a', header=not path.exists(), index=False)
            rows = []
    return pd.read_csv(path)


def case_predictions(tuned, results, case=5):
    """Rerun of the selected model of each group and the Constant baseline on one case, with the
    seeds of the repeat whose CRPS is the median of the five (Fig. 14)."""
    chosen = evaluation.select_models(evaluation.case_scores(results))
    arrays, rows = {}, []
    for group, model in list(chosen.items()) + [('baseline', 'constant')]:
        repeat = evaluation.median_repeat(results, model, case)
        fitted, frames, train_events, test_events = fit_case(model, tuned[model], case, repeat)
        for event in test_events + train_events:
            p = predict_event(fitted, frames, event)
            score = metrics.score(p['y'], p['mean'], p['draws'])
            rows.append({'group': group, 'model': model, 'repeat': repeat, 'event': event,
                         'held_out': event in test_events, 'n': len(p['y']),
                         **{k: score[k] for k in ('mae', 'crps', 'r2', 'cov95', 'width95')}})
            if event in test_events:
                key = f'{model}|{event}'
                arrays[f'{key}|t'] = p['t'].astype('datetime64[s]').astype(np.int64)
                arrays[f'{key}|y'] = p['y']
                arrays[f'{key}|mean'] = p['mean']
                arrays[f'{key}|pct'] = np.percentile(p['draws'], [2.5, 25, 50, 75, 97.5], axis=1)
    return arrays, pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------
# Monte Carlo dropout settings
# ------------------------------------------------------------------------------------------

PASS_GRID = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000]
ENSEMBLE_SIZES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
N_SUBSETS = 10


def _blocks(draws, n_passes):
    return [draws[:, m * n_passes:(m + 1) * n_passes] for m in range(draws.shape[1] // n_passes)]


def passes_per_network(tuned, models=stage2.NETWORKS, case=1, max_passes=1000, min_mean_rul=5.0):
    """CRPS and coverage of the ensemble when every network is run 5 to 1000 times."""
    rows = []
    for model in tqdm(models):
        fitted, frames, _, test_events = fit_case(model, tuned[model], case, 0)
        for event in test_events:
            p = predict_event(fitted, frames, event, max_passes)
            blocks = _blocks(p['draws'], max_passes)
            network = blocks[0]
            keep = network.mean(axis=1) > min_mean_rul
            cv_final = network[keep].std(axis=1, ddof=1) / network[keep].mean(axis=1)
            low_final, high_final = np.percentile(p['draws'], [2.5, 97.5], axis=1)
            for passes in PASS_GRID:
                part = network[:, :passes]
                cv = part[keep].std(axis=1, ddof=1) / part[keep].mean(axis=1)
                pooled = np.concatenate([b[:, :passes] for b in blocks], axis=1)
                low, high = np.percentile(pooled, [2.5, 97.5], axis=1)
                score = metrics.score(p['y'], pooled.mean(axis=1), pooled)
                rows.append({'model': model, 'freq': tuned[model]['resample_freq'], 'event': event,
                             'passes': passes, 'windows_used': int(keep.sum()), 'windows': len(keep),
                             'cv_median': float(np.median(cv)), 'cv_change_median': float(np.median(np.abs(cv / cv_final - 1))),
                             'width95_change_median': float(np.median(np.abs((high - low) / (high_final - low_final) - 1))),
                             'crps': score['crps'], 'cov95': score['cov95']})
    return pd.DataFrame(rows)


def _ensemble_metrics(y, draws):
    return {'crps': metrics.crps(draws, y), 'mae': float(np.abs(y - draws.mean(axis=1)).mean()),
            'cov50': metrics.coverage(y, draws, 0.5), 'cov95': metrics.coverage(y, draws, 0.95),
            'width95': metrics.mean_width(draws, 0.95)}


def ensemble_size(tuned, models=stage2.NETWORKS, cases=range(1, stage2.N_CASES + 1), repeats=stage2.N_REPEATS):
    """The 15 networks of the five repeats of a case, pooled into ensembles of 1 to 15 networks."""
    rows = []
    for model in tqdm(models):
        for case in cases:
            networks, truth = {}, {}
            for repeat in range(repeats):
                fitted, frames, _, test_events = fit_case(model, tuned[model], case, repeat)
                for event in test_events:
                    p = predict_event(fitted, frames, event)
                    truth[event] = p['y']
                    networks.setdefault(event, []).extend(_blocks(p['draws'], N_PASSES))
            rng = np.random.default_rng(zlib.crc32(f'{model}|{case - 1}'.encode()))
            for event, nets in networks.items():
                for size in (s for s in ENSEMBLE_SIZES if s <= len(nets)):
                    picks = ([np.arange(len(nets))] if size == len(nets) else
                             [rng.choice(len(nets), size, replace=False) for _ in range(N_SUBSETS)])
                    scores = [_ensemble_metrics(truth[event], np.concatenate([nets[i] for i in pick], axis=1))
                              for pick in picks]
                    rows.append({'model': model, 'case': case, 'event': event, 'networks': size, 'subsets': len(picks),
                                 **{k: float(np.mean([s[k] for s in scores])) for k in scores[0]}})
    return pd.DataFrame(rows)


def window_draws(tuned, model='cnn+separate', case=5, repeat=0):
    """The 300 draws of the window with the median spread between the three networks (Figs. 8 and 9)."""
    fitted, frames, _, test_events = fit_case(model, tuned[model], case, repeat)
    rows = []
    for event in test_events:
        samples = stage2.windows(frames, event, fitted.columns, fitted.scaler, fitted.window)
        mean, _, draws = stage2.predict(fitted, samples)
        spread = np.stack([b.mean(axis=1) for b in _blocks(draws, N_PASSES)], axis=1).std(axis=1)
        rows += [(spread[i], event, i, samples[i][0], float(samples[i][1]), float(mean[i]), draws[i])
                 for i in range(len(samples))]
    rows.sort(key=lambda row: row[0])
    spread, event, index, window, generated, point, draws = rows[len(rows) // 2]
    return {'draws': draws, 'window': window, 'generated': generated, 'point': max(point, 0.0), 'event': event,
            'window_index': index, 'spread': spread}
