import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from . import config as cfg
from .stage2 import BACKBONES

TUNING_CASE = 1
LEAKING_CASES = [2, 3, 4, 7]
INDEPENDENT_CASES = [5, 6, 8, 9]
CASE_FAILURES = {1: (34, 53), 2: (34, 27), 3: (34, 77), 4: (7, 53), 5: (7, 27), 6: (7, 77),
                 7: (19, 53), 8: (19, 27), 9: (19, 77)}
METRICS = ['mae', 'r2', 'crps', 'cov95', 'width95', 'RS_total', 'RS_under']
COMPONENT_AWARE = ('embed', 'head_linear', 'head_mlp', 'separate')
GROUPS = {
    'specialized': [f'{b}+{m}' for m in ('embed', 'head_linear', 'head_mlp') for b in BACKBONES],
    'independent': [f'{b}+separate' for b in BACKBONES],
    'system': [f'{b}+none' for b in BACKBONES],
    'baseline': ['constant'],
}
BACKBONE_NAME = {'cnn': 'CNN', 'lstm': 'LSTM', 'cnn_lstm': 'CNN-LSTM'}
METHOD_NAME = {'embed': 'with embedding', 'head_linear': 'with linear heads', 'head_mlp': 'with MLP heads'}


def display_name(model):
    if model == 'constant':
        return 'Constant'
    backbone, method = model.split('+')
    if method == 'separate':
        return f'Independent {BACKBONE_NAME[backbone]} models'
    if method == 'none':
        return f'System-level {BACKBONE_NAME[backbone]}'
    return f'{BACKBONE_NAME[backbone]} {METHOD_NAME[method]}'


def group_of(model):
    return next(group for group, members in GROUPS.items() if model in members)


def load_tuned():
    return json.loads((cfg.result_path('stage2', 'tuned_hyperparameters.json')).read_text())


def load_case_results():
    return pd.read_csv(cfg.result_path('stage2', 'case_results.csv'))


def case_scores(results, cases=INDEPENDENT_CASES, metrics=METRICS):
    """One row per model and case: the two held-out failures pooled by their number of windows,
    then averaged over the five repeats."""
    results = results[results['case'].isin(cases)]
    rows = []
    for (model, case), group in results.groupby(['model', 'case']):
        row = {'model': model, 'case': case}
        for metric in metrics:
            row[metric] = float(np.mean([np.average(g[metric], weights=g['n']) for _, g in group.groupby('repeat')]))
        rows.append(row)
    return pd.DataFrame(rows)


def failure_scores(results, cases=INDEPENDENT_CASES, metrics=METRICS):
    """One row per model, case and held-out failure, averaged over the five repeats."""
    results = results[results['case'].isin(cases)]
    return results.groupby(['model', 'case', 'event', 'component'])[metrics].mean().reset_index()


def model_means(scores):
    """Mean of the case (or failure) scores per model, in the order of the paper."""
    order = [m for members in GROUPS.values() for m in members]
    return scores.groupby('model')[[c for c in METRICS if c in scores]].mean().loc[order]


def select_models(scores):
    """The model with the lowest mean CRPS in each group of networks."""
    means = scores.groupby('model')['crps'].mean()
    return {group: means[members].idxmin() for group, members in GROUPS.items() if group != 'baseline'}


def typical_case(scores, models):
    """The case on which the MAE of `models`, averaged, lies closest to its mean over all cases."""
    table = scores[scores['model'].isin(models)].groupby('case')['mae'].mean()
    return int((table - table.mean()).abs().idxmin())


def median_repeat(results, model, case):
    """The repeat whose CRPS on the case (failures pooled by window count) is the median of the five."""
    rows = results[(results['model'] == model) & (results['case'] == case)]
    per_repeat = rows.groupby('repeat').apply(lambda g: np.average(g['crps'], weights=g['n']),
                                              include_groups=False).sort_values()
    return int(per_repeat.index[len(per_repeat) // 2])


def t_test_pairs():
    """The model pairs of the three questions in the methodology."""
    pairs = []
    for backbone in BACKBONES:
        for method in COMPONENT_AWARE:
            pairs.append(('Component information', f'{backbone}+{method}', f'{backbone}+none'))
        for a, b in itertools.combinations(COMPONENT_AWARE, 2):
            pairs.append(('Component-specific parts', f'{backbone}+{a}', f'{backbone}+{b}'))
    for method in COMPONENT_AWARE + ('none',):
        for a, b in itertools.combinations(BACKBONES, 2):
            pairs.append(('Backbone', f'{a}+{method}', f'{b}+{method}'))
    return pairs


def t_tests(scores, metric='mae', alpha=0.05):
    """Paired t-test on the per-case scores of each pair."""
    table = scores.pivot(index='case', columns='model', values=metric)
    rows = []
    for question, a, b in t_test_pairs():
        result = ttest_rel(table[a], table[b])
        rows.append({'question': question, 'model_a': a, 'model_b': b,
                     f'{metric}_a': table[a].mean(), f'{metric}_b': table[b].mean(),
                     'difference': table[a].mean() - table[b].mean(),
                     't': result.statistic, 'p': result.pvalue, 'significant': result.pvalue < alpha})
    return pd.DataFrame(rows)


def t_tests_against_constant(scores, metric='mae', alpha=0.05):
    table = scores.pivot(index='case', columns='model', values=metric)
    rows = []
    for model in [m for members in list(GROUPS.values())[:-1] for m in members]:
        result = ttest_rel(table[model], table['constant'])
        rows.append({'model': model, metric: table[model].mean(), 'constant': table['constant'].mean(),
                     'better': table[model].mean() < table['constant'].mean(),
                     't': result.statistic, 'p': result.pvalue, 'significant': result.pvalue < alpha})
    return pd.DataFrame(rows)


def leak_comparison(results, metric='mae'):
    """Mean score on the leaking and on the independent cases, relative to the Constant baseline."""
    out = {}
    for name, cases in (('independent', INDEPENDENT_CASES), ('leaking', LEAKING_CASES)):
        means = case_scores(results, cases, [metric]).groupby('model')[metric].mean()
        out[name] = means
        out[f'{name}_vs_constant'] = means - means['constant']
    table = pd.DataFrame(out)
    table['rank_independent'] = table['independent'].rank()
    table['rank_leaking'] = table['leaking'].rank()
    return table


# ------------------------------------------------------------------------------------------
# Monte Carlo dropout settings
# ------------------------------------------------------------------------------------------

def load_passes():
    return pd.read_csv(cfg.result_path('stage2', 'mc_passes.csv'))


def passes_at(passes, n=100):
    """Change of the CRPS and coverage at `n` passes relative to the largest pass count, per
    model and event."""
    rows = []
    for (model, event), part in passes.groupby(['model', 'event']):
        part = part.set_index('passes').sort_index()
        final = part.loc[part.index.max()]
        rows.append({'model': model, 'event': event,
                     'crps_change_pct': 100 * (part.loc[n, 'crps'] / final['crps'] - 1),
                     'cov95_change': part.loc[n, 'cov95'] - final['cov95']})
    return pd.DataFrame(rows)


def load_ensemble_size():
    return pd.read_csv(cfg.result_path('stage2', 'mc_ensemble_size.csv'))


def ensemble_size_summary(ensemble, metrics=('crps', 'mae', 'cov50', 'cov95', 'width95')):
    """Scores per model and ensemble size: the two failures of a case averaged, then the cases."""
    return (ensemble.groupby(['model', 'networks', 'case'])[list(metrics)].mean()
            .groupby(['model', 'networks']).mean())
