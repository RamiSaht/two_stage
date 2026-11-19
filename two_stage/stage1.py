import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam

from . import config as cfg
from . import data

MI_TOP_K = 20
MI_SUBSAMPLE = 8000
WINDOW = 36
HIDDEN = 64
FC = 32
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
PATIENCE = 5
VAL_FRACTION = 0.2

RESIDUAL_PERCENTILES = (1, 99)
K_GRID = [0.25, 0.5, 1.0, 2.0]
P_GRID = [95, 99, 99.5, 99.9]
T_C = 72
CARE_BETA = 0.5

# 'transform' and not 'transformer': sensor_40 is described as "Transformator core temperature"
SAME_COMPONENT_KEYWORD = {'bearing': 'rotor bearing', 'transformer': 'transform'}
REFERENCE_CASES = {27, 77, 2, 21}    # the bearing cases evaluated by O'Connor et al. (2026)

# ------------------------------------------------------------------------------------------
# Runs
# ------------------------------------------------------------------------------------------

def runs(info):
    """One NBM per event and component. A failure gets a 'fault' run for its own component and a
    'cross' run for the other one; a healthy event gets a 'healthy' run for both."""
    out = []
    for _, row in info.iterrows():
        is_fault = row['event_label'] == 'anomaly'
        own = cfg.COMPONENT_OF[row['component']] if is_fault else None
        for component in cfg.COMPONENTS:
            out.append({'event': int(row['event_id']), 'component': component,
                        'role': ('fault' if component == own else 'cross') if is_fault else 'healthy',
                        'event_start': row['event_start'], 'event_end': row['event_end']})
    return out


# ------------------------------------------------------------------------------------------
# Normal behavior model
# ------------------------------------------------------------------------------------------

def candidate_features(features, component):
    keyword = SAME_COMPONENT_KEYWORD[component]
    return [f"{row['sensor_name']}_avg" for _, row in features.iterrows()
            if not row['is_angle'] and not row['is_counter']
            and keyword not in str(row['description']).lower()
            and 'average' in str(row['statistics_type'])]


def prepare(event_id, component, features):
    target = cfg.TARGET_OF[component]
    df = data.load_raw(event_id, fix_labels=False)
    candidates = [c for c in candidate_features(features, component) if c in df.columns and c != target]
    df = data.fill_short_gaps(df[['time_stamp', 'train_test', 'status_type_id', target] + candidates])
    train = df[df['train_test'] == 'train']
    prediction = df[df['train_test'] == 'prediction'].copy()
    train = train[train['status_type_id'] == cfg.NORMAL_STATUS].dropna(subset=[target] + candidates).copy()
    return train, prediction, candidates


def build_windows(df, features, target, scaler_x, scaler_y):
    """Windows of WINDOW consecutive 10-minute records, predicting the target at the window end."""
    x = scaler_x.transform(df[features].values).astype(np.float32)
    y = scaler_y.transform(df[[target]].values).astype(np.float32).ravel()
    t = df['time_stamp'].values
    idx = np.arange(WINDOW - 1, len(df))
    span_ok = (t[idx] - t[idx - WINDOW + 1]) == np.timedelta64(10 * (WINDOW - 1), 'm')
    row_ok = ~np.isnan(x).any(axis=1)
    window_ok = np.array([row_ok[i - WINDOW + 1:i + 1].all() for i in idx], dtype=bool)
    idx = idx[span_ok & window_ok & ~np.isnan(y[idx])]
    windows = (np.stack([x[i - WINDOW + 1:i + 1] for i in idx]) if len(idx)
               else np.empty((0, WINDOW, len(features)), np.float32))
    return torch.tensor(windows), torch.tensor(y[idx]), idx


class NBM(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(n_features, HIDDEN, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(HIDDEN, FC), nn.LeakyReLU(), nn.Linear(FC, 1))

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1, :]).squeeze(-1)


@torch.no_grad()
def predict(model, x, batch_size=2048):
    model.eval()
    out = [model(x[b:b + batch_size].to(cfg.DEVICE)).cpu().numpy() for b in range(0, len(x), batch_size)]
    return np.concatenate(out) if out else np.array([])


def train_nbm(x_train, y_train, x_val, y_val):
    model = NBM(x_train.shape[2]).to(cfg.DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    best_val, best_state, waited = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(x_train))
        for b in range(0, len(x_train), BATCH_SIZE):
            batch = order[b:b + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[batch].to(cfg.DEVICE)), y_train[batch].to(cfg.DEVICE))
            loss.backward()
            optimizer.step()
        val_mse = float(np.mean((predict(model, x_val) - y_val.numpy()) ** 2))
        if val_mse < best_val:
            best_val, waited = val_mse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_val


def train_case(event_id, component, features):
    """Train one NBM and return its residuals over the training and prediction periods."""
    target = cfg.TARGET_OF[component]
    train, prediction, candidates = prepare(event_id, component, features)
    sample = train.sample(n=min(MI_SUBSAMPLE, len(train)), random_state=cfg.SEED)
    mi = mutual_info_regression(sample[candidates].values, sample[target].values, random_state=cfg.SEED)
    selected = [candidates[i] for i in np.argsort(mi)[::-1][:MI_TOP_K]]

    scaler_x = StandardScaler().fit(train[selected].values)
    scaler_y = StandardScaler().fit(train[[target]].values)
    x_all, y_all, _ = build_windows(train, selected, target, scaler_x, scaler_y)
    n_val = int(len(x_all) * VAL_FRACTION)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    model, val_mse = train_nbm(x_all[:-n_val], y_all[:-n_val], x_all[-n_val:], y_all[-n_val:])

    y_val = scaler_y.inverse_transform(y_all[-n_val:].numpy().reshape(-1, 1)).ravel()
    y_val_hat = scaler_y.inverse_transform(predict(model, x_all[-n_val:]).reshape(-1, 1)).ravel()
    val_mape = float(np.mean(np.abs((y_val - y_val_hat) / np.clip(np.abs(y_val), 1e-6, None))) * 100)

    parts = []
    for period, df in (('train', train), ('prediction', prediction)):
        x, y, idx = build_windows(df, selected, target, scaler_x, scaler_y)
        if len(x):
            parts.append(pd.DataFrame({'time_stamp': df['time_stamp'].values[idx], 'period': period,
                                       'status_type_id': df['status_type_id'].values[idx],
                                       'y': y.numpy(), 'y_hat': predict(model, x)}))
    residuals = pd.concat(parts, ignore_index=True)
    residuals['residual'] = residuals['y'] - residuals['y_hat']
    meta = {'event': event_id, 'component': component, 'target': target, 'selected_features': selected,
            'val_mse': val_mse, 'val_mape': val_mape,
            'n_train_windows': int(len(x_all) - n_val), 'n_val_windows': int(n_val)}
    return residuals, meta


def save_case(residuals, meta):
    residuals.to_csv(cfg.output_path('stage1', 'residuals', f"{meta['event']}_{meta['component']}.csv.gz"),
                     index=False)
    path = cfg.output_path('stage1', 'nbm_runs.json')
    all_meta = json.loads(path.read_text()) if path.exists() else {}
    all_meta[f"{meta['event']}_{meta['component']}"] = meta
    path.write_text(json.dumps(all_meta, indent=1))


def load_residuals(event_id, component):
    return pd.read_csv(cfg.result_path('stage1', 'residuals', f'{event_id}_{component}.csv.gz'),
                       parse_dates=['time_stamp'])


def load_all_residuals(stage1_runs):
    return {(r['event'], r['component']): load_residuals(r['event'], r['component']) for r in stage1_runs}


# ------------------------------------------------------------------------------------------
# CUSUM detector
# ------------------------------------------------------------------------------------------

def standardization(residuals):
    """Median and MAD of the training residuals within the 1st to 99th percentile."""
    train = residuals.loc[residuals['period'] == 'train', 'residual'].dropna()
    low, high = np.percentile(train, RESIDUAL_PERCENTILES)
    train = train[(train >= low) & (train <= high)]
    median = train.median()
    mad = 1.4826 * np.median(np.abs(train - median))
    if mad == 0:
        mad = train.std() or 1.0
    return train, median, mad


def cusum(standardized, k):
    """Two-sided CUSUM. A missing residual leaves both sums unchanged."""
    s_pos, s_neg = 0.0, 0.0
    out = np.empty(len(standardized))
    for i, value in enumerate(standardized):
        if not np.isnan(value):
            s_pos = max(0.0, s_pos + value - k)
            s_neg = max(0.0, s_neg - value - k)
        out[i] = max(s_pos, s_neg)
    return out


def detect(residuals, k, p):
    """CUSUM score, threshold and flag of every record in the prediction period."""
    train, median, mad = standardization(residuals)
    threshold = np.percentile(cusum(((train - median) / mad).values, k), p)
    prediction = residuals[residuals['period'] == 'prediction'].copy()
    prediction['standardized'] = (prediction['residual'] - median) / mad
    prediction['score'] = cusum(prediction['standardized'].values, k)
    prediction['flag'] = (prediction['score'] > threshold).astype(int)
    prediction['threshold'] = threshold
    return prediction


def criticality(flags):
    """Counter that rises with every flagged record and falls with every other one; an alarm is
    raised when it reaches T_C. Returns the counter and the index of the first alarm (or None)."""
    counter, values, first = 0, np.empty(len(flags), dtype=int), None
    for i, flag in enumerate(flags):
        counter = counter + 1 if flag else max(counter - 1, 0)
        values[i] = counter
        if counter >= T_C and first is None:
            first = i
    return values, first


def evaluated(detection):
    """Only records with a normal status are evaluated."""
    return detection[detection['status_type_id'] == cfg.NORMAL_STATUS]


def alarm_profile(detection):
    """First alarm, how long the alarm is held, and the length of the period, in operating days."""
    part = evaluated(detection)
    flags = part['flag'].values
    times = part['time_stamp']
    step_days = ((times.shift(-1) - times).dt.total_seconds().clip(upper=600).fillna(0) / 86400).values
    counter, first = criticality(flags)
    active = counter >= T_C
    span = float(step_days.sum())
    return {'first_alarm': times.iloc[first] if first is not None else None,
            'alarm_days': float(step_days[active].sum()),
            'flagged_days': float(step_days[flags.astype(bool)].sum()),
            'span_days': span, 'flagged_fraction': float(flags.mean()) if len(flags) else 0.0}


# ------------------------------------------------------------------------------------------
# CARE score
# ------------------------------------------------------------------------------------------

def f_beta(tp, fp, fn, beta=CARE_BETA):
    denominator = (1 + beta ** 2) * tp + beta ** 2 * fn + fp
    return (1 + beta ** 2) * tp / denominator if denominator > 0 else 0.0


def care(component, k, p, stage1_runs, residuals, events=None):
    """Coverage, accuracy, reliability, earliness and the unified CARE score of one detector."""
    coverage, accuracy, earliness, truth, alarmed = [], [], [], [], []
    for run in stage1_runs:
        if run['component'] != component or run['role'] == 'cross':
            continue
        if events is not None and run['event'] not in events:
            continue
        part = evaluated(detect(residuals[(run['event'], component)], k, p))
        flags, times = part['flag'].values, part['time_stamp']
        truth.append(int(run['role'] == 'fault'))
        alarmed.append(int(criticality(flags)[1] is not None))
        if run['role'] == 'fault':
            coverage.append(f_beta(tp=flags.sum(), fp=0, fn=(1 - flags).sum()))
            inside = flags[((times >= run['event_start']) & (times <= run['event_end'])).values]
            if len(inside):
                position = np.linspace(0, 1, len(inside))
                weight = np.where(position <= 0.5, 1.0, 2 * (1 - position))
                earliness.append((weight * inside).sum() / weight.sum())
            else:
                earliness.append(0.0)
        else:
            accuracy.append(1 - flags.mean() if len(flags) else 1.0)

    c, a, e = np.mean(coverage), np.mean(accuracy), np.mean(earliness)
    tp = sum(t & q for t, q in zip(truth, alarmed))
    fp = sum((1 - t) & q for t, q in zip(truth, alarmed))
    fn = sum(t & (1 - q) for t, q in zip(truth, alarmed))
    r = f_beta(tp, fp, fn)
    if sum(coverage) == 0:
        score = 0.0
    elif a < 0.5:
        score = a
    else:
        score = (c + e + r + 2 * a) / 5
    return {'Coverage': c, 'Accuracy': a, 'Reliability': r, 'Earliness': e, 'CARE': score}


def settings_grid(stage1_runs, residuals, components=cfg.COMPONENTS, events=None):
    rows = [{'component': component, 'k': k, 'p': p, **care(component, k, p, stage1_runs, residuals, events)}
            for component in components for k in K_GRID for p in P_GRID]
    return pd.DataFrame(rows)


def choose_settings(grid):
    """(k, p) with the highest CARE score per component."""
    best = grid.sort_values('CARE', ascending=False, kind='stable').groupby('component').head(1)
    return {row['component']: (row['k'], row['p']) for _, row in best.iterrows()}


# ------------------------------------------------------------------------------------------
# Results per event
# ------------------------------------------------------------------------------------------

def event_results(info, residuals, settings):
    """First alarm and alarm duration of both detectors for every event, and the attribution."""
    rows = []
    for _, event in info.iterrows():
        event_id = int(event['event_id'])
        is_fault = event['event_label'] == 'anomaly'
        row = {'event': event_id, 'true_component': cfg.COMPONENT_OF[event['component']] if is_fault else None}
        first = {}
        for component in cfg.COMPONENTS:
            profile = alarm_profile(detect(residuals[(event_id, component)], *settings[component]))
            row[f'{component}_first_alarm'] = profile['first_alarm']
            alarmed = profile['first_alarm'] is not None
            row[f'{component}_alarm_days'] = profile['alarm_days'] if alarmed else np.nan
            row[f'{component}_flagged_days'] = profile['flagged_days'] if alarmed else np.nan
            row[f'{component}_flagged_fraction'] = profile['flagged_fraction']
            row['span_days'] = profile['span_days']
            if alarmed:
                first[component] = profile['first_alarm']
        row['attributed'] = min(first, key=first.get) if first else None
        rows.append(row)
    return pd.DataFrame(rows)


def residual_offsets(info, residuals, settings):
    """Mean residual of every detector over the prediction period, in MADs of its training residuals."""
    rows = []
    for _, event in info.iterrows():
        for component in cfg.COMPONENTS:
            res = residuals[(int(event['event_id']), component)]
            train = res.loc[res['period'] == 'train', 'residual'].dropna()
            median = train.median()
            mad = 1.4826 * np.median(np.abs(train - median))
            prediction = res.loc[res['period'] == 'prediction', 'residual']
            detection = detect(res, *settings[component])
            part = evaluated(detection)
            rows.append({'event': int(event['event_id']), 'component': component,
                         'healthy': event['event_label'] == 'normal',
                         'offset': float(((prediction - median) / mad).mean()),
                         'threshold': float(detection['threshold'].iloc[0]),
                         'alarm': criticality(part['flag'].values)[1] is not None,
                         'flagged_fraction': float(part['flag'].mean())})
    return pd.DataFrame(rows)
