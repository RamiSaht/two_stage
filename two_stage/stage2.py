from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam

from . import config as cfg
from . import data, stage1

N_NETWORKS = 3
N_PASSES = 100
N_REPEATS = 5
N_CASES = 9
GAP_FILL_DAYS = 3.0

PHYSICAL_SENSORS = ['sensor_22', 'sensor_31', 'sensor_40', 'sensor_41', 'sensor_42', 'sensor_47',
                    'sensor_51', 'sensor_52', 'sensor_19', 'sensor_25', 'sensor_26', 'wind_speed_61',
                    'power_62']
STAT_SETS = {'avg': ('avg',), 'avg_std': ('avg', 'std'), 'all': ('avg', 'min', 'max', 'std')}
STAT_AGGREGATION = {'avg': 'mean', 'max': 'max', 'min': 'min', 'std': 'mean'}

BACKBONES = ('cnn', 'lstm', 'cnn_lstm')
METHODS = ('embed', 'head_linear', 'head_mlp', 'separate', 'none')
NETWORKS = [f'{backbone}+{method}' for backbone in BACKBONES for method in METHODS]
MODELS = NETWORKS + ['constant']


def n_bins(days, freq):
    return int(round(pd.Timedelta(days=days) / pd.Timedelta(freq)))


# ------------------------------------------------------------------------------------------
# Data: one frame per failure at the bin size of a model
# ------------------------------------------------------------------------------------------

def candidates(features, stats):
    return [f"{row['sensor_name']}_{stat}" for _, row in features.iterrows()
            if not row['is_angle'] and not row['is_counter']
            and all(data.STAT_DESCRIPTOR[s] in str(row['statistics_type']) for s in stats)
            for stat in stats]


def residual_series(event_id, component, freq):
    """Stage 1 residual in MADs of its own training residuals, averaged per bin."""
    res = stage1.load_residuals(event_id, component)
    train = res.loc[res['period'] == 'train', 'residual'].dropna()
    median = train.median()
    mad = 1.4826 * np.median(np.abs(train - median)) or 1.0
    return ((res.set_index('time_stamp')['residual'] - median) / mad).resample(freq).mean()


def build_frame(event_id, info, columns, freq):
    event = info[info['event_id'] == event_id].iloc[0]
    df = data.load_raw(event_id)
    keep = ['time_stamp', 'train_test', 'status_type_id'] + [c for c in columns if c in df.columns]
    df = data.fill_short_gaps(df[keep])
    features = [c for c in keep if data.stat_of(c)]

    # ---- downtime filter and RUL label
    df = df[df['status_type_id'] == cfg.NORMAL_STATUS].dropna(subset=features)
    df = data.add_operating_rul(df, event['event_end'])

    # ---- normalization by the median and IQR of the case's own training period
    train = df[df['train_test'] == 'train'][features]
    iqr = (train.quantile(0.75) - train.quantile(0.25)).replace(0, 1.0)
    df[features] = (df[features] - train.median()) / iqr

    # ---- resampling, and bridging of empty bins of at most three days
    indexed = df.set_index('time_stamp')
    frame = indexed[features].resample(freq).agg({c: STAT_AGGREGATION[data.stat_of(c)] for c in features})[features]
    frame['rul_days'] = indexed['rul_days'].resample(freq).mean()
    frame['is_pred'] = (indexed['train_test'] == 'prediction').astype(float).resample(freq).max()
    gap = n_bins(GAP_FILL_DAYS, freq)
    frame[features] = frame[features].ffill(limit=gap)
    frame['rul_days'] = frame['rul_days'].ffill(limit=gap)
    frame['is_pred'] = frame['is_pred'].ffill(limit=gap).fillna(0.0)

    residuals = pd.DataFrame({f'residual_{c}': residual_series(event_id, c, freq).reindex(frame.index)
                              .ffill(limit=3).fillna(0.0) for c in cfg.COMPONENTS}, index=frame.index)
    frame = pd.concat([frame, residuals], axis=1).dropna(subset=features)
    return frame, {'event': event_id, 'component': cfg.COMPONENT_OF[event['component']], 'freq': freq}


def load_frames(freq, stat_set='avg_std'):
    """Frames of the six failures and the order in which they appear in the dataset."""
    return _load_frames(freq, stat_set, cfg.active_results())


@lru_cache(maxsize=None)
def _load_frames(freq, stat_set, results):
    info = data.load_info()
    columns = candidates(data.load_feature_description(), STAT_SETS[stat_set])
    failures = [int(e) for e in info[info['event_label'] == 'anomaly']['event_id']]
    return {e: build_frame(e, info, columns, freq) for e in failures}, failures


def cases(frames, failures):
    """Leave one pair out: every combination of one transformer and one bearing failure is held out
    once, giving Cases 1 to 9 (T1-B1, T1-B2, ..., T3-B3)."""
    transformers = [e for e in failures if frames[e][1]['component'] == 'transformer']
    bearings = [e for e in failures if frames[e][1]['component'] == 'bearing']
    return [([e for e in failures if e not in (t, b)], [t, b]) for t in transformers for b in bearings]


def input_columns(frames, train_events, stat_set='avg_std'):
    present = set.intersection(*(set(frames[e][0].columns) for e in train_events))
    return [f'{s}_{stat}' for s in PHYSICAL_SENSORS for stat in STAT_SETS[stat_set] if f'{s}_{stat}' in present]


def fit_scaler(frames, train_events, columns):
    return StandardScaler().fit(np.concatenate([frames[e][0][columns].values.astype(np.float32)
                                                for e in train_events]))


def windows(frames, event_id, columns, scaler, window, require_target=True):
    """One sample per bin of the prediction period: the `window` bins up to and including it,
    without gaps. A sample is (inputs, RUL at the window end, component index, time)."""
    frame, meta = frames[event_id]
    x = scaler.transform(frame[columns].values.astype(np.float32)).astype(np.float32)
    x = np.hstack([x, frame[[f'residual_{c}' for c in cfg.COMPONENTS]].values.astype(np.float32)])
    t, rul, is_pred = frame.index.values, frame['rul_days'].values, frame['is_pred'].values
    step = pd.Timedelta(meta['freq']).to_timedelta64()
    component = cfg.COMPONENT_INDEX[meta['component']]
    samples = []
    for i in range(window - 1, len(frame)):
        if (t[i] - t[i - window + 1]) != step * (window - 1) or not is_pred[i]:
            continue
        if require_target and rul[i] != rul[i]:
            continue
        samples.append((x[i - window + 1:i + 1], np.float32(rul[i]), component, t[i]))
    return samples


# ------------------------------------------------------------------------------------------
# Networks
# ------------------------------------------------------------------------------------------

class LSTMBackbone(nn.Module):
    def __init__(self, n_inputs, params, window):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, params['LSTM_hidden_size'], params['LSTM_num_layers'], batch_first=True,
                            dropout=0.2 if params['LSTM_num_layers'] > 1 else 0.0)
        self.drop = nn.Dropout(params.get('BACKBONE_DROPOUT', params['CNN_DROPOUT']))
        self.out_dim = params['LSTM_hidden_size']

    def forward(self, x):
        return self.drop(self.lstm(x)[0][:, -1, :])


def _conv_layers(n_inputs, params):
    layers, channels = [], n_inputs
    for out in params['CNN_out_channels']:
        layers += [nn.Conv1d(channels, out, params['CNN_kernel_size'], padding='same'),
                   nn.LeakyReLU(), nn.Dropout(params['CNN_DROPOUT'])]
        channels = out
    return nn.Sequential(*layers), channels


class CNNBackbone(nn.Module):
    def __init__(self, n_inputs, params, window):
        super().__init__()
        self.cnn, channels = _conv_layers(n_inputs, params)
        self.pool = params.get('CNN_POOL', 'flatten')
        self.out_dim = channels * window if self.pool == 'flatten' else channels

    def forward(self, x):
        h = self.cnn(x.permute(0, 2, 1))
        if self.pool == 'flatten':
            return h.flatten(1)
        if self.pool == 'last':
            return h[:, :, -1]
        return h.mean(dim=2)


class CNNLSTMBackbone(nn.Module):
    def __init__(self, n_inputs, params, window):
        super().__init__()
        self.cnn, channels = _conv_layers(n_inputs, params)
        self.lstm = nn.LSTM(channels, params['LSTM_hidden_size'], params['LSTM_num_layers'], batch_first=True,
                            dropout=0.2 if params['LSTM_num_layers'] > 1 else 0.0)
        self.out_dim = params['LSTM_hidden_size']

    def forward(self, x):
        return self.lstm(self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1))[0][:, -1, :]


BACKBONE_CLASSES = {'lstm': LSTMBackbone, 'cnn': CNNBackbone, 'cnn_lstm': CNNLSTMBackbone}


def _mlp_head(params):
    return nn.Sequential(nn.Linear(params['SHARED_FC_SIZE'], params['HEAD_NEURONS']), nn.LeakyReLU(),
                         nn.Dropout(params['HEAD_DROPOUT']), nn.Linear(params['HEAD_NEURONS'], 1))


class Stage2Net(nn.Module):
    """Backbone, shared dense layer and head(s). 'embed' concatenates a learned component vector
    to the backbone output, 'head_linear' and 'head_mlp' have one head per component. 'separate'
    and 'none' use a single head; 'separate' trains one such network per component."""

    def __init__(self, backbone, method, n_inputs, params, window):
        super().__init__()
        self.method = 'none' if method == 'separate' else method
        self.backbone = BACKBONE_CLASSES[backbone](n_inputs, params, window)
        width = self.backbone.out_dim
        if self.method == 'embed':
            self.embedding = nn.Embedding(len(cfg.COMPONENTS), params['EMB_DIM'])
            width += params['EMB_DIM']
        self.shared_fc = nn.Linear(width, params['SHARED_FC_SIZE'])
        self.fc_drop = nn.Dropout(params['FC_DROPOUT'])
        if self.method == 'head_mlp':
            self.heads = nn.ModuleList([_mlp_head(params) for _ in cfg.COMPONENTS])
        elif self.method == 'head_linear':
            self.heads = nn.ModuleList([nn.Linear(params['SHARED_FC_SIZE'], 1) for _ in cfg.COMPONENTS])
        else:
            self.heads = nn.ModuleList([_mlp_head(params)])

    def forward(self, x, component):
        z = self.backbone(x)
        if self.method == 'embed':
            z = torch.cat([z, self.embedding(component)], dim=1)
        z = self.fc_drop(F.leaky_relu(self.shared_fc(z)))
        if len(self.heads) == 1:
            return self.heads[0](z).squeeze(-1)
        outs = torch.stack([head(z) for head in self.heads], dim=1)
        return outs.gather(1, component.view(-1, 1, 1)).flatten()


# ------------------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------------------

@dataclass
class Spec:
    backbone: str
    method: str

    @classmethod
    def of(cls, model):
        return cls('-', model) if model == 'constant' else cls(*model.split('+'))

    @property
    def name(self):
        return self.method if self.method == 'constant' else f'{self.backbone}+{self.method}'


@dataclass
class Fitted:
    spec: Spec
    columns: list
    scaler: object
    window: int
    trainer: object = None
    train_rul: np.ndarray = None
    extra: dict = field(default_factory=dict)


def seeds(case_index, repeat):
    """Three seeds per case and repeat: case 0-8 of repeat 0 use seeds 42-68, repeat 1 69-95, ..."""
    block = case_index + repeat * N_CASES
    return [cfg.SEED + block * N_NETWORKS + k for k in range(N_NETWORKS)]


class Trainer:
    """The networks of one ensemble (two sets for 'separate'), trained on the windows of the
    training failures. The target is standardized with the mean and std of all training RULs."""

    def __init__(self, spec, frames, train_events, columns, scaler, params, window, member_seeds):
        self.spec, self.params, self.window, self.seeds = spec, params, window, list(member_seeds)
        if spec.method == 'separate':
            split = {cfg.COMPONENT_INDEX[c]: [e for e in train_events if frames[e][1]['component'] == c]
                     for c in cfg.COMPONENTS}
        else:
            split = {None: list(train_events)}
        grouped = {key: [s for e in events for s in windows(frames, e, columns, scaler, window)]
                   for key, events in split.items()}
        rul = np.concatenate([[s[1] for s in samples] for samples in grouped.values()])
        self.target_mean = float(np.mean(rul, dtype=np.float64))
        self.target_std = float(np.std(rul, dtype=np.float64)) or 1.0
        self.groups = {}
        for key, samples in grouped.items():
            target = (np.array([s[1] for s in samples], dtype=np.float64) - self.target_mean) / self.target_std
            self.groups[key] = {'models': [None] * len(self.seeds), 'optimizers': [None] * len(self.seeds),
                                'n_inputs': samples[0][0].shape[1],
                                'x': torch.tensor(np.stack([s[0] for s in samples])),
                                'y': torch.tensor(target.astype(np.float32)),
                                'component': torch.tensor(np.array([s[2] for s in samples]), dtype=torch.long)}
        self.loss = nn.MSELoss()

    def _build(self, group, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        model = Stage2Net(self.spec.backbone, self.spec.method, group['n_inputs'], self.params,
                          self.window).to(cfg.DEVICE)
        return model, Adam(model.parameters(), lr=self.params['LEARNING_RATE'],
                           weight_decay=self.params['WEIGHT_DECAY'])

    def _epoch(self, group, model, optimizer):
        model.train()
        order = torch.randperm(len(group['x']))
        for b in range(0, len(order), self.params['BATCH_SIZE']):
            batch = order[b:b + self.params['BATCH_SIZE']]
            optimizer.zero_grad()
            prediction = model(group['x'][batch].to(cfg.DEVICE), group['component'][batch].to(cfg.DEVICE))
            self.loss(prediction, group['y'][batch].to(cfg.DEVICE)).backward()
            optimizer.step()

    def train(self, epochs):
        """Every network in turn, each built from its seed just before its training."""
        for group in self.groups.values():
            for i, seed in enumerate(self.seeds):
                model, optimizer = self._build(group, seed)
                for _ in range(epochs):
                    self._epoch(group, model, optimizer)
                group['models'][i], group['optimizers'][i] = model, optimizer

    def step(self):
        """One more epoch for every network, used for early stopping during tuning."""
        for group in self.groups.values():
            if group['models'][0] is None:
                for i, seed in enumerate(self.seeds):
                    group['models'][i], group['optimizers'][i] = self._build(group, seed)
            for model, optimizer in zip(group['models'], group['optimizers']):
                self._epoch(group, model, optimizer)

    def models_for(self, component_index):
        return self.groups[component_index if self.spec.method == 'separate' else None]['models']


def fit(spec, frames, train_events, params, window, epochs, member_seeds, stat_set='avg_std'):
    columns = input_columns(frames, train_events, stat_set)
    scaler = fit_scaler(frames, train_events, columns)
    if spec.method == 'constant':
        rul = np.array([s[1] for e in train_events for s in windows(frames, e, columns, scaler, window)],
                       dtype=np.float64)
        return Fitted(spec, columns, scaler, window, train_rul=rul)
    trainer = Trainer(spec, frames, train_events, columns, scaler, params, window, member_seeds)
    trainer.train(epochs)
    return Fitted(spec, columns, scaler, window, trainer=trainer)


# ------------------------------------------------------------------------------------------
# Prediction with Monte Carlo dropout
# ------------------------------------------------------------------------------------------

@torch.no_grad()
def mc_predict(models, samples, n_passes, target_mean=0.0, target_std=1.0, batch_size=512):
    """Every network predicts every window `n_passes` times with its dropout layers active.
    Returns the mean, the std and the draws (windows x networks*passes), in days."""
    x = torch.tensor(np.stack([s[0] for s in samples]))
    component = torch.tensor(np.array([s[2] for s in samples]), dtype=torch.long)
    draws = []
    for model in models:
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        member = []
        for b in range(0, len(x), batch_size):
            xb, cb = x[b:b + batch_size].to(cfg.DEVICE), component[b:b + batch_size].to(cfg.DEVICE)
            member.append(torch.stack([model(xb, cb) for _ in range(n_passes)]).cpu().numpy())
        draws.append(np.concatenate(member, axis=1))
    draws = np.concatenate(draws, axis=0) * target_std + target_mean
    return draws.mean(axis=0), draws.std(axis=0), np.clip(draws.T, 0.0, None)


def predict_trainer(trainer, samples, n_passes=N_PASSES):
    scale = (trainer.target_mean, trainer.target_std)
    if trainer.spec.method != 'separate':
        return mc_predict(trainer.models_for(None), samples, n_passes, *scale)
    index = np.array([s[2] for s in samples])
    mean, std, draws = np.empty(len(samples)), np.empty(len(samples)), None
    for component in np.unique(index):
        rows = np.where(index == component)[0]
        m, s, d = mc_predict(trainer.models_for(int(component)), [samples[i] for i in rows], n_passes, *scale)
        if draws is None:
            draws = np.empty((len(samples), d.shape[1]))
        mean[rows], std[rows], draws[rows] = m, s, d
    return mean, std, draws


def training_labels(case, config):
    """Largest and smallest RUL label of every training failure of a case, and the Constant
    baseline (the mean of all training labels)."""
    frames, failures = load_frames(config['resample_freq'], config['stat_set'])
    train_events, _ = cases(frames, failures)[case - 1]
    fitted = fit(Spec.of('constant'), frames, train_events, {}, int(config['window']), 0, None, config['stat_set'])
    rows = []
    for event in train_events:
        labels = np.array([s[1] for s in windows(frames, event, fitted.columns, fitted.scaler, fitted.window)])
        rows.append({'event': event, 'name': cfg.FAILURE_NAME[event], 'component': frames[event][1]['component'],
                     'max_label': labels.max(), 'min_label': labels.min(), 'windows': len(labels)})
    table = pd.DataFrame(rows).sort_values(['component', 'max_label'], ascending=[False, True])
    return table, float(np.mean(fitted.train_rul))


def predict(fitted, samples, n_passes=N_PASSES):
    if fitted.spec.method == 'constant':
        rng = np.random.default_rng(cfg.SEED)
        draws = np.clip(rng.choice(fitted.train_rul, size=(len(samples), N_NETWORKS * n_passes),
                                   replace=True), 0.0, None)
        return draws.mean(axis=1), draws.std(axis=1), draws
    return predict_trainer(fitted.trainer, samples, n_passes)
