from functools import lru_cache

import pandas as pd

from . import config as cfg

STAT_SUFFIXES = ('avg', 'max', 'min', 'std')
STAT_DESCRIPTOR = {'avg': 'average', 'max': 'maximum', 'min': 'minimum', 'std': 'std_dev'}

# In the raw files the minimum and std columns are swapped for most sensors. sensor_26 is the
# exception: its `_std` holds the average and its `_avg` a per-unit copy of the active power.
# The energy counters and two other sensors only record an average.
_MIN_STD_SWAPPED = {'min': 'std', 'std': 'min'}
STAT_LABEL_EXCEPTIONS = {'sensor_26': {'avg': 'std', 'std': 'min'}}
STAT_LABEL_RENAMED = {'sensor_26_avg': 'sensor_26_per_unit'}
AVERAGE_ONLY_SENSORS = {'sensor_0', 'sensor_1', 'sensor_2', 'sensor_3', 'sensor_5', 'sensor_9'}


def stat_of(column):
    suffix = column.rsplit('_', 1)[-1]
    return suffix if suffix in STAT_SUFFIXES else None


def load_info(farm_dir=None):
    farm_dir = farm_dir or cfg.DATA_DIR
    info = pd.read_csv(farm_dir / 'event_info.csv', sep=';', parse_dates=['event_start', 'event_end'])
    info['asset_id'] = info['asset_id'].astype(str).str.replace('#', '', regex=False).astype(int)
    return info


def load_feature_description(farm_dir=None):
    return pd.read_csv((farm_dir or cfg.DATA_DIR) / 'feature_description.csv', sep=';')


def fix_statistic_labels(df):
    present = {c.rsplit('_', 1)[0] for c in df.columns if stat_of(c)}
    empty = [f'{base}_{s}' for base in present & AVERAGE_ONLY_SENSORS for s in ('min', 'max', 'std')
             if f'{base}_{s}' in df.columns]
    rename = dict(STAT_LABEL_RENAMED)
    for base in present - AVERAGE_ONLY_SENSORS:
        if all(f'{base}_{s}' in df.columns for s in STAT_SUFFIXES):
            for true_stat, labelled in STAT_LABEL_EXCEPTIONS.get(base, _MIN_STD_SWAPPED).items():
                rename[f'{base}_{labelled}'] = f'{base}_{true_stat}'
    return df.drop(columns=empty).rename(columns={old: new for old, new in rename.items() if old in df.columns})


def fill_short_gaps(df, limit=cfg.FFILL_LIMIT, freq=cfg.RAW_FREQ):
    """Reinsert the absent 10-minute records and forward fill at most `limit` of them.

    The export leaves records out instead of writing empty cells, so the records have to be put
    back on the grid before a forward fill can bridge them. Status and period are carried along,
    so a bridged record keeps the operating state it continues.
    """
    grid = pd.date_range(df['time_stamp'].min(), df['time_stamp'].max(), freq=freq)
    return (df.set_index('time_stamp').reindex(grid).ffill(limit=limit)
            .rename_axis('time_stamp').reset_index())


@lru_cache(maxsize=None)
def _read_event(event_id):
    df = pd.read_csv(cfg.DATA_DIR / 'datasets' / f'{event_id}.csv', sep=';',
                     parse_dates=['time_stamp'], low_memory=False)
    return df.sort_values('time_stamp').reset_index(drop=True)


def load_raw(event_id, fix_labels=True):
    df = _read_event(event_id)
    return fix_statistic_labels(df) if fix_labels else df.copy()


def add_operating_rul(df, event_end):
    """RUL in operating days on a status-filtered 10-minute frame: every kept record counts for at
    most 10 minutes, so time in which the turbine is not operating does not count."""
    ts = df['time_stamp']
    dt = (ts.shift(-1) - ts).dt.total_seconds().clip(upper=600).fillna(0) / 86400
    dt[ts >= event_end] = 0.0
    rul = dt.iloc[::-1].cumsum().iloc[::-1]
    df = df[ts < event_end].copy()
    df['rul_days'] = rul[ts < event_end]
    return df


# ------------------------------------------------------------------------------------------
# Dataset summary (Table 3) and downtime filter (Fig. 4)
# ------------------------------------------------------------------------------------------

def dataset_summary(info=None, features=None):
    """Wind farm B as in Table 3. Main bearing and rotor bearing count as one component."""
    info = load_info() if info is None else info
    features = load_feature_description() if features is None else features
    anomalies = info[info['event_label'] == 'anomaly']
    return pd.Series({
        'Total turbines': info['asset_id'].nunique(),
        'Turbines with anomalies': anomalies['asset_id'].nunique(),
        'Time series': len(info),
        'Anomalies': len(anomalies),
        'Healthy time series': int((info['event_label'] == 'normal').sum()),
        'Features': int(features['statistics_type'].str.split(',').str.len().sum()),
        'Available components': anomalies['component'].map(cfg.COMPONENT_OF).nunique(),
    }, name=f'Wind farm {cfg.WIND_FARM}')


def removed_records(info=None):
    """Records removed by the downtime filter per event and period, by status code."""
    info = load_info() if info is None else info
    rows = []
    for _, event in info.iterrows():
        df = _read_event(int(event['event_id']))[['time_stamp', 'train_test', 'status_type_id']]
        for period, name in (('train', 'training period'), ('prediction', 'prediction period')):
            part = df[df['train_test'] == period]
            counts = part['status_type_id'].value_counts()
            rows.append({'event': int(event['event_id']), 'period': name, 'records': len(part),
                         'removed': int(len(part) - counts.get(cfg.NORMAL_STATUS, 0)),
                         **{f'n_{code}': int(counts.get(code, 0)) for code in cfg.STATUS_TEXT}})
    table = pd.DataFrame(rows)
    table['removed_pct'] = 100 * table['removed'] / table['records']
    return table
