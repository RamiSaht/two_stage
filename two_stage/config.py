import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('TWO_STAGE_DATA', ROOT / 'data'))
WIND_FARM = 'B'
DATA_DIR = DATA_ROOT / f'Wind Farm {WIND_FARM}'

RESULTS_DIRS = {'paper': ROOT / 'results', 'retrained': ROOT / 'results_retrained'}
OUTPUT_DIRS = {'paper': ROOT / 'outputs', 'retrained': ROOT / 'outputs_retrained'}
_results = 'paper'

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RAW_FREQ = '10min'
NORMAL_STATUS = 0
FFILL_LIMIT = 6

COMPONENTS = ('bearing', 'transformer')
COMPONENT_INDEX = {'bearing': 0, 'transformer': 1}
COMPONENT_OF = {'ROTOR_BEARING': 'bearing', 'MAIN_BEARING': 'bearing', 'TRANSFORMER': 'transformer'}
TARGET_OF = {'bearing': 'sensor_51_avg', 'transformer': 'sensor_47_avg'}
FAILURE_NAME = {34: 'T1', 7: 'T2', 19: 'T3', 53: 'B1', 27: 'B2', 77: 'B3'}

STATUS_TEXT = {0: 'normal', 1: 'derated', 2: 'idling', 3: 'service', 4: 'down / fault',
               5: 'other (test, ice, ...)'}

ALPHA_GRID = np.arange(0.0, 1.001, 0.01)


def use_results(name):
    """'paper' reads the stored results of the paper. 'retrained' reads the results of notebook 0
    in results_retrained/, and the stored results for everything that was not retrained."""
    global _results
    if name not in RESULTS_DIRS:
        raise ValueError(f'unknown results {name!r}, use one of {list(RESULTS_DIRS)}')
    _results = name


def active_results():
    return _results


def result_path(*parts):
    """The file to read."""
    path = RESULTS_DIRS[_results].joinpath(*parts)
    return path if path.exists() else RESULTS_DIRS['paper'].joinpath(*parts)


def output_path(*parts):
    """The file to write. The stored results of the paper are never written to."""
    if _results == 'paper':
        raise RuntimeError("results are only written after use_results('retrained')")
    path = RESULTS_DIRS[_results].joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def figure_dir(notebook):
    return OUTPUT_DIRS[_results] / notebook


def failure_label(event):
    return f'{FAILURE_NAME[event]} ({event})' if event in FAILURE_NAME else str(event)
