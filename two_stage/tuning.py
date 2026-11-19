"""Hyperparameter tuning of the stage 2 networks with Optuna, on Case 1 (T1 and B1 held out)."""
import numpy as np

from . import config as cfg
from . import metrics, stage2

TUNING_CASE = 1
N_TRIALS = 20
N_SEED_REPEATS = 3
MAX_EPOCHS = 40
PATIENCE = 5
DROPOUT = 0.5
FREQ_GRID = ['1h', '8h', '24h']
WINDOW_DAYS_GRID = [7.0, 14.0, 28.0]
STAT_SET = 'avg_std'
CNN_LAYOUTS = {'16': (16,), '32': (32,), '16-32': (16, 32), '32-32': (32, 32), '32-64': (32, 64)}
CONSTANT_CONFIG = {'params': {}, 'resample_freq': '8h', 'window_days': 14.0, 'window': 42, 'stat_set': STAT_SET,
                   'epochs': 0}


def suggest(trial, spec):
    params = {
        'SHARED_FC_SIZE': trial.suggest_categorical('SHARED_FC_SIZE', [16, 32, 64]),
        'HEAD_NEURONS': trial.suggest_categorical('HEAD_NEURONS', [16, 32, 64]),
        'CNN_DROPOUT': DROPOUT, 'FC_DROPOUT': DROPOUT, 'HEAD_DROPOUT': DROPOUT,
        'LEARNING_RATE': trial.suggest_float('LEARNING_RATE', 1e-4, 1e-2, log=True),
        'WEIGHT_DECAY': trial.suggest_float('WEIGHT_DECAY', 1e-6, 1e-1, log=True),
        'BATCH_SIZE': trial.suggest_categorical('BATCH_SIZE', [16, 32, 64]),
        'CNN_kernel_size': 3, 'CNN_out_channels': (32,), 'CNN_POOL': 'flatten',
        'LSTM_hidden_size': 32, 'LSTM_num_layers': 1, 'EMB_DIM': 2,
    }
    if 'cnn' in spec.backbone:
        params['CNN_kernel_size'] = trial.suggest_categorical('CNN_kernel_size', [3, 5, 7])
        params['CNN_out_channels'] = CNN_LAYOUTS[trial.suggest_categorical('CNN_out_channels', list(CNN_LAYOUTS))]
    if spec.backbone == 'cnn':
        params['CNN_POOL'] = trial.suggest_categorical('CNN_POOL', ['flatten', 'last', 'avg'])
    if 'lstm' in spec.backbone:
        params['LSTM_hidden_size'] = trial.suggest_categorical('LSTM_hidden_size', [16, 32, 64, 128])
        params['LSTM_num_layers'] = trial.suggest_categorical('LSTM_num_layers', [1, 2])
    if spec.method == 'embed':
        params['EMB_DIM'] = trial.suggest_categorical('EMB_DIM', [2, 4])
    return params


def validation_score(spec, frames, train_events, val_events, params, window, max_epochs=MAX_EPOCHS,
                     patience=PATIENCE, repeats=N_SEED_REPEATS):
    """Train `repeats` independently seeded ensembles one epoch at a time and return the best mean
    validation CRPS and the epoch at which it was reached."""
    columns = stage2.input_columns(frames, train_events, STAT_SET)
    scaler = stage2.fit_scaler(frames, train_events, columns)
    validation = {e: stage2.windows(frames, e, columns, scaler, window) for e in val_events}
    trainers = [stage2.Trainer(spec, frames, train_events, columns, scaler, params, window, stage2.seeds(k, 0))
                for k in range(repeats)]
    best, best_epoch, waited = np.inf, 0, 0
    for epoch in range(1, max_epochs + 1):
        for trainer in trainers:
            trainer.step()
        score = float(np.mean([np.mean([metrics.crps(stage2.predict_trainer(trainer, samples)[2],
                                                     np.array([s[1] for s in samples]))
                                        for samples in validation.values()])
                               for trainer in trainers]))
        if score < best - 1e-9:
            best, best_epoch, waited = score, epoch, 0
        else:
            waited += 1
            if waited >= patience:
                break
    return best, best_epoch


def tune(model, n_trials=N_TRIALS):
    """Tuned configuration of one network: hyperparameters, bin size, window and number of epochs."""
    import optuna

    if model == 'constant':
        return dict(CONSTANT_CONFIG)
    spec = stage2.Spec.of(model)

    def objective(trial):
        params = suggest(trial, spec)
        freq = trial.suggest_categorical('RESAMPLE_FREQ', FREQ_GRID)
        window = stage2.n_bins(trial.suggest_categorical('WINDOW_DAYS', WINDOW_DAYS_GRID), freq)
        frames, failures = stage2.load_frames(freq, STAT_SET)
        train_events, val_events = stage2.cases(frames, failures)[TUNING_CASE - 1]
        score, epoch = validation_score(spec, frames, train_events, val_events, params, window)
        trial.set_user_attr('epoch', epoch)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=cfg.SEED))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    freq, window_days = best.params['RESAMPLE_FREQ'], float(best.params['WINDOW_DAYS'])
    return {'params': suggest(optuna.trial.FixedTrial(best.params), spec), 'resample_freq': freq,
            'window_days': window_days, 'window': stage2.n_bins(window_days, freq), 'stat_set': STAT_SET,
            'epochs': int(best.user_attrs['epoch']), 'score': float(best.value)}
