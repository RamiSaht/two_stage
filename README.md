# Two-stage framework for component-level anomaly detection and RUL prediction

Code of the paper *Two-Stage Framework for Component-Level Anomaly Detection and Remaining Useful Life Prediction in Offshore Wind Turbines* (Rami Sahtout, MSc thesis, Delft University of Technology, 2026).

The first stage monitors each component with its own normal behavior model and a CUSUM detector, and attributes an anomaly to the component whose detector alarms first. The second stage estimates the remaining useful life (RUL) of that component with a neural network that uses the component label, and quantifies its uncertainty with Monte Carlo dropout.

## Setup

```bash
conda create -n two_stage python=3.14
conda activate two_stage
pip install -r requirements.txt
```

The pinned versions are the ones that produced the results. Only notebook 0 needs a GPU. The figures use the font CMU Sans Serif when it is installed and fall back to DejaVu Sans otherwise.

## Data

The CARE-to-Compare dataset, version 6 ([Zenodo](https://doi.org/10.5281/zenodo.15846963)). Extract it into `data/` so that `data/Wind Farm B/event_info.csv` exists. Only wind farm B is used.

If the dataset is already stored elsewhere, there is no need to copy it. Set the environment variable `TWO_STAGE_DATA` to the folder that contains `Wind Farm B` instead, before starting Jupyter.

```bash
export TWO_STAGE_DATA=/path/to/CARE_To_Compare
```

This only lasts for the current terminal session. To keep it set whenever the environment is active, and only then, store it in the conda environment.

```bash
conda env config vars set TWO_STAGE_DATA=/path/to/CARE_To_Compare
```

The code only reads this variable and never changes it. Without it, the data is read from `data/`.

## Notebooks

| Notebook | Contents | In the paper |
|---|---|---|
| `0_fine_tuning_and_training` | Trains the models again and writes `results_retrained/` | |
| `1_anomaly_detection` | Data, downtime filter and stage 1 | Tables 3, 6, 7 and 8, Figs. 4, 10 and 11 |
| `2_prognostics` | Stage 2 | Figs. 7 and 12 to 19 |
| `3_miscellaneous_figures` | Figures without data behind them | Figs. 1, 2, 5, 6, 8 and 9 |

Notebooks 1 to 3 read the stored results in `results/`, so they do not need notebook 0 or a GPU. Each writes its figures as PNG files to its own folder in `outputs/`, with the figures of the paper in `paper/`, named by figure number, and the other figures in `extra/`. Fig. 3 was drawn by hand and is not included.

The setting `RESULTS` at the top of notebooks 1 to 3 chooses the results they read. `'paper'`, the default, reads the stored results in `results/`. `'retrained'` reads the results of notebook 0 in `results_retrained/`, uses the stored results for every step that was not retrained, and writes the figures to `outputs_retrained/`.

## Code

| Module | Contents |
|---|---|
| `config.py` | Paths and constants |
| `data.py` | Loading and the shared preprocessing |
| `stage1.py` | Normal behavior models, CUSUM detector and CARE score |
| `stage2.py` | Data, networks, training and Monte Carlo dropout of stage 2 |
| `metrics.py` | CRPS, coverage, sharpness and reliability score |
| `tuning.py` | Hyperparameter tuning with Optuna |
| `experiments.py` | The training runs behind the results |
| `evaluation.py` | Aggregation of the results, model selection and t-tests |
| `plots.py`, `schematics.py` | Figures |

## Results

`results/` holds the results of the paper. These are the residuals of the 30 normal behavior models, the tuned configuration of every model, the scores of all models on Cases 1 to 9 over five repeats, the Case 5 predictions of Fig. 14, the two Monte Carlo dropout checks, and the draws of Figs. 8 and 9. The trained weights were not kept.

Notebook 0 trains every step again. It writes only to `results_retrained/` and never changes `results/`. Seeds are fixed, but some GPU operations are not deterministic, so retraining matches the stored results closely but not bit for bit. The stored results were also produced before a correction of the one-hour forward fill, which had no effect because the dataset leaves missing records out instead of writing empty rows. The code now puts these records back before filling. The stage 2 windows and RUL labels are unchanged, and the inputs shift slightly.

## References

- C. Gück, C. M. A. Roelofs, and S. Faulstich (2024). CARE to Compare: A real-world benchmark dataset for early fault detection in wind turbine data. *Data*, 9(12), 138.
- C. Gück and C. Roelofs (2025). Wind turbine SCADA data for early fault detection (Version 6) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15846963
- I. de Pater and M. Mitici (2022). Novel metrics to evaluate probabilistic remaining useful life prognostics with applications to turbofan engines. *PHM Society European Conference*, 7(1), 96–109.
- D. O'Connor, V. Pakrashi, and B. Ghosh (2026). Detecting changes in normal wind turbine operations using data driven approaches. *Journal of Physics: Conference Series*, 3224(6), 062018.

## License

The code is released under the MIT license. The stored results are derived from the CARE-to-Compare dataset and are shared under its license, CC BY-SA 4.0.
