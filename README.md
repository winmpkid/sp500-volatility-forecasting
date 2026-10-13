# S&P 500 Volatility Forecasting

Forecasting next-trading-day S&P 500 log realized variance with heterogeneous autoregressive (HAR) models and lag features.

The project compares persistence, daily/weekly/monthly feature combinations, standard HAR, and HAR + Lag 2. Evaluation uses chronological training/validation/test splits and time-series cross-validation with a gap.

## Project files

- `Report.docx`: project report.
- `src/volatility/`: data loading, feature engineering, diagnostics, benchmarks, and final evaluation.
- `datasets/oxfordmanrealizedvolatilityindices.zip`: compressed copy of the original Oxford-Man realized volatility CSV.
- `outputs/`: saved HAR + Lag 2 model and evaluation metadata.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m zipfile -e datasets/oxfordmanrealizedvolatilityindices.zip datasets
```

The dataset is compressed to fit GitHub's web upload limit. Extract it before running; the code expects `datasets/oxfordmanrealizedvolatilityindices.csv`.

## Run

From the repository root:

```bash
# Training-period market and return diagnostics
python src/volatility/cli.py

# Model benchmarks and time-series cross-validation
python src/volatility/benchmark.py

# Final held-out evaluation and model export
python src/volatility/final_test.py
```

The original source files are included unchanged. `predict.py` is currently an empty placeholder.

## Saved evaluation

The included model metadata records 924 final-test observations and the following HAR + Lag 2 results:

| Metric | Value |
| --- | ---: |
| MSE (log realized variance) | 0.422655 |
| MAE (log realized variance) | 0.511852 |
| MSE improvement vs. persistence | 11.60% |
| MSE improvement vs. standard HAR | 1.49% |

These are the saved project results, not a new execution. The saved model records scikit-learn 1.9.0; retraining with the local environment is preferable if loading it with a different version.
