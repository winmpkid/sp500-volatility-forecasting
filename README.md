# S&P 500 Realized Volatility Forecasting with HAR-RV

This project forecasts the next trading day's log realized variance for the S&P 500 Index. It starts with a persistence benchmark, evaluates the daily, weekly, and monthly components of a heterogeneous autoregressive realized volatility model, and uses validation residual diagnostics to test whether a second-lag feature adds predictive information.

The selected specification is HAR + Lag 2. On the independent test set, it reduces mean squared error by 11.60% and mean absolute error by 7.16% relative to persistence. Compared with the standard HAR-RV model, the improvements are 1.49% and 0.61%, respectively. The Lag 2 contribution is therefore best described as a small but consistent incremental gain. Significant test-period residual autocorrelation remains.

## Objectives

- Construct every feature from information available at the forecast origin.
- Compare persistence, standard HAR-RV, and HAR + Lag 2 in chronological order.
- Combine forecast errors with residual diagnostics when evaluating additional features.
- Save the fitted model, metadata, and prediction history for reproducibility.

## Data and Forecast Target

The source data is stored at:

```text
datasets/oxfordmanrealizedvolatilityindices.csv
```

The model uses `.SPX` observations. The source variable `rv5` is treated as daily realized variance, and the target is:

```text
target(t) = log(RV(t+1))
```

The log transformation compresses the scale of realized variance. Although the project title uses the common phrase “volatility forecasting,” the model target and reported error metrics are defined on realized variance and log realized variance.

## Feature Construction

When forecasting `log(RV(t+1))`, the model uses only information available through date `t`:

| Feature | Definition | Interpretation |
|---|---|---|
| `log_rv_day` | `log(RV(t))` | Current daily realized variance |
| `log_rv_lag2` | `log(RV(t-1))` | Realized variance two trading days behind the target |
| `log_rv_week` | `log(mean(RV(t-4:t)))` | Log of mean realized variance over the latest 5 trading days |
| `log_rv_month` | `log(mean(RV(t-21:t)))` | Log of mean realized variance over the latest 22 trading days |

The selected model is:

```text
log(RV(t+1))
    = beta_0
    + beta_day   * log_rv_day
    + beta_lag2  * log_rv_lag2
    + beta_week  * log_rv_week
    + beta_month * log_rv_month
    + error(t+1)
```

The weekly and monthly components first average realized variance and then apply the logarithm. Training and prediction use the same feature-building functions.

## Why Chronological Validation Is Required

Random cross-validation can train on later observations and evaluate on earlier observations, which does not reproduce a real forecasting workflow. Because each feature row ends on date `t` and its target occurs on `t+1`, randomly splitting adjacent rows can also allow a validation target to appear inside a training feature window.

The project limits leakage as follows:

- It creates initial 60%, 20%, and 20% training, validation, and test periods in date order.
- It checks both feature dates and target dates at split boundaries.
- Development-stage cross-validation uses `TimeSeriesSplit(n_splits=5, gap=1)`.
- The test set is excluded from feature and model selection.
- The selected model is evaluated on the independent test set only after the specification is frozen.

## Model Selection Evidence

The persistence benchmark uses the current `log(RV)` as the forecast for the next trading day. The development process then compares Day, Day + Week, and Day + Week + Month to determine whether the weekly and monthly components reduce forecast error.

The standard HAR-RV model's validation residuals have lag-2 ACF and PACF values of approximately 0.088, while lags 3 and 4 are close to zero. This motivates `log_rv_lag2` as a candidate rather than adding arbitrary extra lags. After adding Lag 2, the validation residual Ljung-Box p-values at lags 5 and 10 rise from below 0.05 to above 0.05. At the 5% level, those two tests no longer reject the null of no residual autocorrelation, although this does not establish complete residual independence.

Across five chronological validation folds, the mean MSE decreases from approximately 0.350979 for standard HAR-RV to 0.345829 for HAR + Lag 2, an improvement of about 1.47%. MSE decreases in all five folds. The comparison holds Day, Week, and Month fixed and changes only the Lag 2 component.

Residual diagnostics generate candidate features. Cross-validation and ablation evidence determine whether to retain them. The independent test set evaluates the frozen specification and is not used for repeated feature tuning.

## Final Test Results

After model selection, the training and validation periods are combined for refitting. The final evaluation contains 924 independent test observations. All error metrics are calculated on the log realized variance scale.

| Model | MSE | MAE | MSE Improvement vs. Persistence | MAE Improvement vs. Persistence |
|---|---:|---:|---:|---:|
| Persistence | 0.478135 | 0.551313 | 0.00% | 0.00% |
| HAR | 0.429034 | 0.514996 | 10.27% | 6.59% |
| HAR + Lag 2 | **0.422655** | **0.511852** | **11.60%** | **7.16%** |

Relative to standard HAR-RV, HAR + Lag 2 reduces MSE by approximately 1.49% and MAE by approximately 0.61%. Validation, time-series cross-validation, and final testing all show improvement in the same direction. Statistical significance would require additional forecast-comparison testing.

## Repository Structure

```text
sp500-volatility-forecasting/
├── reports/
│   ├── HAR_RV_Project_Report.docx
│   └── archive/
│       └── HAR_RV_Project_Report_Early_Version.docx
├── datasets/
│   └── oxfordmanrealizedvolatilityindices.csv
├── outputs/
│   ├── har_lag2_model.joblib
│   ├── har_lag2_model_info.json
│   └── predictions.csv
├── src/volatility/
│   ├── data.py
│   ├── features.py
│   ├── validation.py
│   ├── models.py
│   ├── evaluation.py
│   ├── diagnostics.py
│   ├── cli.py
│   ├── benchmark.py
│   ├── final_test.py
│   └── predict.py
├── tests/
│   ├── test_features.py
│   ├── test_model_artifacts.py
│   └── test_validation.py
├── requirements.txt
├── run_pipeline.py
└── README.md
```

| Module | Purpose |
|---|---|
| `data.py` | Load the source data |
| `features.py` | Build training and prediction features |
| `validation.py` | Create chronological training, validation, and test periods |
| `models.py` | Fit models and save model artifacts |
| `evaluation.py` | Calculate MSE and MAE |
| `diagnostics.py` | Produce descriptive statistics, ACF, PACF, Q-Q, and Ljung-Box diagnostics |
| `benchmark.py` | Run feature ablation, time-series cross-validation, and candidate-model comparisons |
| `final_test.py` | Run the independent final evaluation and save the selected model |
| `predict.py` | Load the saved model and produce the next historical forecast |

## Environment Setup

The project has been verified with Python 3.14.6. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The model metadata records the scikit-learn version used when the artifact was saved. Loading the model with the same version reduces serialization compatibility risk.

## Run the Project

Run the complete workflow from the repository root:

```bash
python run_pipeline.py --stage all
```

The default uses a headless plotting backend. To display Matplotlib windows:

```bash
python run_pipeline.py --stage all --interactive-plots
```

Individual stages are also available:

```bash
python run_pipeline.py --stage diagnostics
python run_pipeline.py --stage benchmark
python run_pipeline.py --stage final-test
python run_pipeline.py --stage predict
```

### Training-Period Diagnostics

```bash
python src/volatility/cli.py
```

### Feature Ablation and Time-Series Cross-Validation

```bash
python src/volatility/benchmark.py
```

This stage uses development data only. Complete model selection before running the final test, and do not repeatedly tune the specification against test results.

### Final Test and Model Export

```bash
MPLBACKEND=Agg python src/volatility/final_test.py
```

This updates:

```text
outputs/har_lag2_model.joblib
outputs/har_lag2_model_info.json
```

The joblib file contains the fitted linear regression. The JSON file records feature order, coefficients, training and test periods, software version, and evaluation metrics.

### Generate a Prediction

```bash
python src/volatility/predict.py
```

Prediction history is written to:

```text
outputs/predictions.csv
```

Runs are deduplicated by feature date and model save time. Repeating a prediction with the same model version updates the existing record; retraining and saving a new model version retains a separate record.

## Automated Tests

```bash
python -m unittest discover -s tests -v
```

The tests verify that:

- Prediction features retain the latest valid feature date.
- Training features omit the final row when its next-day target is unavailable.
- `target` equals the next observation's `log(RV)`.
- Chronological split boundaries do not overlap future targets.
- Saved model feature counts and order match the JSON metadata.
- The saved model can be loaded and used for prediction.

## Converting Predictions Back to the Original Scale

The direct model output is next-trading-day log realized variance:

```text
predicted_log_rv
```

Exponentiation returns the value to the realized variance scale:

```python
predicted_rv = np.exp(predicted_log_rv)
```

Taking the square root gives the corresponding daily volatility value:

```python
predicted_volatility = np.sqrt(predicted_rv)
```

The current saved prediction uses the feature timestamp `2018-06-26T23:00:00+00:00`, corresponding to `2018-06-27 00:00:00+01:00` in the source data. The code normalizes timestamps to UTC. This is a forecast at the end of the historical sample, not a live market forecast.

## Limitations

- Significant Ljung-Box results remain in the final test residuals, so the model does not capture all test-period time dependence.
- The linear specification has limited ability to represent nonlinear relationships or regime changes and does not include VIX, volume, macroeconomic variables, jumps, or leverage effects.
- `exp(predicted_log_rv)` is not automatically the conditional mean on the original scale. Mean forecasts may require log-transformation bias correction.
- The saved model is fitted on the training and validation periods. A deployment version could be refitted on all available history after freezing the specification, while preserving the current evaluation record.
- Historical backtest performance does not guarantee future results. This project is not investment advice.

## Possible Extensions

- Add volatility-specific loss functions such as QLIKE.
- Apply a Diebold-Mariano test to compare forecast errors.
- Compare Ridge, Lasso, GARCH, EGARCH, and nonlinear models.
- Add external market variables while retaining chronological validation.
- Expand tests for split boundaries, invalid observations, and prediction-history deduplication.
- Compare rolling-window and expanding-window refitting schemes.

## Conclusion

Under the current sample and validation design, HAR-RV outperforms persistence and Lag 2 provides a further small reduction in forecast error. The evidence supports retaining Lag 2 in this specification, but it does not establish that the feature will remain useful in other market regimes. Residual autocorrelation in the independent test period remains the main unresolved modeling limitation.
