# Energy Consumption Forecasting & Cost Optimization — Steel Manufacturing Plant

Predicts a steel plant's energy consumption (`Usage_kWh`) from industrial sensor
readings, benchmarks the model against a naive baseline forecast, and simulates
a tariff-based load-shifting strategy to estimate potential cost savings.

Dataset: [Steel Industry Energy Consumption](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption)
(UCI ML Repository) — 15-minute-interval readings for one year from a steel
plant, including reactive/leading power, power factor, CO2 output, load type,
and timestamp features.

## Pipeline

```
src/component/data_ingestion.py       -> reads notebook/cleaned_data.csv, splits train/test (80/20)
src/component/data_transformation.py  -> imputes, scales, encodes features into a sklearn ColumnTransformer
src/component/model_trainer.py        -> trains + compares 10 regressors, tunes and ensembles the best 3
src/component/cost_optimization.py    -> tariff-based load-shifting cost simulation
src/pipeline/train_pipeline.py        -> runs the full pipeline end-to-end
src/pipeline/predict_pipeline.py      -> loads the saved model/preprocessor for single predictions
application.py                        -> Flask app serving a prediction form (templates/home.html)
```

Run the full pipeline:

```bash
pip install -r requirements.txt
python -m src.pipeline.train_pipeline
```

This ingests the data, retrains the model, regenerates
`artifacts/metrics_report.json`, and reruns the load-shift cost simulation.

Run the web app:

```bash
python application.py   # serves on http://localhost:5001
```

## Model

Ten regressors (Linear/Ridge/Lasso, KNN, Decision Tree, Random Forest,
XGBoost, CatBoost, Gradient Boosting, AdaBoost) are trained and compared by
test R². The top candidates (CatBoost, tuned via `RandomizedSearchCV`; KNN,
tuned via `GridSearchCV`; and XGBoost) are combined into a weighted
`VotingRegressor` (weights `[3, 2, 1]`), which is saved as the final model
(`artifacts/model.pkl`).

Typical test-set performance (varies run-to-run — search isn't seeded):

| Metric | Value |
|---|---|
| R² | ~0.81–0.89 |
| MAE | ~5.8–8.9 kWh |
| RMSE | ~11.4–14.4 kWh |
| MAPE | ~20–49% |

Full numbers from the latest run are written to `artifacts/metrics_report.json`.

### Baseline comparison

The model's MAPE is benchmarked against a **seasonal-naive baseline** — for
each reading, predict the historical average `Usage_kWh` for the same
`Load_Type` / `WeekStatus` / 2-hour time bucket (i.e. "what a plant engineer
would guess without a model"). This baseline is computed in
`seasonal_naive_baseline()` (`src/utils.py`) and compared against the trained
model in `ModelTrainer._evaluate_baseline()`. The improvement percentage is
logged and saved alongside the other metrics — nothing here is asserted
without a measured comparison.

## Cost optimization: tariff-based load shifting

`src/component/cost_optimization.py` simulates shifting a share of the
plant's peak-hour load into cheaper off-peak hours under a configurable
Time-of-Use (ToU) tariff:

- **`TariffConfig`** — peak / standard / off-peak ₹-per-kWh rates and the
  hour-of-day windows they apply to. Defaults approximate a typical
  industrial ToD tariff card; replace with the plant's real one.
- **`LoadShiftConfig`** — `shiftable_fraction`, the share of peak-hour energy
  assumed movable to off-peak hours (flexible batch/material-handling work,
  not continuous furnace processes).

The simulator aggregates actual usage into an hourly profile, moves
`shiftable_fraction` of each peak hour's energy into the off-peak hours
(total energy is conserved — only its timing changes), and compares tariff
cost before vs. after.

Run standalone:

```bash
python -m src.component.cost_optimization
```

Outputs (Power BI-ready):

| File | Contents |
|---|---|
| `artifacts/load_shift_report.csv` | Per-hour usage/cost before vs. after shifting |
| `artifacts/load_shift_scenarios.csv` | Savings % at shift fractions 10/20/30/40/50% |
| `artifacts/load_shift_summary.json` | Headline numbers for the default scenario |

At the default 40% shiftable fraction, the simulation projects roughly a
**12% reduction** in peak-tariff energy cost (see `load_shift_scenarios.csv`
for the full sensitivity range, ~3–15% across the tested fractions).

## Project structure

```
artifacts/           trained model, preprocessor, train/test/raw data, generated reports
notebook/             EDA notebook and raw/cleaned source data
src/component/         data ingestion, transformation, model training, cost optimization
src/pipeline/          training and prediction pipelines
src/exception.py       custom exception wrapper
src/logger.py          logging config (writes to logs/)
templates/              Flask HTML templates
application.py          Flask app entry point
```
