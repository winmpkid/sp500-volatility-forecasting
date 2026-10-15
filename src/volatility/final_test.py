from pathlib import Path

import pandas as pd

from data import load_raw_data
from diagnostics import (
    plot_final_test_comparison,
    plot_residual_diagnostics,
    run_ljung_box_tests,
)
from evaluation import evaluate_model
from features import build_features
from models import save_model_artifacts, train_model
from validation import chronological_split


FINAL_MODELS = {
    "HAR": [
        "log_rv_day",
        "log_rv_week",
        "log_rv_month",
    ],
    "HAR + Lag 2": [
        "log_rv_day",
        "log_rv_lag2",
        "log_rv_week",
        "log_rv_month",
    ],
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def relative_improvement(reference_value, candidate_value):
    return (reference_value - candidate_value) / reference_value


def run_final_test():
    """Evaluate the frozen specification once on the untouched test set."""
    raw_data = load_raw_data()
    model_data = build_features(raw_data)
    train_data, validation_data, test_data = chronological_split(
        model_data
    )

    # Model selection is complete, so refit on all development data.
    development_data = (
        pd.concat(
            [train_data, validation_data],
            ignore_index=True,
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    y_development = development_data["target"]
    y_test = test_data["target"]

    predictions = {
        "Persistence": test_data["log_rv_day"].to_numpy(),
    }
    trained_models = {}

    for model_name, feature_names in FINAL_MODELS.items():
        model = train_model(
            development_data[feature_names],
            y_development,
        )
        trained_models[model_name] = model
        predictions[model_name] = model.predict(
            test_data[feature_names]
        )

    rows = []
    for model_name, prediction in predictions.items():
        metrics = evaluate_model(y_test, prediction)
        rows.append({"model": model_name, **metrics})

    result_table = pd.DataFrame(rows).set_index("model")

    persistence_mse = result_table.loc["Persistence", "mse"]
    persistence_mae = result_table.loc["Persistence", "mae"]
    result_table["mse_improvement_vs_persistence"] = (
        persistence_mse - result_table["mse"]
    ) / persistence_mse
    result_table["mae_improvement_vs_persistence"] = (
        persistence_mae - result_table["mae"]
    ) / persistence_mae

    har_mse = result_table.loc["HAR", "mse"]
    har_mae = result_table.loc["HAR", "mae"]
    lag2_mse = result_table.loc["HAR + Lag 2", "mse"]
    lag2_mae = result_table.loc["HAR + Lag 2", "mae"]

    print("\nFinal test period:")
    print(
        f"{test_data['target_date'].min()} to "
        f"{test_data['target_date'].max()}"
    )
    print(f"Test observations: {len(test_data)}")

    print("\nFinal test results:")
    print(
        result_table.to_string(
            formatters={
                "mse": "{:.6f}".format,
                "mae": "{:.6f}".format,
                "mse_improvement_vs_persistence": "{:.2%}".format,
                "mae_improvement_vs_persistence": "{:.2%}".format,
            }
        )
    )

    print(
        "\nHAR + Lag 2 MSE improvement over standard HAR: "
        f"{relative_improvement(har_mse, lag2_mse):.2%}"
    )
    print(
        "HAR + Lag 2 MAE improvement over standard HAR: "
        f"{relative_improvement(har_mae, lag2_mae):.2%}"
    )

    selected_prediction = predictions["HAR + Lag 2"]
    selected_model_name = "HAR + Lag 2"
    selected_features = FINAL_MODELS[selected_model_name]

    saved_paths = save_model_artifacts(
        model=trained_models[selected_model_name],
        feature_names=selected_features,
        output_dir=OUTPUT_DIR,
        model_name=selected_model_name,
        target_name="next-trading-day log realized variance (target)",
        training_info={
            "dataset": "training + validation",
            "n_rows": int(len(development_data)),
            "feature_date_start": str(development_data["date"].min()),
            "feature_date_end": str(development_data["date"].max()),
            "target_date_start": str(development_data["target_date"].min()),
            "target_date_end": str(development_data["target_date"].max()),
        },
        evaluation_info={
            "dataset": "final test",
            "n_rows": int(len(test_data)),
            "target_date_start": str(test_data["target_date"].min()),
            "target_date_end": str(test_data["target_date"].max()),
            "mse": float(result_table.loc[selected_model_name, "mse"]),
            "mae": float(result_table.loc[selected_model_name, "mae"]),
            "mse_improvement_vs_persistence": float(
                result_table.loc[
                    selected_model_name,
                    "mse_improvement_vs_persistence",
                ]
            ),
            "mae_improvement_vs_persistence": float(
                result_table.loc[
                    selected_model_name,
                    "mae_improvement_vs_persistence",
                ]
            ),
            "mse_improvement_vs_har": float(
                relative_improvement(har_mse, lag2_mse)
            ),
            "mae_improvement_vs_har": float(
                relative_improvement(har_mae, lag2_mae)
            ),
        },
    )

    print("\nModel saved:")
    print(saved_paths["model"])
    print("Model metadata saved:")
    print(saved_paths["model_info"])

    run_ljung_box_tests(
        y_true=y_test,
        y_pred=selected_prediction,
        lags=(5, 10, 20, 30),
        model_name="HAR + Lag 2 (final test)",
    )

    plot_final_test_comparison(
        test_data=test_data,
        predictions=predictions,
        result_table=result_table,
    )

    plot_residual_diagnostics(
        y_true=y_test,
        y_pred=selected_prediction,
        lags=30,
        model_name="HAR + Lag 2",
        sample_name="Final test",
    )

    return result_table


if __name__ == "__main__":
    run_final_test()
