import pandas as pd
from data import load_raw_data
from diagnostics import (
    plot_lag_feature_evidence,
    plot_residual_diagnostics,
    run_ljung_box_tests,
)
from evaluation import evaluate_model
from features import build_features
from models import train_model
from validation import chronological_split
from sklearn.model_selection import TimeSeriesSplit


FEATURE_SETS = {
    "Day": [
        "log_rv_day",
    ],
    "Day + Week": [
        "log_rv_day",
        "log_rv_week",
    ],
    "Day + Week + Month": [
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



def relative_improvement(baseline_value, model_value):
    return (baseline_value - model_value) / baseline_value

def evaluate_feature_set(
    train_data,
    validation_data,
    features,
):
    X_train = train_data[features]
    y_train = train_data["target"]

    X_validation = validation_data[features]
    y_validation = validation_data["target"]

    model = train_model(X_train, y_train)
    prediction = model.predict(X_validation)

    return evaluate_model(y_validation, prediction)

def run_time_series_cv(
    development_data,
    n_splits=5,
):
    data = (
        development_data
        .sort_values("date")
        .reset_index(drop=True)
    )

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        gap=1,
    )

    rows = []

    for fold, (train_index, val_index) in enumerate(
        splitter.split(data),
        start=1,
    ):
        fold_train = data.iloc[train_index]
        fold_validation = data.iloc[val_index]

        y_validation = fold_validation["target"]

        # 每一折都重新计算持续性基准
        persistence_prediction = fold_validation[
            "log_rv_day"
        ].to_numpy()

        persistence_results = evaluate_model(
            y_validation,
            persistence_prediction,
        )

        rows.append({
            "fold": fold,
            "model": "Persistence",
            "mse": persistence_results["mse"],
            "mae": persistence_results["mae"],
        })

        # 每一折都重新训练全部特征组合
        for model_name, features in FEATURE_SETS.items():
            results = evaluate_feature_set(
                train_data=fold_train,
                validation_data=fold_validation,
                features=features,
            )

            rows.append({
                "fold": fold,
                "model": model_name,
                "mse": results["mse"],
                "mae": results["mae"],
            })

    cv_results = pd.DataFrame(rows)

    # 找到每一折对应的持续性基准误差
    fold_baseline = (
        cv_results.loc[
            cv_results["model"].eq("Persistence"),
            ["fold", "mse", "mae"],
        ]
        .rename(columns={
            "mse": "baseline_mse",
            "mae": "baseline_mae",
        })
    )

    cv_results = cv_results.merge(
        fold_baseline,
        on="fold",
        how="left",
    )

    # 每个模型在每一折相对基准改善多少
    cv_results["mse_improvement"] = (
        cv_results["baseline_mse"]
        - cv_results["mse"]
    ) / cv_results["baseline_mse"]

    cv_results["mae_improvement"] = (
        cv_results["baseline_mae"]
        - cv_results["mae"]
    ) / cv_results["baseline_mae"]

    return cv_results

def run_benchmark():
    raw_data = load_raw_data()
    model_data = build_features(raw_data)

    train_data, validation_data, _test_data = chronological_split(model_data)

    development_data = pd.concat(
        [
            train_data,
            validation_data,
        ],
        ignore_index=True,
    )
    cv_results = run_time_series_cv(
        development_data=development_data,
        n_splits=5,
    )

    print("\n各折 MSE：")
    print(
        cv_results.pivot(
            index="fold",
            columns="model",
            values="mse",
        ).to_string()
    )

    cv_summary = (
        cv_results
        .groupby("model")
        .agg(
            mean_mse=("mse", "mean"),
            std_mse=("mse", "std"),
            mean_mae=("mae", "mean"),
            mean_mse_improvement=("mse_improvement", "mean"),
        )
    )

    print("\n时间序列交叉验证汇总：")
    print(
        cv_summary.to_string(
            formatters={
                "mean_mse": "{:.6f}".format,
                "std_mse": "{:.6f}".format,
                "mean_mae": "{:.6f}".format,
                "mean_mse_improvement": "{:.2%}".format,
            }
        )
    )

    y_validation = validation_data["target"]

    persistence_prediction = validation_data["log_rv_day"].to_numpy()
    persistence_results = evaluate_model(
        y_validation,
        persistence_prediction,
    )

    rows = [
        {
            "model": "Persistence",
            "mse": persistence_results["mse"],
            "mae": persistence_results["mae"],
        }
    ]

    for model_name, features in FEATURE_SETS.items():
        results = evaluate_feature_set(
            train_data,
            validation_data,
            features,
        )
        rows.append({
            "model": model_name,
            "mse": results["mse"],
            "mae": results["mae"],
        })

    result_table = pd.DataFrame(rows).set_index("model")

    baseline_mse = result_table.loc["Persistence", "mse"]
    baseline_mae = result_table.loc["Persistence", "mae"]

    result_table["mse_improvement_vs_baseline"] = (
        baseline_mse - result_table["mse"]
    ) / baseline_mse

    result_table["mae_improvement_vs_baseline"] = (
        baseline_mae - result_table["mae"]
    ) / baseline_mae

    print("\n模型消融结果：")
    print(
        result_table.to_string(
            formatters={
                "mse": "{:.6f}".format,
                "mae": "{:.6f}".format,
                "mse_improvement_vs_baseline": "{:.2%}".format,
                "mae_improvement_vs_baseline": "{:.2%}".format,
            }
        )
    )

    week_improvement = relative_improvement(
        result_table.loc["Day", "mse"],
        result_table.loc["Day + Week", "mse"],
    )

    month_improvement = relative_improvement(
        result_table.loc["Day + Week", "mse"],
        result_table.loc["Day + Week + Month", "mse"],
    )

    lag2_improvement = relative_improvement(
        result_table.loc["Day + Week + Month", "mse"],
        result_table.loc["HAR + Lag 2", "mse"],
    )

    print(
        f"\n加入 Week 后的 MSE 改进："
        f"{week_improvement:.2%}"
    )
    print(
        f"加入 Month 后的 MSE 改进："
        f"{month_improvement:.2%}"
    )
    print(
        f"加入 Lag 2 后的 MSE 改进："
        f"{lag2_improvement:.2%}"
    )

    base_model_name = "Day + Week + Month"
    candidate_model_name = "HAR + Lag 2"

    base_features = FEATURE_SETS[base_model_name]
    candidate_features = FEATURE_SETS[candidate_model_name]

    X_train = train_data[base_features]
    y_train = train_data["target"]
    X_validation = validation_data[base_features]

    base_model = train_model(X_train, y_train)
    base_prediction = base_model.predict(X_validation)

    candidate_model = train_model(
        train_data[candidate_features],
        y_train,
    )
    candidate_prediction = candidate_model.predict(
        validation_data[candidate_features]
    )

    run_ljung_box_tests(
        y_true=y_validation,
        y_pred=base_prediction,
        lags=(5, 10, 20, 30),
        model_name=base_model_name,
    )

    run_ljung_box_tests(
        y_true=y_validation,
        y_pred=candidate_prediction,
        lags=(5, 10, 20, 30),
        model_name=candidate_model_name,
    )

    plot_lag_feature_evidence(
        y_true=y_validation,
        base_prediction=base_prediction,
        candidate_prediction=candidate_prediction,
        cv_results=cv_results,
        base_model_name=base_model_name,
        candidate_model_name=candidate_model_name,
        lags=30,
        ljung_box_lags=(5, 10, 20, 30),
    )

    plot_residual_diagnostics(
        y_true=y_validation,
        y_pred=candidate_prediction,
        lags=30,
        model_name=candidate_model_name,
    )

if __name__ == "__main__":
    run_benchmark()
