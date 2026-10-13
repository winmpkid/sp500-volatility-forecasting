import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


def describe_market(
    raw_data: pd.DataFrame,
    end_date,
    symbols=(".SPX", ".DJI", ".RUT"),
) -> pd.DataFrame:
    columns = [
        "Unnamed: 0",
        "Symbol",
        "open_to_close",
        "rv5",
    ]
    df = raw_data.loc[
        raw_data["Symbol"].isin(symbols),
        columns,
    ].copy()

    missing_symbols = set(symbols) - set(df["Symbol"].unique())
    if missing_symbols:
        raise ValueError(f"没有找到这些指数：{sorted(missing_symbols)}")

    df = df.rename(columns={"Unnamed: 0": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.loc[df["date"] <= end_date].copy()

    rows = []
    for symbol in symbols:
        part = df.loc[df["Symbol"].eq(symbol)]

        returns = pd.to_numeric(
            part["open_to_close"],
            errors="coerce",
        )

        returns = returns.replace([np.inf, -np.inf], np.nan)

        rv = pd.to_numeric(part["rv5"], errors="coerce")
        positive_rv = rv.where(np.isfinite(rv) & rv.gt(0))

        rows.append({
            "symbol": symbol,
            "n_rows": len(part),
            "n_returns": returns.count(),
            "mean_return": returns.mean(),
            "std_return": returns.std(),
            "skewness": returns.skew(),
            "excess_kurtosis": returns.kurt(),
            "mean_rv5": positive_rv.mean(),
            "n_invalid_rv5": positive_rv.isna().sum(),
        })

    return pd.DataFrame(rows).set_index("symbol")


def plot_return_diagnostics(
    raw_data: pd.DataFrame,
    end_date,
    symbol: str = ".SPX",
    lags: int = 30,
):
    df = raw_data.loc[
        raw_data["Symbol"].eq(symbol),
        ["Unnamed: 0", "open_to_close"],
    ].copy()

    if df.empty:
        raise ValueError(f"没有找到指数代码：{symbol}")

    df = df.rename(columns={"Unnamed: 0": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date")

    # 只分析训练期，避免利用验证集和测试集设计模型
    df = df.loc[df["date"] <= end_date].copy()

    returns = pd.to_numeric(
        df["open_to_close"],
        errors="coerce",
    )
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty:
        raise ValueError(f"{symbol} 没有可用的收益率数据")

    mean_return = returns.mean()
    std_return = returns.std()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
    )

    # 1. 收益率直方图与拟合的正态密度
    axes[0, 0].hist(
        returns,
        bins=60,
        density=True,
        alpha=0.7,
        edgecolor="black",
        label="Observed returns",
    )

    x = np.linspace(
        returns.min(),
        returns.max(),
        500,
    )

    normal_density = stats.norm.pdf(
        x,
        loc=mean_return,
        scale=std_return,
    )

    axes[0, 0].plot(
        x,
        normal_density,
        color="red",
        linewidth=2,
        label="Fitted normal",
    )

    axes[0, 0].set_title("Return distribution")
    axes[0, 0].set_xlabel("Open-to-close return")
    axes[0, 0].legend()

    # 2. 正态 Q-Q 图
    stats.probplot(
        returns,
        dist="norm",
        plot=axes[0, 1],
    )
    axes[0, 1].set_title("Normal Q-Q plot")

    # 3. 原始收益率的自相关
    plot_acf(
        returns,
        lags=lags,
        zero=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("ACF of returns")

    # 4. 平方收益率的自相关
    plot_acf(
        returns.pow(2),
        lags=lags,
        zero=False,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("ACF of squared returns")

    fig.suptitle(
        f"{symbol} training-period diagnostics",
        fontsize=14,
    )
    fig.tight_layout()
    plt.show()


def plot_residual_diagnostics(
    y_true,
    y_pred,
    lags: int = 30,
    model_name: str = "HAR",
    sample_name: str = "Validation",
):
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)

    if true_values.shape != predicted_values.shape:
        raise ValueError("真实值和预测值长度不一致")

    residuals = pd.Series(true_values - predicted_values).dropna()
    if residuals.empty:
        raise ValueError("没有可用于分析的残差")

    max_lags = min(lags, len(residuals) // 2 - 1)
    if max_lags < 1:
        raise ValueError("残差数量不足，无法计算 ACF")

    print(f"\n{model_name} {sample_name} 残差统计：")
    print(f"样本数量：{len(residuals)}")
    print(f"残差均值：{residuals.mean():.6f}")
    print(f"残差标准差：{residuals.std():.6f}")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
    )

    axes[0, 0].plot(
        residuals.to_numpy(),
        linewidth=1,
    )
    axes[0, 0].axhline(
        0,
        color="red",
        linestyle="--",
    )
    axes[0, 0].set_title(f"{sample_name} residuals")
    axes[0, 0].set_xlabel(f"{sample_name} observation")
    axes[0, 0].set_ylabel("Actual - predicted")

    stats.probplot(
        residuals,
        dist="norm",
        plot=axes[0, 1],
    )
    axes[0, 1].set_title("Residual Q-Q plot")

    plot_acf(
        residuals,
        lags=max_lags,
        zero=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("ACF of residuals")

    plot_acf(
        residuals.pow(2),
        lags=max_lags,
        zero=False,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("ACF of squared residuals")

    fig.suptitle(
        f"{model_name} {sample_name.lower()} residual diagnostics",
        fontsize=14,
    )
    fig.tight_layout()
    plt.show()


def plot_final_test_comparison(
    test_data: pd.DataFrame,
    predictions: dict,
    result_table: pd.DataFrame,
):
    """展示最终测试期预测轨迹以及各模型的误差指标。"""
    required_columns = {"target_date", "target"}
    missing_columns = required_columns - set(test_data.columns)
    if missing_columns:
        raise ValueError(
            f"测试数据缺少列：{sorted(missing_columns)}"
        )

    if not predictions:
        raise ValueError("至少需要提供一个模型的预测值")

    n_observations = len(test_data)
    for model_name, values in predictions.items():
        if len(values) != n_observations:
            raise ValueError(
                f"{model_name} 的预测长度与测试集不一致"
            )

    metric_columns = ["mse", "mae"]
    missing_metrics = set(metric_columns) - set(result_table.columns)
    if missing_metrics:
        raise ValueError(
            f"结果表缺少指标：{sorted(missing_metrics)}"
        )

    dates = pd.to_datetime(test_data["target_date"])
    actual = test_data["target"].to_numpy()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    axes[0].plot(
        dates,
        actual,
        color="black",
        linewidth=1.4,
        label="Actual",
    )
    for model_name, values in predictions.items():
        axes[0].plot(
            dates,
            np.asarray(values),
            linewidth=1,
            alpha=0.8,
            label=model_name,
        )

    axes[0].set_title("Final test: actual vs predicted log RV")
    axes[0].set_xlabel("Target date")
    axes[0].set_ylabel("Log realized volatility")
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.2)

    positions = np.arange(len(result_table))
    width = 0.36
    mse_bars = axes[1].bar(
        positions - width / 2,
        result_table["mse"],
        width,
        label="MSE",
    )
    mae_bars = axes[1].bar(
        positions + width / 2,
        result_table["mae"],
        width,
        label="MAE",
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(result_table.index)
    axes[1].set_title("Final test error comparison (lower is better)")
    axes[1].set_ylabel("Error")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)

    for bars in (mse_bars, mae_bars):
        for bar in bars:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Frozen-model final test evaluation", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


def run_ljung_box_tests(
    y_true,
    y_pred,
    lags=(5, 10, 20, 30),
    model_name: str = "HAR",
):
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)

    if true_values.shape != predicted_values.shape:
        raise ValueError("真实值和预测值长度不一致")

    residuals = pd.Series(true_values - predicted_values).dropna()
    if residuals.empty:
        raise ValueError("没有可用于 Ljung-Box 检验的残差")

    valid_lags = [
        lag
        for lag in lags
        if 0 < lag < len(residuals)
    ]
    if not valid_lags:
        raise ValueError("没有适合当前残差长度的 Ljung-Box 滞后期")

    residual_test = acorr_ljungbox(
        residuals,
        lags=valid_lags,
        return_df=True,
    )
    squared_residual_test = acorr_ljungbox(
        residuals.pow(2),
        lags=valid_lags,
        return_df=True,
    )

    print(f"\n{model_name} 残差 Ljung-Box 检验：")
    print(
        residual_test.to_string(
            float_format=lambda value: f"{value:.6f}"
        )
    )

    print(f"\n{model_name} 平方残差 Ljung-Box 检验：")
    print(
        squared_residual_test.to_string(
            float_format=lambda value: f"{value:.6f}"
        )
    )

    return residual_test, squared_residual_test


def plot_lag_feature_evidence(
    y_true,
    base_prediction,
    candidate_prediction,
    cv_results: pd.DataFrame,
    base_model_name: str,
    candidate_model_name: str,
    lags: int = 30,
    ljung_box_lags=(5, 10, 20, 30),
):
    true_values = np.asarray(y_true, dtype=float)
    base_values = np.asarray(base_prediction, dtype=float)
    candidate_values = np.asarray(candidate_prediction, dtype=float)

    if not (
        true_values.shape
        == base_values.shape
        == candidate_values.shape
    ):
        raise ValueError("真实值和两个模型的预测值长度必须一致")

    base_residuals = pd.Series(true_values - base_values).dropna()
    candidate_residuals = pd.Series(
        true_values - candidate_values
    ).dropna()

    max_lags = min(
        lags,
        len(base_residuals) // 2 - 1,
        len(candidate_residuals) // 2 - 1,
    )
    if max_lags < 1:
        raise ValueError("残差数量不足，无法生成 Lag 证据图")

    fold_mse = (
        cv_results.loc[
            cv_results["model"].isin(
                [base_model_name, candidate_model_name]
            ),
            ["fold", "model", "mse"],
        ]
        .pivot(index="fold", columns="model", values="mse")
        .dropna()
    )

    required_models = {base_model_name, candidate_model_name}
    if not required_models.issubset(fold_mse.columns):
        raise ValueError("交叉验证结果缺少需要比较的模型")

    fold_improvement = (
        fold_mse[base_model_name]
        - fold_mse[candidate_model_name]
    ) / fold_mse[base_model_name]

    valid_lags = [
        lag
        for lag in ljung_box_lags
        if 0 < lag < len(base_residuals)
        and lag < len(candidate_residuals)
    ]
    if not valid_lags:
        raise ValueError("没有适合当前残差长度的 Ljung-Box 滞后期")

    tests = {
        f"{base_model_name} residual": acorr_ljungbox(
            base_residuals,
            lags=valid_lags,
            return_df=True,
        ),
        f"{candidate_model_name} residual": acorr_ljungbox(
            candidate_residuals,
            lags=valid_lags,
            return_df=True,
        ),
        f"{base_model_name} squared": acorr_ljungbox(
            base_residuals.pow(2),
            lags=valid_lags,
            return_df=True,
        ),
        f"{candidate_model_name} squared": acorr_ljungbox(
            candidate_residuals.pow(2),
            lags=valid_lags,
            return_df=True,
        ),
    }

    evidence_table = fold_mse.copy()
    evidence_table["candidate_improvement"] = fold_improvement
    print("\nLag 特征逐折证据：")
    print(
        evidence_table.to_string(
            formatters={
                base_model_name: "{:.6f}".format,
                candidate_model_name: "{:.6f}".format,
                "candidate_improvement": "{:.2%}".format,
            }
        )
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 14),
    )

    plot_acf(
        base_residuals,
        lags=max_lags,
        zero=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title(f"{base_model_name}: residual ACF")

    plot_acf(
        candidate_residuals,
        lags=max_lags,
        zero=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title(
        f"{candidate_model_name}: residual ACF"
    )

    plot_pacf(
        base_residuals,
        lags=max_lags,
        zero=False,
        method="ywm",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title(f"{base_model_name}: residual PACF")

    plot_pacf(
        candidate_residuals,
        lags=max_lags,
        zero=False,
        method="ywm",
        ax=axes[1, 1],
    )
    axes[1, 1].set_title(
        f"{candidate_model_name}: residual PACF"
    )

    colors = [
        "tab:green" if value > 0 else "tab:red"
        for value in fold_improvement
    ]
    bars = axes[2, 0].bar(
        fold_improvement.index.astype(str),
        fold_improvement,
        color=colors,
    )
    axes[2, 0].axhline(0, color="black", linewidth=1)
    axes[2, 0].set_title("Candidate MSE improvement by CV fold")
    axes[2, 0].set_xlabel("Fold")
    axes[2, 0].set_ylabel("Relative MSE improvement")
    for bar, value in zip(bars, fold_improvement):
        axes[2, 0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1%}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )

    for label, test in tests.items():
        pvalues = np.clip(
            test["lb_pvalue"].to_numpy(),
            np.finfo(float).tiny,
            1.0,
        )
        linestyle = "--" if "squared" in label else "-"
        axes[2, 1].plot(
            valid_lags,
            pvalues,
            marker="o",
            linestyle=linestyle,
            label=label,
        )

    axes[2, 1].axhline(
        0.05,
        color="red",
        linestyle=":",
        label="5% significance level",
    )
    axes[2, 1].set_yscale("log")
    axes[2, 1].set_title("Ljung-Box p-value comparison")
    axes[2, 1].set_xlabel("Maximum lag")
    axes[2, 1].set_ylabel("p-value (log scale)")
    axes[2, 1].legend(fontsize=8)

    fig.suptitle(
        "Lag feature evidence chain",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()

    return evidence_table, tests
