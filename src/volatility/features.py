import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "log_rv_day",
    "log_rv_lag2",
    "log_rv_week",
    "log_rv_month",
]


def _prepare_feature_frame(
    raw_data: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """统一构造训练和预测都会使用的 HAR 特征。"""
    df = raw_data.loc[
        raw_data["Symbol"].eq(symbol),
        ["Unnamed: 0", "Symbol", "rv5"],
    ].copy()

    if df.empty:
        raise ValueError(f"没有找到指数代码：{symbol}")

    df = df.rename(columns={"Unnamed: 0": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].isna().any():
        raise ValueError("存在缺失日期，需要先检查。")
    if df["date"].duplicated().any():
        raise ValueError("同一指数存在重复日期，需要先检查。")

    rv = pd.to_numeric(df["rv5"], errors="coerce")
    valid = np.isfinite(rv) & rv.gt(0)
    print(f"{symbol} 原始记录数：{len(df)}")
    print(f"无法用于对数计算的方差记录数：{(~valid).sum()}")

    # 保留无效记录的时间位置，避免把更远的观测误接成相邻日期。
    rv = rv.where(valid)
    df["log_rv_day"] = np.log(rv)
    # 对下一日目标 RV_{t+1} 而言，RV_{t-1} 是第二日滞后。
    df["log_rv_lag2"] = df["log_rv_day"].shift(1)
    df["log_rv_week"] = np.log(
        rv.rolling(5, min_periods=5).mean()
    )
    df["log_rv_month"] = np.log(
        rv.rolling(22, min_periods=22).mean()
    )
    return df


def build_features(
    raw_data: pd.DataFrame,
    symbol: str = ".SPX",
) -> pd.DataFrame:
    df = _prepare_feature_frame(raw_data, symbol)
    df["target"] = df["log_rv_day"].shift(-1)
    df["target_date"] = df["date"].shift(-1)

    required = [
        *FEATURE_COLUMNS,
        "target",
        "target_date",
    ]
    model_df = df.dropna(subset=required).copy()
    print(f"可建模记录数：{len(model_df)}")
    return model_df


def build_prediction_features(
    raw_data: pd.DataFrame,
    symbol: str = ".SPX",
) -> pd.DataFrame:
    """构造下一交易日预测特征，不要求最新一行存在 target。"""
    df = _prepare_feature_frame(raw_data, symbol)
    prediction_df = df.dropna(subset=FEATURE_COLUMNS).copy()

    if prediction_df.empty:
        raise ValueError("没有足够的有效历史记录构造预测特征。")

    print(f"可用于预测的记录数：{len(prediction_df)}")
    return prediction_df
