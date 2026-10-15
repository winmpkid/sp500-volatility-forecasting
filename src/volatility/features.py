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
    """Build the HAR features shared by training and prediction."""
    df = raw_data.loc[
        raw_data["Symbol"].eq(symbol),
        ["Unnamed: 0", "Symbol", "rv5"],
    ].copy()

    if df.empty:
        raise ValueError(f"Symbol not found: {symbol}")

    df = df.rename(columns={"Unnamed: 0": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].isna().any():
        raise ValueError("Missing dates detected; inspect the source data first.")
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates detected for the same symbol.")

    rv = pd.to_numeric(df["rv5"], errors="coerce")
    valid = np.isfinite(rv) & rv.gt(0)
    print(f"{symbol} raw observations: {len(df)}")
    print(f"Variance observations invalid for log transform: {(~valid).sum()}")

    # Preserve invalid rows in the timeline so distant observations are not
    # treated as adjacent trading days.
    rv = rv.where(valid)
    df["log_rv_day"] = np.log(rv)
    # Relative to the next-day target RV_{t+1}, RV_{t-1} is lag 2.
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
    print(f"Model-ready observations: {len(model_df)}")
    return model_df


def build_prediction_features(
    raw_data: pd.DataFrame,
    symbol: str = ".SPX",
) -> pd.DataFrame:
    """Build next-day features without requiring a target for the latest row."""
    df = _prepare_feature_frame(raw_data, symbol)
    prediction_df = df.dropna(subset=FEATURE_COLUMNS).copy()

    if prediction_df.empty:
        raise ValueError("Not enough valid history to build prediction features.")

    print(f"Prediction-ready observations: {len(prediction_df)}")
    return prediction_df
