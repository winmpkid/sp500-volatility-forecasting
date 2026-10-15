import pandas as pd

def chronological_split(
    data: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    df = data.sort_values("date").copy()
    dates = pd.Index(df["date"].drop_duplicates())
    n_dates = len(dates)

    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))

    if not (0 < train_end < val_end < n_dates):
        raise ValueError("Not enough dates to create three non-empty periods.")

    val_start = dates[train_end]
    test_start = dates[val_end]

    # Training set: both feature and target dates precede validation.
    train_df = df.loc[
        (df["date"] < val_start)
        & (df["target_date"] < val_start)
    ].copy()

    # Validation set: feature dates are in validation and targets stay
    # outside the test period.
    val_df = df.loc[
        (df["date"] >= val_start)
        & (df["date"] < test_start)
        & (df["target_date"] < test_start)
    ].copy()

    # Test set: the final chronological period.
    test_df = df.loc[
        df["date"] >= test_start
    ].copy()

    if any(part.empty for part in (train_df, val_df, test_df)):
        raise ValueError("A split is empty after boundary checks; adjust the ratios.")

    return train_df, val_df, test_df
