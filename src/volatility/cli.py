from features import build_features
from validation import chronological_split
from data import load_raw_data
from diagnostics import (
    describe_market,
    plot_return_diagnostics,
)

if __name__ == "__main__":
    df = load_raw_data()
    model_df = build_features(df)
    train_df, val_df, test_df = chronological_split(model_df)

    summary = describe_market(
        raw_data=df,
        end_date=train_df["date"].max(),
    )

    print("\nTraining-period market statistics:")
    print(summary.to_string())

    plot_return_diagnostics(
        raw_data=df,
        end_date=train_df["date"].max(),
        symbol=".SPX",
    )
