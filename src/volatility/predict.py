import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data import load_raw_data
from features import build_prediction_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "har_lag2_model.joblib"
MODEL_INFO_PATH = OUTPUT_DIR / "har_lag2_model_info.json"
PREDICTION_PATH = OUTPUT_DIR / "predictions.csv"


def _load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found; run final_test.py first: {MODEL_PATH}"
        )
    if not MODEL_INFO_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata file not found: {MODEL_INFO_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    with MODEL_INFO_PATH.open("r", encoding="utf-8") as file:
        model_info = json.load(file)

    feature_names = model_info.get("feature_order")
    if not feature_names:
        raise ValueError("Model metadata is missing feature_order.")

    if model.n_features_in_ != len(feature_names):
        raise ValueError("Model feature count does not match the metadata.")

    trained_feature_names = getattr(model, "feature_names_in_", None)
    if (
        trained_feature_names is not None
        and list(trained_feature_names) != feature_names
    ):
        raise ValueError("Model feature order does not match the metadata.")

    return model, model_info, feature_names


def _save_prediction(result: pd.DataFrame) -> None:
    """Save history and update duplicate runs for the same model and date."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if PREDICTION_PATH.exists():
        history = pd.read_csv(PREDICTION_PATH)
        history = pd.concat([history, result], ignore_index=True)
        history = history.drop_duplicates(
            subset=["feature_date", "model_saved_at_utc"],
            keep="last",
        )
    else:
        history = result

    history = history.sort_values("feature_date").reset_index(drop=True)
    history.to_csv(PREDICTION_PATH, index=False)


def run_prediction(symbol: str = ".SPX") -> pd.DataFrame:
    model, model_info, feature_names = _load_model_bundle()

    raw_data = load_raw_data()
    feature_data = build_prediction_features(raw_data, symbol=symbol)

    latest_row = feature_data.iloc[[-1]]
    X_latest = latest_row.loc[:, feature_names]

    if X_latest.isna().any().any():
        raise ValueError("The latest observation has missing prediction features.")

    predicted_log_rv = float(model.predict(X_latest)[0])
    predicted_rv = float(np.exp(predicted_log_rv))
    predicted_volatility = float(np.sqrt(predicted_rv))

    result_values = {
        "feature_date": latest_row["date"].iloc[0].isoformat(),
        "forecast_horizon": "next trading day",
        "symbol": symbol,
        "model_name": model_info["model_name"],
        "model_saved_at_utc": model_info["saved_at_utc"],
        "prediction_created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "predicted_log_rv": predicted_log_rv,
        "predicted_rv": predicted_rv,
        "predicted_volatility": predicted_volatility,
    }
    result_values.update(
        {
            feature: float(X_latest.iloc[0][feature])
            for feature in feature_names
        }
    )
    result = pd.DataFrame([result_values])

    _save_prediction(result)

    print("\nFeatures used for this prediction:")
    print(X_latest.to_string(index=False))
    print("\nPrediction:")
    print(result.to_string(index=False))
    print(f"\nPrediction history saved: {PREDICTION_PATH}")

    return result


if __name__ == "__main__":
    run_prediction()
