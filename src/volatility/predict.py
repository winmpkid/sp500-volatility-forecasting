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
            f"没有找到模型文件，请先运行 final_test.py：{MODEL_PATH}"
        )
    if not MODEL_INFO_PATH.exists():
        raise FileNotFoundError(
            f"没有找到模型信息文件：{MODEL_INFO_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    with MODEL_INFO_PATH.open("r", encoding="utf-8") as file:
        model_info = json.load(file)

    feature_names = model_info.get("feature_order")
    if not feature_names:
        raise ValueError("模型信息中缺少 feature_order。")

    if model.n_features_in_ != len(feature_names):
        raise ValueError("模型特征数量与模型信息不一致。")

    trained_feature_names = getattr(model, "feature_names_in_", None)
    if (
        trained_feature_names is not None
        and list(trained_feature_names) != feature_names
    ):
        raise ValueError("模型内部特征顺序与模型信息不一致。")

    return model, model_info, feature_names


def _save_prediction(result: pd.DataFrame) -> None:
    """保存预测历史；同一模型对同一特征日期重复运行时更新旧记录。"""
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
        raise ValueError("最新记录存在缺失特征，无法预测。")

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

    print("\n本次预测使用的特征：")
    print(X_latest.to_string(index=False))
    print("\n预测结果：")
    print(result.to_string(index=False))
    print(f"\n预测记录已保存：{PREDICTION_PATH}")

    return result


if __name__ == "__main__":
    run_prediction()
