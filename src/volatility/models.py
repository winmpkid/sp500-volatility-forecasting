import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.linear_model import LinearRegression


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def save_model_artifacts(
    model,
    feature_names,
    output_dir,
    model_name,
    target_name,
    training_info,
    evaluation_info,
):
    """保存可加载的模型，以及便于审阅和复现的模型信息。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "har_lag2_model.joblib"
    info_path = output_dir / "har_lag2_model_info.json"

    joblib.dump(model, model_path)

    model_info = {
        "schema_version": 1,
        "model_name": model_name,
        "model_type": type(model).__name__,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": target_name,
        "feature_order": list(feature_names),
        "intercept": float(model.intercept_),
        "coefficients": {
            feature: float(coefficient)
            for feature, coefficient in zip(feature_names, model.coef_)
        },
        "training": training_info,
        "evaluation": evaluation_info,
        "software": {
            "scikit_learn": sklearn.__version__,
        },
        "model_file": model_path.name,
    }

    with info_path.open("w", encoding="utf-8") as file:
        json.dump(model_info, file, ensure_ascii=False, indent=2)

    return {
        "model": model_path,
        "model_info": info_path,
    }
