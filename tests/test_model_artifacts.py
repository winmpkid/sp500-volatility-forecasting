import json
import unittest
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "har_lag2_model.joblib"
INFO_PATH = OUTPUT_DIR / "har_lag2_model_info.json"


class ModelArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists() or not INFO_PATH.exists():
            raise unittest.SkipTest(
                "请先运行 final_test.py 生成模型文件。"
            )

        cls.model = joblib.load(MODEL_PATH)
        cls.model_info = json.loads(
            INFO_PATH.read_text(encoding="utf-8")
        )

    def test_feature_order_matches_saved_model(self):
        feature_order = self.model_info["feature_order"]

        self.assertEqual(self.model.n_features_in_, len(feature_order))
        if hasattr(self.model, "feature_names_in_"):
            self.assertEqual(
                list(self.model.feature_names_in_),
                feature_order,
            )

    def test_coefficients_cover_every_feature(self):
        self.assertEqual(
            list(self.model_info["coefficients"]),
            self.model_info["feature_order"],
        )

    def test_saved_model_can_predict(self):
        feature_order = self.model_info["feature_order"]
        sample = pd.DataFrame(
            [[0.0] * len(feature_order)],
            columns=feature_order,
        )
        prediction = self.model.predict(sample)

        self.assertEqual(prediction.shape, (1,))


if __name__ == "__main__":
    unittest.main()
