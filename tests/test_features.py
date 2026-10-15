import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "src" / "volatility"
sys.path.insert(0, str(CODE_DIR))

from features import (  # noqa: E402
    FEATURE_COLUMNS,
    build_features,
    build_prediction_features,
)


class FeatureConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.date_range(
            "2020-01-01",
            periods=30,
            freq="B",
            tz="UTC",
        )
        cls.raw_data = pd.DataFrame(
            {
                "Unnamed: 0": dates.astype(str),
                "Symbol": ".SPX",
                "rv5": np.linspace(0.0001, 0.0030, len(dates)),
            }
        )

    def test_prediction_features_keep_latest_valid_date(self):
        prediction_data = build_prediction_features(self.raw_data)

        self.assertEqual(
            prediction_data["date"].max(),
            pd.to_datetime(
                self.raw_data["Unnamed: 0"],
                utc=True,
            ).max(),
        )
        self.assertFalse(
            prediction_data[FEATURE_COLUMNS].isna().any().any()
        )

    def test_training_features_drop_row_without_future_target(self):
        model_data = build_features(self.raw_data)
        prediction_data = build_prediction_features(self.raw_data)

        self.assertEqual(len(prediction_data), len(model_data) + 1)
        self.assertLess(
            model_data["date"].max(),
            prediction_data["date"].max(),
        )

    def test_target_is_next_observation_log_rv(self):
        model_data = build_features(self.raw_data)
        row = model_data.iloc[-1]

        next_rv = self.raw_data.loc[
            pd.to_datetime(
                self.raw_data["Unnamed: 0"],
                utc=True,
            ).eq(row["target_date"]),
            "rv5",
        ].iloc[0]

        self.assertAlmostEqual(row["target"], np.log(next_rv))
        self.assertGreater(row["target_date"], row["date"])


if __name__ == "__main__":
    unittest.main()
