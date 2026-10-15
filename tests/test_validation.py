import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "src" / "volatility"
sys.path.insert(0, str(CODE_DIR))

from validation import chronological_split  # noqa: E402


class ChronologicalSplitTests(unittest.TestCase):
    def test_split_boundaries_do_not_overlap_future_targets(self):
        dates = pd.date_range(
            "2020-01-01",
            periods=100,
            freq="B",
            tz="UTC",
        )
        data = pd.DataFrame(
            {
                "date": dates[:-1],
                "target_date": dates[1:],
                "target": range(len(dates) - 1),
            }
        )

        train, validation, test = chronological_split(data)

        self.assertLess(
            train["target_date"].max(),
            validation["date"].min(),
        )
        self.assertLess(
            validation["target_date"].max(),
            test["date"].min(),
        )
        self.assertTrue(train["date"].is_monotonic_increasing)
        self.assertTrue(validation["date"].is_monotonic_increasing)
        self.assertTrue(test["date"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
