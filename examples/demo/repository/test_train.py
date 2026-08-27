from __future__ import annotations

import unittest

from train import FEATURES, LABELS, accuracy, predict


class ThresholdClassifierTest(unittest.TestCase):
    def test_full_dataset_matches_paper_accuracy(self) -> None:
        self.assertAlmostEqual(accuracy(predict(FEATURES), LABELS), 0.9)


if __name__ == "__main__":
    unittest.main()
