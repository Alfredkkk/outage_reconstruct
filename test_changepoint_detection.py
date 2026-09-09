"""Focused compatibility tests for change-point evaluation."""

import unittest
import concurrent.futures as cf
import os

import pandas as pd

from utils.ChangePointDetection import ChangePointDetection
from utils.algorithms import _changepoint_executor_settings


class ChangePointDetectionTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows-specific executor behavior')
    def test_large_windows_input_uses_shared_memory_threads(self):
        executor_class, max_workers = _changepoint_executor_settings(1_000_000)

        self.assertIs(executor_class, cf.ThreadPoolExecutor)
        self.assertEqual(max_workers, 2)

    def test_evaluate_model_preserves_county_with_pandas_groupby(self):
        detector = ChangePointDetection.__new__(ChangePointDetection)
        timestamps = pd.to_datetime(
            [
                '2024-01-01 00:00:00',
                '2024-01-01 00:15:00',
                '2024-01-01 00:30:00',
            ],
            utc=True,
        )
        data = pd.DataFrame(
            {
                'county': ['Alpha'] * 3 + ['Beta'] * 3,
                'CustomersOut': [5, 0, 3, 4, 0, 2],
                'CustomersTracked': [100] * 6,
                'RecordDateTime': list(timestamps) * 2,
            }
        )

        results, changepoints = detector.evaluate_model(data, 'l2', 0.1)

        self.assertEqual(set(results['county']), {'Alpha', 'Beta'})
        self.assertEqual(set(changepoints.index), {'Alpha', 'Beta'})


if __name__ == '__main__':
    unittest.main()
