"""Regression tests for reloading previously exported event tables."""

import unittest

import pandas as pd

from base_mvpipeline import BaseMVPipeline


class LoadedDataframeTransformTests(unittest.TestCase):
    def test_mixed_fractional_seconds_are_parsed_when_ground_truth_is_reloaded(self):
        pipeline = BaseMVPipeline.__new__(BaseMVPipeline)
        pipeline._ground_truth_df = pd.DataFrame(
            {
                'start_time': [
                    '2024-07-18 23:00:10.123456-04:00',
                    '2024-07-18 23:15:10-04:00',
                ],
                'end_time': [
                    '2024-07-18 23:15:10.123456-04:00',
                    '2024-07-18 23:30:10-04:00',
                ],
                'weighted_customers_out': [150.0, 120.0],
                'average_customers_out': [10.0, 8.0],
                'duration': [15.0, 15.0],
                'duration_timestamp_diff': [15.0, 15.0],
                'duration_mean': [15.0, 15.0],
                'outage_point': ['[(-84.1, 33.7)]', '[(-84.2, 33.8)]'],
            }
        )

        pipeline._transform_loaded_aggregated_df('ground_truth')

        self.assertFalse(pipeline._ground_truth_df['start_time'].isna().any())
        self.assertFalse(pipeline._ground_truth_df['end_time'].isna().any())
        self.assertEqual(
            str(pipeline._ground_truth_df['start_time'].dt.tz),
            'US/Eastern',
        )


if __name__ == '__main__':
    unittest.main()
