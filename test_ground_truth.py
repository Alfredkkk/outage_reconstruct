"""Focused regression tests for validated ground-truth construction."""

import unittest

import pandas as pd

from utils.ground_truth import (
    GroundTruthSettings,
    build_ground_truth_events,
    prepare_outage_snapshots,
    validate_ground_truth_events,
)


class GroundTruthBuilderTests(unittest.TestCase):
    def _build(self, raw, event_id_columns=('event_id',), **kwargs):
        prepared, report = prepare_outage_snapshots(
            raw,
            event_id_columns=event_id_columns,
            start_column='start_time',
            customers_column='customers',
            **kwargs,
        )
        return prepared, build_ground_truth_events(prepared), report

    def test_unsorted_snapshots_produce_bounded_customer_metrics(self):
        raw = pd.DataFrame(
            {
                'event_id': ['A', 'A', 'A'],
                'timestamp': [
                    '2024-01-01 00:30:00',
                    '2024-01-01 00:00:00',
                    '2024-01-01 00:15:00',
                ],
                'start_time': ['2024-01-01 00:00:00'] * 3,
                'customers': [30, 10, 20],
            }
        )

        _, events, _ = self._build(raw)
        event = events.iloc[0]

        self.assertEqual(event['snapshot_count'], 3)
        self.assertAlmostEqual(event['average_customers_out'], 22.0)
        self.assertAlmostEqual(event['weighted_customers_out'], 825.0)
        self.assertGreaterEqual(
            event['average_customers_out'], event['source_customer_min']
        )
        self.assertLessEqual(
            event['average_customers_out'], event['source_customer_max']
        )
        self.assertGreaterEqual(event['weighted_customers_out'], 0)

    def test_sentinel_start_falls_back_and_long_gap_splits_episode(self):
        raw = pd.DataFrame(
            {
                'event_id': ['A', 'A'],
                'timestamp': ['2024-01-01 00:00:00', '2024-01-01 01:00:00'],
                'start_time': ['1899-01-01 00:00:00'] * 2,
                'customers': [4, 3],
            }
        )

        _, events, _ = self._build(raw)

        self.assertEqual(len(events), 2)
        self.assertTrue(events['source_start_inferred'].all())
        self.assertTrue(events['duration'].eq(7.5).all())
        self.assertLess(events['duration'].max(), 24 * 60)

    def test_reused_composite_id_is_split_by_source_start(self):
        raw = pd.DataFrame(
            {
                'utility': ['U', 'U', 'U', 'U'],
                'event_id': ['11', '11', '11', '11'],
                'timestamp': [
                    '2023-06-01 10:00:00',
                    '2023-06-01 10:15:00',
                    '2024-06-01 10:00:00',
                    '2024-06-01 10:15:00',
                ],
                'start_time': [
                    '2023-06-01 09:55:00',
                    '2023-06-01 09:55:00',
                    '2024-06-01 09:55:00',
                    '2024-06-01 09:55:00',
                ],
                'customers': [10, 8, 7, 5],
            }
        )

        _, events, _ = self._build(
            raw, event_id_columns=('utility', 'event_id')
        )

        self.assertEqual(len(events), 2)
        self.assertTrue(events['source_event_id_reused'].all())
        self.assertTrue(events['episode_id'].is_unique)

    def test_small_start_time_corrections_remain_one_episode(self):
        raw = pd.DataFrame(
            {
                'event_id': ['A', 'A'],
                'timestamp': ['2024-01-01 00:15:00', '2024-01-01 00:30:00'],
                'start_time': ['2024-01-01 00:00:00', '2024-01-01 00:10:00'],
                'customers': [10, 8],
            }
        )

        _, events, _ = self._build(raw)
        self.assertEqual(len(events), 1)

    def test_derived_location_identity_handles_moving_california_start(self):
        raw = pd.DataFrame(
            {
                'event_id': ['LAWP::location'] * 3,
                'timestamp': [
                    '2023-08-06 03:42:02',
                    '2023-08-06 03:57:02',
                    '2023-08-06 04:12:02',
                ],
                'start_time': [
                    '2023-08-06 03:30:08',
                    '2023-08-06 03:45:10',
                    '2023-08-06 04:00:09',
                ],
                'customers': [11, 14, 8],
            }
        )

        prepared, events, _ = self._build(
            raw,
            identity_is_derived=True,
            settings=GroundTruthSettings(start_time_merge_tolerance_minutes=30),
        )
        all_events, _, _, _ = validate_ground_truth_events(events)

        self.assertEqual(prepared['_gt_episode_id'].nunique(), 1)
        self.assertEqual(len(all_events), 1)
        self.assertTrue(all_events.iloc[0]['identity_is_derived'])
        self.assertIn('derived_event_identity', all_events.iloc[0]['quality_flags'])

    def test_conflicting_same_timestamp_snapshot_keeps_last_and_flags(self):
        raw = pd.DataFrame(
            {
                'event_id': ['A', 'A'],
                'timestamp': ['2024-01-01 00:15:00'] * 2,
                'start_time': ['2024-01-01 00:00:00'] * 2,
                'customers': [10, 20],
            }
        )

        _, events, _ = self._build(raw)
        event = events.iloc[0]

        self.assertEqual(event['snapshot_count'], 1)
        self.assertEqual(event['average_customers_out'], 20)
        self.assertTrue(event['has_conflicting_snapshots'])

    def test_source_duration_is_an_elapsed_lower_bound(self):
        raw = pd.DataFrame(
            {
                'event_id': ['A', 'A'],
                'timestamp': ['2024-01-01 00:15:00', '2024-01-01 00:30:00'],
                'start_time': ['2024-01-01 00:00:00'] * 2,
                'customers': [10, 8],
                'reported_duration': [15, 30],
            }
        )

        _, events, _ = self._build(
            raw, source_duration_column='reported_duration'
        )
        self.assertEqual(events.iloc[0]['duration_lower_bound'], 30)
        self.assertEqual(events.iloc[0]['duration'], 37.5)

    def test_mixed_datetime_resolutions_select_source_start_losslessly(self):
        raw = pd.DataFrame(
            {
                'event_id': ['A', 'A'],
                'timestamp': [
                    '2024-02-23 10:00:00.123456',
                    '2024-02-23 10:15:00.123456',
                ],
                'start_time': [
                    '2024-02-23 09:59:59.555000064',
                    '2024-02-23 09:59:59.555000064',
                ],
                'customers': [10, 8],
            }
        )

        _, events, _ = self._build(raw)

        self.assertEqual(len(events), 1)
        self.assertEqual(
            events.iloc[0]['start_time'].nanosecond,
            64,
        )

    def test_validation_excludes_mathematically_invalid_event(self):
        raw = pd.DataFrame(
            {
                'event_id': ['A', 'A'],
                'timestamp': ['2024-01-01 00:00:00', '2024-01-01 00:15:00'],
                'start_time': ['2024-01-01 00:00:00'] * 2,
                'customers': [5, 10],
            }
        )
        _, events, _ = self._build(raw)
        events.loc[0, 'average_customers_out'] = 100

        all_events, valid, flagged, report = validate_ground_truth_events(events)

        self.assertFalse(all_events.iloc[0]['is_valid'])
        self.assertIn(
            'customers_above_source_max', all_events.iloc[0]['hard_quality_flags']
        )
        self.assertTrue(valid.empty)
        self.assertEqual(len(flagged), 1)
        self.assertFalse(report.empty)


if __name__ == '__main__':
    unittest.main()
