"""Shared preparation, aggregation, and validation for outage reference events.

The raw ``per_outage`` files are snapshots of outage-map records.  They are not
already event-level ground truth.  This module turns those snapshots into
auditable reference-event episodes without bridging unobserved collection gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


GROUND_TRUTH_VERSION = "2.0"


@dataclass(frozen=True)
class GroundTruthSettings:
    """Configuration for constructing reference events from outage snapshots."""

    expected_snapshot_interval_minutes: float = 15.0
    max_observation_gap_minutes: float = 31.0
    start_time_merge_tolerance_minutes: float = 16.0
    minimum_valid_year: int = 2000


def coerce_datetime_series(values: pd.Series, timezone: str = "US/Eastern") -> pd.Series:
    """Parse mixed datetime strings as UTC and convert them to ``timezone``.

    ``format='mixed'`` is fast on newer pandas versions.  The fallback preserves
    compatibility with the older pandas version originally pinned by this repo.
    Invalid values become ``NaT`` so they can be audited instead of crashing an
    entire utility run.
    """

    try:
        parsed = pd.to_datetime(values, format="mixed", errors="coerce", utc=True)
    except (TypeError, ValueError):
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert(timezone)


def _report_row(stage: str, issue: str, count: int, action: str) -> dict:
    return {"stage": stage, "issue": issue, "count": int(count), "action": action}


def _normalise_event_id(frame: pd.DataFrame, columns: Sequence[str]) -> Tuple[pd.Series, pd.Series]:
    if not columns:
        raise ValueError("At least one event-id column is required")

    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Missing event-id columns: {missing_columns}")

    components = frame[list(columns)].astype("string")
    for column in components.columns:
        components[column] = components[column].str.strip()

    missing = components.isna().any(axis=1)
    missing |= components.eq("").any(axis=1)
    event_id = components.fillna("").agg("::".join, axis=1)
    return event_id, missing


def _cluster_source_start_times(
    frame: pd.DataFrame,
    tolerance_minutes: float,
) -> pd.DataFrame:
    """Map nearby corrections of one start time to a stable canonical value.

    A change smaller than the nominal snapshot interval is treated as a source
    correction.  A change greater than the interval begins a new
    episode.  This separates IDs reused every polling cycle or across years while
    keeping small minute-level corrections together.
    """

    valid_pairs = (
        frame.loc[
            frame["_gt_source_start_valid"],
            ["_gt_raw_event_id", "_gt_source_start_time"],
        ]
        .drop_duplicates()
        .sort_values(["_gt_raw_event_id", "_gt_source_start_time"])
    )

    if valid_pairs.empty:
        frame["_gt_canonical_start_time"] = pd.Series(
            pd.NaT,
            index=frame.index,
            dtype=frame["_gt_source_start_time"].dtype,
        )
        frame["_gt_source_event_id_reused"] = False
        return frame

    start_delta = (
        valid_pairs.groupby("_gt_raw_event_id")["_gt_source_start_time"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
    )
    valid_pairs["_gt_new_start_cluster"] = start_delta.isna() | start_delta.gt(
        tolerance_minutes
    )
    valid_pairs["_gt_start_cluster"] = (
        valid_pairs.groupby("_gt_raw_event_id")["_gt_new_start_cluster"].cumsum() - 1
    )
    valid_pairs["_gt_canonical_start_time"] = valid_pairs.groupby(
        ["_gt_raw_event_id", "_gt_start_cluster"]
    )["_gt_source_start_time"].transform("min")

    start_lookup = valid_pairs[
        ["_gt_raw_event_id", "_gt_source_start_time", "_gt_canonical_start_time"]
    ]
    frame = frame.merge(
        start_lookup,
        on=["_gt_raw_event_id", "_gt_source_start_time"],
        how="left",
        sort=False,
    )

    start_counts = (
        valid_pairs.groupby("_gt_raw_event_id")["_gt_canonical_start_time"]
        .nunique()
    )
    frame["_gt_source_event_id_reused"] = (
        frame["_gt_raw_event_id"].map(start_counts).fillna(0).gt(1)
    )

    single_start = start_counts[start_counts.eq(1)].index
    single_start_lookup = (
        valid_pairs[valid_pairs["_gt_raw_event_id"].isin(single_start)]
        .drop_duplicates("_gt_raw_event_id")
        .set_index("_gt_raw_event_id")["_gt_canonical_start_time"]
    )
    can_impute = frame["_gt_canonical_start_time"].isna() & frame[
        "_gt_raw_event_id"
    ].isin(single_start)
    # Some pandas versions attempt to cast an empty timezone-aware mapper to
    # float when Series.map() is called with an empty selection.  Skip the
    # operation when there is nothing to impute and use a plain dictionary to
    # keep mapping dtype inference stable across supported pandas versions.
    if can_impute.any():
        frame.loc[can_impute, "_gt_canonical_start_time"] = frame.loc[
            can_impute, "_gt_raw_event_id"
        ].map(single_start_lookup.to_dict())

    return frame


def prepare_outage_snapshots(
    dataframe: pd.DataFrame,
    event_id_columns: Sequence[str],
    start_column: str,
    customers_column: str,
    *,
    timestamp_column: str = "timestamp",
    source_duration_column: Optional[str] = None,
    identity_is_derived: bool = False,
    settings: GroundTruthSettings = GroundTruthSettings(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean and episode-key raw outage snapshots before event aggregation.

    Rows that cannot define an outage observation (missing event ID/timestamp or
    a missing/negative customer count) are removed with an audit count.  Invalid
    source start times are retained through an explicit fallback to the observed
    timestamp; they are never allowed to create century-long durations.
    """

    required = [timestamp_column, start_column, customers_column, *event_id_columns]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise KeyError(f"Missing required per_outage columns: {missing}")

    frame = dataframe.copy()
    frame["_gt_source_row_order"] = np.arange(len(frame), dtype=np.int64)
    frame["_gt_raw_event_id"], missing_event_id = _normalise_event_id(
        frame, event_id_columns
    )

    frame[timestamp_column] = coerce_datetime_series(frame[timestamp_column])
    frame[start_column] = coerce_datetime_series(frame[start_column])
    frame[customers_column] = pd.to_numeric(frame[customers_column], errors="coerce")

    frame["_gt_source_start_time"] = frame[start_column]
    frame["_gt_customers_out"] = frame[customers_column]
    frame["_gt_identity_is_derived"] = bool(identity_is_derived)
    if source_duration_column and source_duration_column in frame.columns:
        frame["_gt_source_duration_minutes"] = pd.to_numeric(
            frame[source_duration_column], errors="coerce"
        )
        frame.loc[
            frame["_gt_source_duration_minutes"].le(0),
            "_gt_source_duration_minutes",
        ] = np.nan
    else:
        frame["_gt_source_duration_minutes"] = np.nan

    invalid_timestamp = frame[timestamp_column].isna()
    invalid_customers = frame["_gt_customers_out"].isna() | frame[
        "_gt_customers_out"
    ].lt(0)
    source_start_missing = frame["_gt_source_start_time"].isna()
    source_start_sentinel = (
        frame["_gt_source_start_time"].dt.year.lt(settings.minimum_valid_year).fillna(False)
    )
    source_start_after_snapshot = (
        frame["_gt_source_start_time"]
        .gt(
            frame[timestamp_column]
            + pd.to_timedelta(settings.expected_snapshot_interval_minutes, unit="m")
        )
        .fillna(False)
    )
    frame["_gt_source_start_invalid"] = (
        source_start_missing | source_start_sentinel | source_start_after_snapshot
    )
    frame["_gt_source_start_valid"] = ~frame["_gt_source_start_invalid"]

    report = [
        _report_row("snapshot_cleaning", "input_rows", len(frame), "observed"),
        _report_row(
            "snapshot_cleaning",
            "missing_event_id_rows",
            missing_event_id.sum(),
            "removed",
        ),
        _report_row(
            "snapshot_cleaning",
            "invalid_timestamp_rows",
            invalid_timestamp.sum(),
            "removed",
        ),
        _report_row(
            "snapshot_cleaning",
            "invalid_customer_rows",
            invalid_customers.sum(),
            "removed",
        ),
        _report_row(
            "snapshot_cleaning",
            "missing_source_start_rows",
            source_start_missing.sum(),
            "fallback_and_flag",
        ),
        _report_row(
            "snapshot_cleaning",
            "sentinel_source_start_rows",
            source_start_sentinel.sum(),
            "fallback_and_flag",
        ),
        _report_row(
            "snapshot_cleaning",
            "source_start_after_snapshot_rows",
            source_start_after_snapshot.sum(),
            "fallback_and_flag",
        ),
    ]

    hard_invalid = missing_event_id | invalid_timestamp | invalid_customers
    frame = frame.loc[~hard_invalid].copy()

    duplicate_subset = [
        "_gt_raw_event_id",
        timestamp_column,
        "_gt_source_start_time",
        "_gt_customers_out",
    ]
    exact_duplicate = frame.duplicated(duplicate_subset, keep="first")
    report.append(
        _report_row(
            "snapshot_cleaning",
            "exact_duplicate_rows",
            exact_duplicate.sum(),
            "removed",
        )
    )
    frame = frame.loc[~exact_duplicate].copy()

    frame = _cluster_source_start_times(
        frame, settings.start_time_merge_tolerance_minutes
    )
    no_canonical_start = frame["_gt_canonical_start_time"].isna()
    # With no trustworthy source start, keep consecutive observations for the
    # same raw ID together and let the observation-gap rule split episodes.
    # Using each row's timestamp as its own anchor would incorrectly turn every
    # snapshot into a one-row event.
    fallback_start = frame.groupby("_gt_raw_event_id")[timestamp_column].transform(
        "min"
    )
    # Round in UTC so the repeated local hour at the autumn DST transition is
    # unambiguous, then restore the source timezone for readable episode IDs.
    source_timezone = fallback_start.dt.tz
    fallback_start = (
        fallback_start.dt.tz_convert("UTC")
        .dt.floor(f"{int(settings.expected_snapshot_interval_minutes)}min")
        .dt.tz_convert(source_timezone)
    )
    frame.loc[no_canonical_start, "_gt_canonical_start_time"] = fallback_start.loc[
        no_canonical_start
    ]
    frame["_gt_source_start_inferred"] = no_canonical_start

    frame["_gt_episode_base"] = (
        frame["_gt_raw_event_id"].astype(str)
        + "::"
        + frame["_gt_canonical_start_time"].astype(str)
    )

    conflict_key = ["_gt_episode_base", timestamp_column]
    conflict_sizes = frame.groupby(conflict_key)["_gt_customers_out"].transform("nunique")
    frame["_gt_conflicting_snapshot"] = conflict_sizes.gt(1)
    conflicting_rows = int(frame["_gt_conflicting_snapshot"].sum())

    frame = frame.sort_values(
        ["_gt_episode_base", timestamp_column, "_gt_source_row_order"]
    )
    duplicate_timestamp = frame.duplicated(conflict_key, keep="last")
    report.append(
        _report_row(
            "snapshot_cleaning",
            "conflicting_snapshot_rows",
            conflicting_rows,
            "kept_last_row_and_flagged",
        )
    )
    report.append(
        _report_row(
            "snapshot_cleaning",
            "duplicate_timestamp_rows",
            duplicate_timestamp.sum(),
            "kept_last_row",
        )
    )
    frame = frame.loc[~duplicate_timestamp].copy()

    observed_gap = (
        frame.groupby("_gt_episode_base")[timestamp_column]
        .diff()
        .dt.total_seconds()
        .div(60.0)
    )
    frame["_gt_gap_before"] = observed_gap.gt(settings.max_observation_gap_minutes)
    frame["_gt_gap_after"] = (
        frame.groupby("_gt_episode_base")["_gt_gap_before"]
        .shift(-1, fill_value=False)
        .astype(bool)
    )
    frame["_gt_episode_number"] = frame.groupby("_gt_episode_base")[
        "_gt_gap_before"
    ].cumsum()
    frame["_gt_episode_id"] = (
        frame["_gt_episode_base"]
        + "::episode-"
        + frame["_gt_episode_number"].astype(int).astype(str)
    )
    frame = frame.sort_values(["_gt_episode_id", timestamp_column]).reset_index(drop=True)

    reused_id_count = int(
        frame.loc[frame["_gt_source_event_id_reused"], "_gt_raw_event_id"].nunique()
    )
    report.extend(
        [
            _report_row(
                "snapshot_cleaning",
                "source_event_ids_reused",
                reused_id_count,
                "split_by_start_time",
            ),
            _report_row(
                "snapshot_cleaning",
                "long_observation_gaps",
                frame["_gt_gap_before"].sum(),
                "split_into_new_episode",
            ),
            _report_row(
                "snapshot_cleaning", "output_rows", len(frame), "retained"
            ),
            _report_row(
                "snapshot_cleaning",
                "output_episodes",
                frame["_gt_episode_id"].nunique(),
                "retained",
            ),
        ]
    )

    return frame, pd.DataFrame(report)


def build_ground_truth_events(
    snapshots: pd.DataFrame,
    *,
    dataset_lower_bound: Optional[pd.Timestamp] = None,
    dataset_upper_bound: Optional[pd.Timestamp] = None,
    settings: GroundTruthSettings = GroundTruthSettings(),
) -> pd.DataFrame:
    """Aggregate prepared snapshots into consistent reference-event metrics."""

    if snapshots.empty:
        return pd.DataFrame()

    timestamp_column = "timestamp"
    frame = snapshots.sort_values(["_gt_episode_id", timestamp_column]).copy()
    grouped = frame.groupby("_gt_episode_id", sort=False)

    events = grouped.agg(
        raw_event_id=("_gt_raw_event_id", "first"),
        source_start_time=("_gt_canonical_start_time", "min"),
        observed_start_time=(timestamp_column, "min"),
        observed_end_time=(timestamp_column, "max"),
        snapshot_count=(timestamp_column, "size"),
        source_customer_min=("_gt_customers_out", "min"),
        source_customer_max=("_gt_customers_out", "max"),
        first_customers_out=("_gt_customers_out", "first"),
        last_customers_out=("_gt_customers_out", "last"),
        source_duration_minutes=("_gt_source_duration_minutes", "max"),
        source_start_was_invalid=("_gt_source_start_invalid", "max"),
        source_start_inferred=("_gt_source_start_inferred", "min"),
        source_event_id_reused=("_gt_source_event_id_reused", "max"),
        identity_is_derived=("_gt_identity_is_derived", "max"),
        has_conflicting_snapshots=("_gt_conflicting_snapshot", "max"),
        has_gap_before=("_gt_gap_before", "max"),
        has_gap_after=("_gt_gap_after", "max"),
    )

    previous_timestamp = grouped[timestamp_column].shift(1)
    previous_customers = grouped["_gt_customers_out"].shift(1)
    interval_minutes = (
        frame[timestamp_column]
        .sub(previous_timestamp)
        .dt.total_seconds()
        .div(60.0)
        .fillna(0.0)
    )
    trapezoid_area = (
        (frame["_gt_customers_out"] + previous_customers).div(2.0)
        * interval_minutes
    ).fillna(0.0)
    frame["_gt_interval_minutes"] = interval_minutes
    frame["_gt_trapezoid_area"] = trapezoid_area
    exposure = frame.groupby("_gt_episode_id", sort=False).agg(
        observed_minutes=("_gt_interval_minutes", "sum"),
        observed_customer_minutes=("_gt_trapezoid_area", "sum"),
    )
    events = events.join(exposure)

    if dataset_lower_bound is not None:
        lower_bound = pd.Timestamp(dataset_lower_bound)
        window_left_censored = events["source_start_time"].lt(lower_bound)
    else:
        window_left_censored = pd.Series(False, index=events.index)

    if dataset_upper_bound is not None:
        upper_bound = pd.Timestamp(dataset_upper_bound)
        right_margin = pd.to_timedelta(
            settings.expected_snapshot_interval_minutes, unit="m"
        )
        window_right_censored = events["observed_end_time"].ge(
            upper_bound - right_margin
        )
    else:
        window_right_censored = pd.Series(False, index=events.index)

    events["is_left_censored"] = window_left_censored | events["has_gap_before"]
    events["is_right_censored"] = window_right_censored | events["has_gap_after"]

    has_valid_source_start = ~events["source_start_inferred"]
    use_source_start = has_valid_source_start & ~events["is_left_censored"]
    # pandas 3 preserves the parsed datetime resolution (for example, ``us``
    # for snapshot timestamps and ``ns`` for some California source starts)
    # and rejects a lossy masked assignment between them.  Series.where()
    # resolves both inputs to their common lossless datetime dtype first.
    events["start_time"] = events["observed_start_time"].where(
        ~use_source_start, events["source_start_time"]
    )

    observed_span = (
        events["observed_end_time"]
        .sub(events["observed_start_time"])
        .dt.total_seconds()
        .div(60.0)
    )
    duration_from_start = (
        events["observed_end_time"]
        .sub(events["start_time"])
        .dt.total_seconds()
        .div(60.0)
        .clip(lower=0.0)
    )
    duration_lower = pd.concat(
        [observed_span.rename("observed_span"), duration_from_start.rename("from_start")],
        axis=1,
    ).max(axis=1)

    usable_source_duration = (
        events["source_duration_minutes"].notna()
        & events["source_duration_minutes"].gt(0)
        & use_source_start
    )
    duration_lower.loc[usable_source_duration] = pd.concat(
        [
            duration_lower.loc[usable_source_duration],
            events.loc[usable_source_duration, "source_duration_minutes"],
        ],
        axis=1,
    ).max(axis=1)

    half_interval = settings.expected_snapshot_interval_minutes / 2.0
    events["duration_lower_bound"] = duration_lower
    events["duration_upper_bound"] = (
        duration_lower + settings.expected_snapshot_interval_minutes
    )
    events["duration"] = duration_lower + half_interval
    events["duration_timestamp_diff"] = observed_span
    events["duration_mean"] = events["duration"]
    events["duration_max"] = events["duration_upper_bound"]
    events["end_time"] = events["start_time"] + pd.to_timedelta(
        events["duration"], unit="m"
    )
    # Extend the first observed count over the unobserved interval before the
    # first snapshot and the last count over the half-interval midpoint after
    # the final snapshot.  The interior is integrated with the trapezoidal
    # rule.  This keeps customer-minutes and the reported average consistent.
    events["pre_observation_minutes"] = (
        events["duration_lower_bound"] - observed_span
    ).clip(lower=0.0)
    events["post_observation_minutes"] = half_interval
    events["weighted_customers_out"] = (
        events["first_customers_out"] * events["pre_observation_minutes"]
        + events["observed_customer_minutes"]
        + events["last_customers_out"] * events["post_observation_minutes"]
    )
    events["average_customers_out"] = (
        events["weighted_customers_out"] / events["duration"]
    )

    events["duration_method"] = "snapshot_interval_midpoint"
    events.loc[usable_source_duration, "duration_method"] = (
        "reported_elapsed_plus_half_interval"
    )
    events.loc[events["is_left_censored"], "duration_method"] = (
        "observed_span_left_censored_plus_half_interval"
    )

    duration_for_coverage = events["duration_lower_bound"].replace(0, np.nan)
    events["observation_coverage_ratio"] = (
        events["observed_minutes"] / duration_for_coverage
    ).clip(upper=1.0)
    events["sparse_observation"] = (
        events["duration_lower_bound"].gt(settings.max_observation_gap_minutes)
        & events["observation_coverage_ratio"].fillna(0).lt(0.5)
    )
    events["reported_duration_inconsistent"] = usable_source_duration & (
        events["source_duration_minutes"].sub(duration_from_start).abs().gt(
            settings.max_observation_gap_minutes
        )
    )
    events["ground_truth_version"] = GROUND_TRUTH_VERSION

    return events.reset_index().rename(columns={"_gt_episode_id": "episode_id"})


def _append_flag(flags: pd.Series, mask: pd.Series, label: str) -> pd.Series:
    separator = flags.ne("").map({True: ";", False: ""})
    addition = separator + label
    return flags.where(~mask.fillna(False), flags + addition)


def validate_ground_truth_events(
    events: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply hard invariants and non-destructive review flags to event rows.

    Returns ``(all_events, valid_events, flagged_events, quality_report)``.
    "Valid" means that all mathematical invariants hold.  Soft flags such as a
    long duration or one-snapshot event do not silently remove an event.
    """

    frame = events.copy()
    if frame.empty:
        report = pd.DataFrame(
            [_report_row("event_validation", "generated_events", 0, "observed")]
        )
        return frame, frame, frame, report

    duration_from_bounds = (
        pd.to_datetime(frame["end_time"], utc=True, errors="coerce")
        - pd.to_datetime(frame["start_time"], utc=True, errors="coerce")
    ).dt.total_seconds().div(60.0)
    expected_customer_minutes = (
        frame["average_customers_out"] * frame["duration"]
    )
    customer_minute_tolerance = expected_customer_minutes.abs().mul(1e-9).clip(
        lower=1e-6
    )
    source_id_value_count = frame.get(
        "source_event_id_value_count",
        pd.Series(1, index=frame.index),
    )

    hard_checks = {
        "missing_episode_id": frame["episode_id"].isna()
        | frame["episode_id"].astype("string").str.strip().eq(""),
        "invalid_event_time": frame["start_time"].isna() | frame["end_time"].isna(),
        "nonpositive_duration": frame["duration"].isna() | frame["duration"].le(0),
        "duration_end_mismatch": duration_from_bounds.sub(frame["duration"]).abs().gt(1e-6),
        "invalid_duration_bounds": frame["duration_lower_bound"].lt(0)
        | frame["duration_upper_bound"].lt(frame["duration_lower_bound"])
        | frame["duration"].lt(frame["duration_lower_bound"])
        | frame["duration"].gt(frame["duration_upper_bound"]),
        "negative_customers": frame["average_customers_out"].isna()
        | frame["average_customers_out"].lt(0),
        "customers_below_source_min": frame["average_customers_out"].lt(
            frame["source_customer_min"] - 1e-9
        ),
        "customers_above_source_max": frame["average_customers_out"].gt(
            frame["source_customer_max"] + 1e-9
        ),
        "negative_customer_minutes": frame["weighted_customers_out"].isna()
        | frame["weighted_customers_out"].lt(0),
        "customer_minutes_mismatch": frame["weighted_customers_out"]
        .sub(expected_customer_minutes)
        .abs()
        .gt(customer_minute_tolerance),
        "duplicate_episode_id": frame.duplicated("episode_id", keep=False),
    }
    soft_checks = {
        "zero_customers": frame["average_customers_out"].eq(0),
        "single_snapshot": frame["snapshot_count"].eq(1),
        "source_start_invalid": frame["source_start_was_invalid"].astype(bool),
        "source_start_inferred": frame["source_start_inferred"].astype(bool),
        "source_event_id_reused": frame["source_event_id_reused"].astype(bool),
        "source_event_id_changed": source_id_value_count.gt(1),
        "derived_event_identity": frame["identity_is_derived"].astype(bool),
        "conflicting_snapshots": frame["has_conflicting_snapshots"].astype(bool),
        "left_censored": frame["is_left_censored"].astype(bool),
        "right_censored": frame["is_right_censored"].astype(bool),
        "observation_gap_split": frame["has_gap_before"].astype(bool)
        | frame["has_gap_after"].astype(bool),
        "sparse_observation": frame["sparse_observation"].astype(bool),
        "reported_duration_inconsistent": frame[
            "reported_duration_inconsistent"
        ].astype(bool),
        "duration_over_1_day": frame["duration"].gt(24 * 60),
        "duration_over_7_days": frame["duration"].gt(7 * 24 * 60),
        "duration_over_30_days": frame["duration"].gt(30 * 24 * 60),
    }

    hard_flags = pd.Series("", index=frame.index, dtype="object")
    soft_flags = pd.Series("", index=frame.index, dtype="object")
    for label, mask in hard_checks.items():
        hard_flags = _append_flag(hard_flags, mask, label)
    for label, mask in soft_checks.items():
        soft_flags = _append_flag(soft_flags, mask, label)

    frame["hard_quality_flags"] = hard_flags
    frame["quality_flags"] = soft_flags
    frame["is_valid"] = hard_flags.eq("")
    frame["quality_status"] = np.select(
        [~frame["is_valid"], soft_flags.ne("")],
        ["invalid", "flagged"],
        default="valid",
    )
    frame["eligible_for_duration_metrics"] = (
        frame["is_valid"]
        & ~frame["is_left_censored"]
        & ~frame["is_right_censored"]
        & ~frame["sparse_observation"]
        & ~frame["source_start_inferred"]
    )

    report = [
        _report_row("event_validation", "generated_events", len(frame), "observed"),
        _report_row(
            "event_validation", "valid_events", frame["is_valid"].sum(), "retained"
        ),
        _report_row(
            "event_validation",
            "hard_invalid_events",
            (~frame["is_valid"]).sum(),
            "excluded_from_ground_truth_valid",
        ),
        _report_row(
            "event_validation",
            "soft_flagged_events",
            frame["quality_flags"].ne("").sum(),
            "retained_and_flagged",
        ),
        _report_row(
            "event_validation",
            "eligible_for_duration_metrics",
            frame["eligible_for_duration_metrics"].sum(),
            "advisory",
        ),
    ]
    for label, mask in hard_checks.items():
        report.append(
            _report_row("event_validation", label, mask.sum(), "hard_invariant")
        )
    for label, mask in soft_checks.items():
        report.append(_report_row("event_validation", label, mask.sum(), "review_flag"))

    valid = frame.loc[frame["is_valid"]].copy()
    flagged = frame.loc[frame["quality_status"].ne("valid")].copy()
    return frame, valid, flagged, pd.DataFrame(report)
