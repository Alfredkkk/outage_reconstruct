# Ground-Truth Cleaning and Generation Changes

## Status

This document describes ground-truth builder version **2.0**. The changes fix
the construction of reference outage events from `per_outage` snapshot data and
add a post-generation quality gate. They do **not** change the four outage-event
reconstruction algorithms that operate on `per_county` data.

The raw CSV files are never overwritten. Cleaning is performed on an in-memory
copy, and every removal, fallback, split, and validation result is recorded in
an audit output.

## Why the builder was changed

The legacy implementation grouped source rows before consistently sorting them
by timestamp. Its left- and right-weight calculations could therefore contain
negative intervals. It also treated provider IDs as globally stable and trusted
sentinel dates such as 1899. These behaviors could produce negative
customer-minutes, customer averages outside the observed source range,
century-long durations, and unrelated California events merged into one row.

The new builder treats `per_outage` as a sequence of observations of outage-map
records, not as an already-clean utility event log.

## Files changed

| File | Modification |
|---|---|
| `utils/ground_truth.py` | New shared module for snapshot cleaning, episode construction, event aggregation, validation, and audit counts. |
| `base_mvpipeline.py` | Adds shared ground-truth settings and workflow, uses the validated builder for all layouts, exposes all/valid/flagged views, exports audit files, and loads geographic maps from either the server or local repository. |
| `mvpipeline.py` | Removes four duplicated legacy aggregation implementations. Layout classes now only normalize source schemas and declare the columns used by the shared builder. Adds a California-specific derived identity because `IncidentId` is not stable in all feeds. |
| `test_ground_truth.py` | Adds focused regression tests for ordering, invalid dates, reused IDs, gaps, duplicate timestamps, reported durations, California identity behavior, and hard validation failures. |
| `GROUND_TRUTH_CHANGES.md` | This implementation and reproducibility record. |

### Compatibility note

The start-time imputation branch explicitly skips empty mappings and converts
non-empty lookup Series to dictionaries. This avoids a pandas-version-specific
`DatetimeArray`-to-`float64` cast error on Windows without changing episode
construction.

## New data flow

The ground-truth path is now:

1. Load raw `per_outage` snapshots.
2. Apply the layout adapter in `mvpipeline.py`.
3. Restrict snapshots to the study window.
4. Clean observations and construct episode IDs with
   `prepare_outage_snapshots()`.
5. Aggregate each sorted episode with `build_ground_truth_events()`.
6. Add utility and geographic metadata in `BaseMVPipeline`.
7. Apply hard invariants and non-destructive review flags with
   `validate_ground_truth_events()`.
8. Export the main table and its audit artifacts.

## Default construction settings

| Setting | Default | Rationale |
|---|---:|---|
| Expected snapshot interval | 15 minutes | The intended polling cadence and the uncertainty interval used for event restoration. |
| Maximum observation gap | 31 minutes | Two nominal polling intervals plus one minute for collection-time jitter. A larger gap starts a new episode. |
| Start-time correction tolerance | 16 minutes | One nominal interval plus one minute for clock and collection jitter. Nearby source corrections are treated as the same start. |
| Minimum valid source year | 2000 | Rejects known placeholder dates such as 1899 and year 1. |

California uses a 31-minute start-time correction tolerance by default because
some feeds replace `StartDate` at approximately every poll. All values can be
overridden for a provider in a YAML configuration:

```yaml
ground_truth:
  expected_snapshot_interval_minutes: 15
  max_observation_gap_minutes: 31
  start_time_merge_tolerance_minutes: 16
  minimum_valid_year: 2000
```

If this block is added to the California provider, its explicit value replaces
the California class default as well.

## Step 1: cleaning snapshots and constructing episodes

### Datetime and numeric normalization

- Snapshot and source-start timestamps are parsed with invalid values converted
  to `NaT` for auditing.
- Timestamps are represented in US Eastern time, matching the existing study
  window.
- Customer counts are converted to numeric values.
- Source-reported durations, where available, are converted to minutes.
- Rounding used to create fallback episode anchors is performed in UTC so the
  repeated local hour at the autumn daylight-saving transition is unambiguous.

### Rows removed before aggregation

A row is removed only if it cannot define a usable outage observation:

- the event identity is missing;
- the snapshot timestamp is invalid;
- the customer count is missing, nonnumeric, or negative; or
- the row is an exact duplicate of an earlier observation.

Counts for every category are written to the quality report. No row is removed
solely because the outage is short or has few customers.

### Source start-time handling

A source start is marked invalid when it is missing, earlier than year 2000, or
more than one expected snapshot interval after its observation timestamp.
Invalid starts cannot contribute to a long duration. The builder falls back to
the first observed timestamp for the raw identity and adds explicit flags.

Nearby changes in a valid source start are treated as corrections. Larger
changes create separate source-start clusters so an ID reused across events or
years is not aggregated into one event.

### Observation gaps and duplicate timestamps

- All observations are sorted by episode key and timestamp before any
  difference or weighting is calculated.
- A gap longer than the configured maximum starts a new episode.
- If the same episode and timestamp contain identical values, one row is kept.
- If they contain conflicting customer values, the last source row is kept and
  the event is flagged as containing a conflict.

### California event identity

For the California aggregate, `IncidentId` is not consistently stable. In the
LAWP portion of the supplied file, the same coordinate can retain an outage
while `IncidentId`, `GlobalID`, `OBJECTID`, and `StartDate` change between
snapshots. Grouping only by `IncidentId` caused unrelated records to be joined
over long periods.

The California adapter now uses the following continuity key:

```text
UtilityCompany + County + longitude rounded to 5 decimals + latitude rounded to 5 decimals
```

If coordinates are unavailable, it falls back to utility plus `IncidentId`.
Time gaps and source-start clusters still split repeated outages at the same
location. Events built with this heuristic carry the
`derived_event_identity` quality flag, and the original `IncidentId` remains in
the exported metadata. `source_event_id_values` records every distinct source
ID assigned to the reconstructed episode, and `source_event_id_value_count`
makes ID changes directly testable.

This is an auditable repair, not proof that the California source provides a
true utility event identifier. Coordinate rounding should be included in the
sensitivity analysis for the revised manuscript.

## Step 2: consistent event metrics and the quality gate

### Duration

For an episode, define:

- `observed_span` as the final snapshot timestamp minus the first snapshot;
- `from_start` as the final snapshot minus a valid, uncensored source start;
- `reported_elapsed` as the maximum positive source-reported duration when that
  field exists.

The duration lower bound is the maximum applicable value above. The nominal
upper bound adds one 15-minute polling interval, and the primary point estimate
is its midpoint:

```text
duration_lower_bound = max(applicable elapsed-time evidence)
duration_upper_bound = duration_lower_bound + 15 minutes
duration              = duration_lower_bound + 7.5 minutes
```

This produces a positive midpoint estimate for a one-snapshot event without
claiming that its exact restoration time is observed. Left- and right-censored
events remain in the auditable table but are flagged and are not marked
`eligible_for_duration_metrics`.

### Customers affected and customer-minutes

Interior customer exposure is integrated over chronologically sorted snapshots
using the trapezoidal rule. The first observed customer count is extended over
the inferred pre-observation interval, and the final count is extended over the
7.5-minute endpoint midpoint interval:

```text
customer_minutes =
    first_count * pre_observation_minutes
    + trapezoidal_area_between_snapshots
    + last_count * 7.5 minutes

average_customers_out = customer_minutes / duration
```

Consequently, `weighted_customers_out` is always consistent with
`average_customers_out * duration`, and the average must remain between the
minimum and maximum customer values observed for that episode.

### Hard invariants

An event is excluded from the main valid ground-truth table if any of these
mathematical invariants fail:

- valid start and end timestamps;
- positive duration;
- internally ordered duration bounds and a point estimate inside those bounds;
- `end_time - start_time` equals `duration`;
- nonnegative average customers and customer-minutes;
- `customer_minutes` equals `average_customers_out * duration`;
- average customers lies within the episode's observed source minimum and
  maximum; and
- unique episode ID.

The event remains in `ground_truth_all` with its `hard_quality_flags`.

### Review flags that do not remove events

The quality gate flags, but retains, events with:

- zero customers or only one snapshot;
- invalid or inferred source starts;
- a reused or changed source ID, or a derived California identity;
- conflicting same-time observations;
- left or right censoring;
- an observation-gap split or sparse coverage;
- disagreement between source-reported and timestamp-derived duration; or
- duration longer than 1, 7, or 30 days.

`eligible_for_duration_metrics` is an advisory field for sensitivity analyses.
It is false for hard-invalid, censored, sparse, or inferred-start events. The
current comparison-table code still uses all hard-valid events; it does not
silently change the submitted evaluation population.

## Exported files

Calling:

```python
pipeline.create_groundtruth(force_agg=True)
pipeline.export_file(df_name='ground_truth', replace=True)
```

now produces four dated CSV files:

```text
{provider}_ground_truth__{start}_to_{end}.csv
{provider}_ground_truth_all__{start}_to_{end}.csv
{provider}_ground_truth_flagged__{start}_to_{end}.csv
{provider}_ground_truth_quality_report__{start}_to_{end}.csv
```

- `ground_truth` contains events that pass all hard invariants. Soft-flagged
  events remain included.
- `ground_truth_all` contains every generated event, including hard-invalid
  rows.
- `ground_truth_flagged` contains all hard-invalid or soft-flagged rows.
- `ground_truth_quality_report` contains category counts from snapshot cleaning
  and event validation.

Every generated row includes `ground_truth_version = 2.0`. An existing legacy
ground-truth CSV without this version is regenerated instead of being silently
loaded as current output.

## Running from VS Code

The existing VS Code workflow remains valid:

1. Open `mvp_handler.py`.
2. Confirm that the selected YAML file points `LOCAL_FILE_BASE_PATH` to the raw
   data directory. In the current workspace, the supplied data are under the
   older `m&v-pipeline/raw_data` directory.
3. Use VS Code's **Run Python File** action and supply the configuration and any
   requested algorithm options in the Run/Debug configuration.
4. The handler calls `transform_datasets()`,
   `create_groundtruth(force_agg=True)`, and
   `export_file('ground_truth', replace=True)`. The audit files are generated
   automatically by the last call.

For a ground-truth-only diagnostic run, the same three method calls can be
executed from a VS Code Python file or interactive window; no reconstruction
algorithm needs to run.

## Verification completed

The implementation has focused regression tests covering:

- unsorted snapshots and bounded customer metrics;
- 1899 sentinel starts;
- splitting a long observation gap;
- IDs reused across years;
- small source-start corrections;
- moving California starts with a derived location identity;
- conflicting observations at the same timestamp;
- source-reported duration; and
- deliberate hard-invariant failure.

Real-data spot checks were also run on Satilla REMC, Walton EMC, and a
California LAWP slice. The checked outputs contained no negative customer
averages, no average above its event-level observed maximum, and no negative
customer-minutes. The known 1899 Satilla record was converted into a short,
flagged, one-snapshot event rather than a century-long outage. A full
all-provider experiment has not yet been rerun; these changes define the input
for that later revised experiment.

## Interpretation limits

- These tables are reconstructed reference events from public outage-map
  snapshots. They should not be described as exact utility outage logs.
- A one-snapshot event has interval uncertainty; it is not evidence that the
  exact duration was 7.5 minutes.
- No five-minute minimum-duration filter is applied. Short-event inclusion or
  exclusion should be reported as a separate sensitivity analysis rather than
  hidden inside ground-truth cleaning.
- Hard-invalid rows must be inspected in the audit output before a revised
  experiment is accepted.
- The archived submitted outputs should be preserved. Builder version 2.0 is a
  corrected experiment and should not overwrite the only copy of the submitted
  baseline.
