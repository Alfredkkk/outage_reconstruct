# Reproducing the Submitted GROWER M&V Experiment in VS Code

> **Ground-truth implementation update:** the current source tree now uses the
> validated ground-truth builder version 2.0, so a new run will not reproduce
> the submitted legacy ground-truth values exactly. Preserve the archived
> submitted outputs for baseline reproduction. See
> [`GROUND_TRUTH_CHANGES.md`](GROUND_TRUTH_CHANGES.md) for the corrected builder,
> quality gate, audit files, and validation status.

## Material Passport

- **Artifact type:** Legacy experiment reproduction guide
- **Experiment:** County-snapshot-to-outage-event reconstruction benchmark
- **Analysis window:** 2023-11-01 00:00:00 through 2024-11-30 23:59:59 (US Eastern Time)
- **Expected evaluation units:** 26 (12 Georgia, 13 Texas, and one aggregated California unit)
- **Verification status:** Reconstructed from the pipeline documentation, source code, configuration files, and archived outputs; the full experiment has not yet been rerun from scratch

## Scope

This document describes how to reproduce the experiment reported in the submitted manuscript. It is a reproduction of the **legacy analysis**, including the large errors reported in the manuscript. It is not the redesigned experiment for the resubmission.

The primary workflow in this guide uses VS Code's **Run Python File** action. No experiment command needs to be typed manually. VS Code still opens an integrated terminal to host the Python process, but the scripts are started from the editor's Run button.

The manuscript calls the proposed method **DTHA**. The code and output files call the same method `mixed_threshold`.

The original workflow was documented as three stages:

1. Generate the ground-truth and algorithm-reconstructed event tables.
2. Generate one comparison table for each evaluation unit.
3. Merge the comparison tables and rank the parameter configurations.

For the final manuscript experiment, the original single-parameter implementation was extended to evaluate parameter grids. The authoritative reproduction path is therefore the three staged scripts described below, using the current `base_mvpipeline.py` implementation.

## Repository and data layout

The historical server configuration assumed the following layout:

```text
/root/autodl-tmp/
├── mvpipeline/
│   ├── base_mvpipeline.py
│   ├── mvpipeline.py
│   ├── mvp_config.yaml
│   ├── generate_dataframes.py
│   ├── generate_comparison_table.py
│   ├── generate_model_rankings.py
│   ├── utils/
│   │   ├── zip_to_county_name.json
│   │   ├── zip_to_county_fips.json
│   │   └── zip_to_state_name.json
│   └── output/
└── raw_data/
    ├── ga/
    ├── tx/
    └── ca/
```



Each configured evaluation unit must have both files below:

```text
raw_data/{state}/layout_{layout}/per_county_{provider}.csv
raw_data/{state}/layout_{layout}/per_outage_{provider}.csv
```

The California aggregate must therefore include:

```text
raw_data/ca/layout_investor/per_county_investor_all.csv
raw_data/ca/layout_investor/per_outage_investor_all.csv
```

## Important path assumptions

The legacy code contains absolute server paths in addition to the data path in `mvp_config.yaml`:

- `generate_dataframes.py`, `generate_comparison_table.py`, and `generate_model_rankings.py` load `/root/autodl-tmp/mvpipeline/mvp_config.yaml`.
- `base_mvpipeline.py` loads the three geographic mapping JSON files from `/root/autodl-tmp/mvpipeline/utils/`.
- `mvp_config.yaml` sets `LOCAL_FILE_BASE_PATH` to `/root/autodl-tmp/raw_data`.

The least invasive way to reproduce the historical run is to restore this directory layout on the server. If the repository or raw data are stored elsewhere, update all of the paths above before running. Path changes should not alter the experiment itself.


## Preserve the submitted results before rerunning

The generation scripts use `force_agg=True` and `replace=True`. They overwrite matching files under `output/transformed_data/`. Before starting a full reproduction, copy the existing `output/transformed_data` directory to a separate location, for example `transformed_data_submitted_backup`.

Do not use the backup directory as the output directory for the new run.

## Step 1: verify the configured data inputs

The authoritative final sample is the active provider list in `mvp_config.yaml`. It should contain 26 evaluation units. Before running:

1. Open `mvp_config.yaml`.
2. Confirm that `LOCAL_FILE_BASE_PATH` points to the server's `raw_data` directory.
3. Confirm that the active, uncommented provider list contains 26 entries.
4. Confirm that each provider has both a matching `per_county` and `per_outage` file under the configured state and layout directory.
5. Confirm in particular that the California aggregate contains:

   ```text
   ca/layout_investor/per_county_investor_all.csv
   ca/layout_investor/per_outage_investor_all.csv
   ```

## Step 2: generate ground truth and reconstructed event tables in VS Code

Run `generate_dataframes.py`.


For every configured evaluation unit, this script performs the following operations:

1. Loads `per_county` and `per_outage` raw data.
2. Applies the layout-specific schema conversion.
3. Filters both datasets to the fixed analysis window.
4. Aggregates `per_outage` snapshots into the ground-truth event table.
5. Applies Do, Ganz, mixed-threshold/DTHA, and change-point reconstruction to `per_county` snapshots.
6. Exports each generated table under `output/transformed_data/{state}/layout_{layout}/{provider}/`.

This is the expensive stage. The change-point configurations use a 16-process pool, while other work may remain serial. Runtime depends strongly on the provider, raw-data size, processor count, memory, and storage speed. Run it in a persistent server session and retain the log.

### Parameter configurations generated by the current legacy code

| Method in manuscript | Code name | Parameter grid | Number of configurations |
|---|---|---:|---:|
| Do | `do` | `0.001`, then `0.1` through `1.0` in increments of `0.1` | 11 |
| Ganz | `ganz` | `0.001`, then `0.1` through `1.0` in increments of `0.1` | 11 |
| DTHA | `mixed_threshold` | relative threshold `0.05` through `0.30` in increments of `0.025`; absolute difference `1` through `10` | 110 |
| Change point / PELT family | `changepoints` | five cost models, with ten penalties per model | 50 |

The change-point cost models and penalty ranges are:

| Model | Penalty range |
|---|---:|
| `rbf` | `0.1` to `1.0` |
| `normal` | `0.01` to `0.1` |
| `rank` | `1e-5` to `1e-4` |
| `l2` | `1e-5` to `1e-4` |
| `l1` | `1e-7` to `1e-6` |

There are 182 non-ground-truth configurations in total. In the DTHA code, a relative threshold of `0.05` means **5%**, not `0.05%`.

### Expected files per evaluation unit

```text
{provider}_ground_truth__2023-11-01_to_2024-11-30.csv
{provider}_do__2023-11-01_to_2024-11-30.csv
{provider}_ganz__2023-11-01_to_2024-11-30.csv
{provider}_mixed_threshold__2023-11-01_to_2024-11-30.csv
{provider}_changepoints__2023-11-01_to_2024-11-30.csv
```

Review the log before continuing. Do not silently omit a provider that failed during loading, transformation, or algorithm execution.

## Step 3: generate one comparison table per evaluation unit in VS Code

After Step 2 completes successfully:

Run `generate_comparison_table.py`.

For each method and parameter configuration, the comparison stage:

1. Removes reconstructed events whose duration is not positive.
2. Calculates event frequency, mean duration, mean customers out, and total customer outage time.
3. Compares those four aggregate values with the corresponding ground-truth aggregates.
4. Calculates absolute differences and absolute percentage differences.

The output for each evaluation unit is:

```text
{provider}_comparison_table__2023-11-01_to_2024-11-30.csv
```

A comparison table can contain fewer than 183 rows when a high-threshold configuration produces no positive-duration event. This is why the final merged table does not contain exactly `26 x 183` rows.

## Step 4: merge the comparison tables and generate model rankings in VS Code

After all 26 comparison tables have been generated:

Run `generate_model_rankings.py`.

This script concatenates the 26 utility-level comparison tables and groups rows by `model_type` and `model_name`. It calculates the unweighted mean across available evaluation units for four percentage-error metrics, ranks every configuration on each metric, and applies the following rank weights:

| Ranked metric | Weight |
|---|---:|
| Frequency percentage difference | 0.5 |
| Mean-duration percentage difference | 0.2 |
| Mean-customers-out percentage difference | 0.2 |
| Total-customer-outage-time percentage difference | 0.1 |

Expected final files:

```text
output/transformed_data/merged comparison tables__2023-11-01_to_2024-11-30.csv
output/transformed_data/model_rankings__2023-11-01_to_2024-11-30.csv
```






## Known limitations of the reproduced experiment

Successful reproduction means that the submitted results have been recovered; it does not establish that the analysis is methodologically adequate. The legacy experiment has the following known limitations:

- Parameter selection and reporting use the same collection of evaluation units; there is no independent held-out test set in the executed workflow.
- The ground-truth construction filters raw outage snapshots by scraper `timestamp`, not consistently by the actual outage start time.
- Raw data contain placeholder or malformed dates and implausibly long event durations that were not fully removed by the legacy pipeline.
- Mean absolute percentage differences are unstable for utilities with small ground-truth denominators.
- The final ranking is driven by manually selected weights, with 50% assigned to the frequency rank.
- Means are unweighted across evaluation units rather than weighted by event count, customer population, or utility size.
- Configurations missing from some utility-level comparison tables may be averaged over fewer than 26 evaluation units.
- DTHA includes event-tracking and event-matching heuristics beyond the two thresholds described in the manuscript.
- The change-point implementation includes preprocessing and segmentation choices that should be audited before a revised experiment.

The archived output should therefore be retained as the **submitted baseline**. Any corrected ground-truth rules, new evaluation metrics, held-out parameter selection, or algorithm changes should be run into a separate output directory and reported as a new experiment rather than silently replacing this baseline.
