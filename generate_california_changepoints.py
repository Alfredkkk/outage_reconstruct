"""Resume Stage 1 by generating only California change-point results.

This entry point reuses the current California ground-truth CSV and reads only
the California per-county snapshots.  It does not regenerate or overwrite any
other provider or any other reconstruction method.
"""

from pathlib import Path

import pandas as pd
import yaml

from base_mvpipeline import _path_for_io
from mvpipeline import CAInvestor


START_TIME = "2023-11-01 00:00:00"
END_TIME = "2024-11-30 23:59:59"


def main():
    config_path = Path(__file__).resolve().parent / "mvp_config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    california_providers = [
        provider
        for provider in config["providers"]
        if provider["state"] == "ca"
        and str(provider["layout"]) == "investor"
        and provider["name"] == "investor_all"
    ]
    if len(california_providers) != 1:
        raise RuntimeError(
            "Expected exactly one ca/investor/investor_all provider in "
            "mvp_config.yaml"
        )

    provider = california_providers[0]
    pipeline = CAInvestor(
        provider,
        config["globals"]["LOCAL_FILE_BASE_PATH"],
        start_time=START_TIME,
        end_time=END_TIME,
    )

    # Load and transform only per_county.  The 1.4 GB California per_outage
    # source is unnecessary because this run reuses the validated ground truth.
    per_county_path, _ = pipeline._construct_raw_file_path()
    pipeline._per_county = pd.read_csv(_path_for_io(per_county_path))
    pipeline._transform_per_county()
    pipeline._per_county = pipeline._per_county[
        (pipeline._per_county["timestamp"] >= pipeline._dataset_lower_bound)
        & (pipeline._per_county["timestamp"] <= pipeline._dataset_upper_bound)
    ]
    if pipeline._per_county.empty:
        raise RuntimeError("California per_county data are empty in the study window")

    # Load the already-generated v2.0 California ground truth, then run only
    # the missing change-point grid and replace its stale output file.
    pipeline._load_aggregated_df("ground_truth")
    pipeline._transform_loaded_aggregated_df("ground_truth")
    ground_truth_versions = (
        pipeline._ground_truth_df["ground_truth_version"].astype(str).unique()
    )
    if set(ground_truth_versions) != {"2.0"}:
        raise RuntimeError(
            "California ground truth is not exclusively builder version 2.0"
        )

    print("Running only ca_investor investor_all changepoints")
    pipeline.changepoints_method(force_agg=True)
    parameter_count = pipeline._changepoints_df["params"].nunique()
    if parameter_count != 50:
        raise RuntimeError(
            f"Expected 50 change-point parameter configurations, got "
            f"{parameter_count}"
        )

    pipeline.export_file(df_name="changepoints", replace=True)
    print(
        "California changepoints complete: "
        f"{len(pipeline._changepoints_df)} rows across {parameter_count} "
        "parameter configurations"
    )


if __name__ == "__main__":
    main()
