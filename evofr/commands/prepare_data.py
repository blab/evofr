import argparse
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default Data Types
CASES_DTYPES = {"location": "string", "cases": "int64"}
SEQ_COUNTS_DTYPES = {"location": "string", "clade": "string", "sequences": "int64"}
DEFAULT_CUTOFF_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def load_data(file_path, dtype_map):
    """Load a TSV file into a pandas DataFrame, or return None if file is missing."""
    if file_path and os.path.exists(file_path):
        return pd.read_csv(file_path, sep="\t", parse_dates=["date"], dtype=dtype_map)
    return None


def override_config_with_cli(config, args):
    """Override YAML config values with CLI arguments if provided."""
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config


def prepare_data(args):
    """Prepare case counts and sequence counts by subsetting, pruning, and collapsing variants."""

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f).get("prepare_data", {})

    # Override with CLI arguments
    config = override_config_with_cli(config, args)

    # Convert max_date to datetime object
    logger.info(f"Setting max date (inclusive) as {config['max_date']}.")
    max_date = datetime.strptime(config["max_date"], "%Y-%m-%d")

    # Determine min_date if provided
    min_date = None
    if "included_days" in config:
        min_date = max_date - timedelta(days=(config["included_days"] - 1))
        logger.info(f"Setting min date (inclusive) as {min_date.strftime('%Y-%m-%d')}.")

    # Load sequence counts
    seq_counts = load_data(config["seq_counts"], SEQ_COUNTS_DTYPES)
    if seq_counts is None:
        raise ValueError("Sequence counts file is required but missing.")

    # Load and filter case counts (optional)
    cases = load_data(config.get("cases"), CASES_DTYPES)
    if cases is not None:
        cases = cases[
            (cases["date"] >= min_date if min_date else True)
            & (cases["date"] <= max_date)
        ]
        if "output_cases" in config:
            cases.to_csv(config["output_cases"], sep="\t", index=False)
            logger.info(f"Processed case counts saved to {config['output_cases']}")

    # Process sequence counts
    seq_counts["variant"] = seq_counts["clade"]

    # Handle clade collapsing
    force_included_clades = set()
    if "force_include_clades" in config:
        for clade_mapping in config["force_include_clades"]:
            clade, variant = clade_mapping.split("=")
            seq_counts.loc[seq_counts["clade"] == clade, "variant"] = variant
            force_included_clades.add(clade)

    # Prune sequence counts for recent days
    if "prune_seq_days" in config:
        max_clade_date = max_date - timedelta(days=config["prune_seq_days"])
        seq_counts = seq_counts[seq_counts["date"] <= max_clade_date]

    # Save processed sequence counts
    seq_counts.to_csv(config["output_seq_counts"], sep="\t", index=False)
    logger.info(f"Processed sequence counts saved to {config['output_seq_counts']}")


def main():
    """CLI wrapper for prepare-data command."""
    parser = argparse.ArgumentParser(
        description="Prepare case and sequence count data for evofr models."
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--seq-counts", help="Optional override: input sequence counts TSV"
    )
    parser.add_argument(
        "--cases", help="Optional override: input case counts TSV (optional)"
    )
    parser.add_argument(
        "--output-seq-counts", help="Optional override: output sequence counts TSV"
    )
    parser.add_argument(
        "--output-cases", help="Optional override: output case counts TSV (optional)"
    )
    args = parser.parse_args()
    prepare_data(args)


if __name__ == "__main__":
    main()
