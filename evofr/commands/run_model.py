#!/usr/bin/env python
import argparse
import inspect
import json
import logging
import os
from datetime import date

import pandas as pd
import yaml

import evofr as ef
from evofr.posterior.posterior_helpers import get_sites_variants_tidy

# Set up basic logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import registries
from evofr.commands.registries import INFERENCE_REGISTRY, PRIOR_REGISTRY

# Import base classes that automatically register subclasses.
from evofr.data.data_spec import DataSpec  # DataSpec.registry will be used for data.
from evofr.models.model_spec import (  # ModelSpec.registry will be used for models.
    ModelSpec,
)


# ----------------------------------------------------------------------------
# Helper: Filter Constructor Arguments
# ----------------------------------------------------------------------------
def filter_constructor_args(cls, config_dict):
    """
    Uses introspection to return a dict of only those keys in config_dict that
    match the parameters of cls.__init__.
    """
    sig = inspect.signature(cls.__init__)
    valid_args = {}
    for param in sig.parameters.values():
        if param.name == "self":
            continue
        if param.name in config_dict:
            valid_args[param.name] = config_dict[param.name]
    return valid_args


# ----------------------------------------------------------------------------
# Recursive Instantiation Function
# ----------------------------------------------------------------------------
def instantiate_from_config(config):
    """
    Recursively instantiate a component from a configuration dictionary.
    If a dict contains a "type" key, then:
      - First, check ModelSpec.registry for a model,
      - Next, check DataSpec.registry for a data class,
      - Then check INFERENCE_REGISTRY and PRIOR_REGISTRY.
    Otherwise, recurse into nested dicts/lists.
    """
    if isinstance(config, dict):
        if "type" in config:
            comp_type = config["type"]
            config_copy = {
                k: instantiate_from_config(v) for k, v in config.items() if k != "type"
            }
            cls = None
            if comp_type in ModelSpec.registry:
                cls = ModelSpec.registry[comp_type]
            elif comp_type in DataSpec.registry:
                cls = DataSpec.registry[comp_type]
            elif comp_type in INFERENCE_REGISTRY:
                cls = INFERENCE_REGISTRY[comp_type]
            elif comp_type in PRIOR_REGISTRY:
                cls = PRIOR_REGISTRY[comp_type]
            else:
                raise ValueError(
                    f"Component type '{comp_type}' not found in any registry."
                )
            args = filter_constructor_args(cls, config_copy)
            return cls(**args)
        else:
            return {k: instantiate_from_config(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [instantiate_from_config(item) for item in config]
    else:
        return config


# ----------------------------------------------------------------------------
# Override Data File Paths
# ----------------------------------------------------------------------------
def override_generic_paths_in_config(config, cli_args):
    """
    Recursively process a configuration dictionary (for model or data),
    loading any file paths for keys ending with '_path' and storing the loaded
    DataFrame under a new key (with the '_path' suffix removed).

    Parameters:
      config (dict): A configuration dictionary (e.g. config["model"] or config["data"]).
      cli_args: The command-line arguments namespace.

    Returns:
      dict: The updated configuration dictionary with file paths replaced by DataFrames.
    """
    if isinstance(config, dict):
        new_config = {}
        for key, value in config.items():
            if isinstance(value, dict) or isinstance(value, list):
                new_config[key] = override_generic_paths_in_config(value, cli_args)
            elif isinstance(value, str) and key.endswith("_path"):
                # Use CLI override if available, otherwise use the YAML value.
                file_path = getattr(cli_args, key, None) or value
                try:
                    df = pd.read_csv(file_path, sep="\t")
                    new_key = key[:-5]  # Remove the '_path' suffix.
                    new_config[new_key] = df
                except Exception as e:
                    raise ValueError(
                        f"Failed to load file for key '{key}' from {file_path}: {e}"
                    )
            else:
                new_config[key] = value
        return new_config
    elif isinstance(config, list):
        return [override_generic_paths_in_config(item, cli_args) for item in config]
    else:
        return config


# ----------------------------------------------------------------------------
# Exporting Results
# ----------------------------------------------------------------------------
def export_results(posterior, export_config):
    """
    Exports selected sites of interest from the posterior samples.

    Parameters:
    - posterior: PosteriorHandler object containing model results.
    - export_config: Dictionary containing `export_path`, `sites`, `dated`, and `forecasts`.

    Saves results to JSON in the specified export directory.
    """
    export_path = export_config.get("export_path", "results/")
    sites = export_config["sites"]
    dated = export_config["dated"]
    forecasts = export_config["forecasts"]

    os.makedirs(export_path, exist_ok=True)

    logger.info(f"Exporting results for model: {posterior.name}")

    results = get_sites_variants_tidy(
        samples=posterior.samples,
        data=posterior.data,
        sites=sites,
        dated=dated,
        forecasts=forecasts,
        ps=[0.5, 0.8, 0.95],  # Default percentiles
        name=posterior.name,
        ps_point_estimator=export_config.get("ps_point_estimator", "median"),
    )

    results["metadata"]["updated"] = pd.to_datetime(date.today()).isoformat()

    export_file = os.path.join(export_path, "results.json")
    ef.save_json(results, path=export_file)

    logger.info(f"Results exported to {export_file}")


# ----------------------------------------------------------------------------
# run_model Function (Exported)
# ----------------------------------------------------------------------------
def run_model(args):
    """
    Run an evofr model using a YAML configuration and command-line data file overrides.

    Steps:
      1. Load the YAML configuration from args.config.
      2. Validate that the config contains "model", "data", and "inference" sections.
      3. Override any keys in the data config that include "_path" with CLI values.
      4. Instantiate the model, data, and inference components via recursive instantiation.
      5. Run inference (fit the model to the data).
      6. Export results to JSON using the package's export functions.
    """
    # Load YAML configuration.
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {args.config}.")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

    # Validate required sections.
    for section in ["model", "data", "inference"]:
        if section not in config:
            raise ValueError(f"Configuration file must contain a '{section}' section.")

    # Override file paths in the data config using command-line arguments.
    config["model"] = override_generic_paths_in_config(config["model"], args)
    config["data"] = override_generic_paths_in_config(config["data"], args)
    if args.cases_path:
        try:
            config["data"]["cases_path"] = pd.read_csv(args.cases_path, sep="\t")
            logger.info(f"Overrode cases_path with file {args.cases_path}.")
        except Exception as e:
            logger.error(f"Failed to load cases file from {args.cases_path}: {e}")
            raise

    # Instantiate components.
    try:
        model = instantiate_from_config(config["model"])
        data = instantiate_from_config(config["data"])
        inference = instantiate_from_config(config["inference"])
        logger.info("Successfully instantiated components:")
        logger.info(f"  Model: {model}")
        logger.info(f"  Data: {data}")
        logger.info(f"  Inference: {inference}")
    except Exception as e:
        logger.error(f"Component instantiation failed: {e}")
        raise

    # Run model fitting.
    try:
        posterior = inference.fit(model, data)
        logger.info(f"Inference complete. Posterior: {posterior}")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

    # Export results using the package's export functions.
    export_results(posterior, config["export"])


# ----------------------------------------------------------------------------
# For Direct Execution (Testing)
# ----------------------------------------------------------------------------
def _main():
    parser = argparse.ArgumentParser(
        description="Run an evofr model using a YAML configuration and command-line data file overrides."
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML configuration file."
    )
    # Data file overrides: keys must include "_path".
    parser.add_argument(
        "--raw-seq-path", dest="raw_seq_path", help="Path to raw sequence TSV file."
    )
    parser.add_argument(
        "--raw-variant-parents-path",
        dest="raw_variant_parents_path",
        help="Path to raw variant parents TSV file.",
    )
    parser.add_argument(
        "--cases-path", help="Path to raw cases TSV file (if applicable)."
    )
    parser.add_argument("--export-path", help="Path to export JSON results.")
    args = parser.parse_args()
    run_model(args)


if __name__ == "__main__":
    _main()
