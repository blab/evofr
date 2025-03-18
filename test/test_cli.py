import shutil
import subprocess
from pathlib import Path

import pytest

TEST_CONFIG_DIR = Path("test/configs")
TEST_DATA_DIR = Path("test/data")
TEST_OUTPUT_DIR = Path("test/output")
CLI_COMMAND = "poetry run evofr"  # Adjust if necessary


@pytest.fixture(scope="session", autouse=True)
def generate_test_data():
    """Ensure test data exists by running generate_test_data.py if needed."""
    raw_seq_file = TEST_DATA_DIR / "raw_sequences.tsv"
    raw_cases_file = TEST_DATA_DIR / "raw_cases.tsv"
    if not raw_seq_file.exists() or not raw_cases_file.exists():
        subprocess.run("python test/generate_cli_test_data.py", shell=True, check=True)
    # Clean output directory.
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_cli(command, expected_code=0):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    assert result.returncode == expected_code, f"Command failed: {result.stderr}"
    return result.stdout, result.stderr


# --- Tests for prepare-data command ---
def test_prepare_data_creates_output_files():
    """Test that prepare-data produces expected output files."""
    config_path = TEST_CONFIG_DIR / "prepare_data.yaml"
    command = f"{CLI_COMMAND} prepare-data --config {config_path}"
    run_cli(command)
    seq_out = TEST_DATA_DIR / "processed_sequences.tsv"
    # Check cases output only if config contains 'cases' key
    config_text = config_path.read_text()
    if (
        "cases:" in config_text
        and config_text.strip().split("cases:")[1].strip() != '""'
    ):
        cases_out = TEST_DATA_DIR / "processed_cases.tsv"
        assert cases_out.exists(), "Case output file not created."
    assert seq_out.exists(), "Sequence output file not created."


# --- Tests for run-model command ---
def test_run_model_creates_results():
    """Test that run-model produces a results JSON file."""
    config_path = TEST_CONFIG_DIR / "run_model.yaml"
    export_dir = TEST_OUTPUT_DIR / "results"
    command = (
        f"{CLI_COMMAND} run-model --config {config_path} --export-path {export_dir}"
    )
    run_cli(command)
    result_file = export_dir / "results.json"
    assert result_file.exists(), "Results JSON file not created."
