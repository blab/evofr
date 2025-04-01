import argparse

from evofr.commands.prepare_data import prepare_data
from evofr.commands.run_model import run_model


def main():
    parser = argparse.ArgumentParser(
        prog="evofr", description="Evolutionary forecasting CLI"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Prepare-data command
    parser_prepare = subparsers.add_parser(
        "prepare-data", help="Prepare case and variant data"
    )
    parser_prepare.add_argument(
        "--config", required=True, help="Path to YAML configuration file"
    )
    parser_prepare.add_argument(
        "--seq-counts", help="Optional override: input sequence counts TSV"
    )
    parser_prepare.add_argument(
        "--cases", help="Optional override: input case counts TSV (optional)"
    )
    parser_prepare.add_argument(
        "--output-seq-counts", help="Optional override: output sequence counts TSV"
    )
    parser_prepare.add_argument(
        "--output-cases", help="Optional override: output case counts TSV (optional)"
    )
    parser_prepare.set_defaults(func=prepare_data)

    # Run-model command
    parser_model = subparsers.add_parser("run-model", help="Run an evofr model")
    parser_model.add_argument(
        "--config", required=True, help="Path to YAML configuration file"
    )
    parser_model.add_argument(
        "--export-path", help="Optional export directory override"
    )
    parser_model.add_argument("--seq-path", help="Optional sequence data override")
    parser_model.add_argument("--cases-path", help="Optional case data override")
    parser_model.add_argument("--pivot", help="Optional variant pivot override")
    parser_model.set_defaults(func=run_model)

    args = parser.parse_args()
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
