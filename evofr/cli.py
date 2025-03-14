import argparse
from evofr.commands.run_model import run_model


def main():
    parser = argparse.ArgumentParser(prog="evofr", description="Evolutionary forecasting CLI")
    subparsers = parser.add_subparsers(dest="command")
    # Run-model command
    parser_model = subparsers.add_parser("run-model", help="Run an evofr model")
    parser_model.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser_model.add_argument("--export-path", help="Optional export directory override")
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
