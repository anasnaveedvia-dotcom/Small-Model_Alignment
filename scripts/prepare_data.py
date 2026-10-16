import argparse

from alignment.data import build_alpaca_eval_subset, prepare_run_data
from alignment.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_dir", default="outputs/data")
    parser.add_argument("--alpaca_output", default=None)
    args = parser.parse_args()

    if args.config is None and args.alpaca_output is None:
        raise SystemExit("Pass --config to prepare training data or --alpaca_output to build the AlpacaEval subset")

    if args.alpaca_output is not None:
        build_alpaca_eval_subset(args.alpaca_output)

    if args.config is not None:
        config = load_config(args.config)
        prepare_run_data(config, args.output_dir)


if __name__ == "__main__":
    main()
