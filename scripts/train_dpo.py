import argparse

from alignment.dpo import run_dpo
from alignment.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dpo(config, args.sft_path, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
