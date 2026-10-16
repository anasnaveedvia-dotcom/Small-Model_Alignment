import argparse

from alignment.generation import run_generation
from alignment.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--eval_set", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--config", default="configs/generation.yaml")
    parser.add_argument("--sanity", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_generation(
        args.model_path,
        args.adapter_path,
        args.eval_set,
        args.generator,
        args.output_path,
        config,
        args.sanity,
    )


if __name__ == "__main__":
    main()
