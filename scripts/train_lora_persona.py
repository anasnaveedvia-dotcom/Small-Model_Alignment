import argparse

from alignment.lora_persona import run_lora_persona
from alignment.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    run_lora_persona(config, args.output_dir)


if __name__ == "__main__":
    main()
