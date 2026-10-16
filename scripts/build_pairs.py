import argparse

from alignment.judging import build_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_path", required=True)
    parser.add_argument("--baseline_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_pairs(args.candidate_path, args.baseline_path, args.output_path, args.seed)


if __name__ == "__main__":
    main()
