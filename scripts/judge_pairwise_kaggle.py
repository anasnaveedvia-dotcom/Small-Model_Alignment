import argparse

from alignment.judging import load_judge, run_judging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--judge_model", default="google/gemini-3.6-flash")
    parser.add_argument("--rubric", required=True, choices=["general", "safety", "persona"])
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--sleep_seconds", type=float, default=5.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--reuse_judgments", action="store_true")
    args = parser.parse_args()

    if args.sleep_seconds < 5:
        raise SystemExit("--sleep_seconds must be at least 5")
    if args.batch_size < 1 or args.max_retries < 1:
        raise SystemExit("--batch_size and --max_retries must be at least 1")

    judge = load_judge(args.judge_model)
    run_judging(
        args.pairs_path,
        args.output_path,
        judge,
        args.judge_model,
        args.rubric,
        args.batch_size,
        args.sleep_seconds,
        args.max_retries,
        args.reuse_judgments,
    )


if __name__ == "__main__":
    main()
