import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alignment.utils import read_jsonl


def collect_series(records, split, key):
    steps = []
    values = []
    for record in records:
        if record.get("split") != split:
            continue
        value = record.get(key)
        if value is None:
            continue
        steps.append(record["step"])
        values.append(value)
    return steps, values


def find_best_step(records):
    best_step = None
    for record in records:
        if record.get("split") != "val":
            continue
        if record.get("best"):
            best_step = record["step"]
    return best_step


def mark_best(axis, best_step):
    if best_step is None:
        return
    axis.axvline(best_step, color="gray", linestyle="--", linewidth=1, label=f"best checkpoint (step {best_step})")


def plot_loss(records, run_name, best_step, output_path):
    train_steps, train_values = collect_series(records, "train", "loss")
    val_steps, val_values = collect_series(records, "val", "loss")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    if len(train_steps) > 0:
        axis.plot(train_steps, train_values, label="train loss", linewidth=1)
    if len(val_steps) > 0:
        axis.plot(val_steps, val_values, marker="o", label="validation loss", linewidth=1.5)
    mark_best(axis, best_step)
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("loss")
    axis.set_title(f"{run_name}: loss (raw values, no smoothing)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved {output_path}", flush=True)


def plot_rewards(records, run_name, best_step, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    panels = [
        ("reward_accuracy", "reward accuracy (r_w > r_l)"),
        ("reward_margin", "reward margin (mean r_w - r_l)"),
    ]
    for panel_index in range(len(panels)):
        key, label = panels[panel_index]
        axis = axes[panel_index]
        train_steps, train_values = collect_series(records, "train", key)
        val_steps, val_values = collect_series(records, "val", key)
        if len(train_steps) > 0:
            axis.plot(train_steps, train_values, label=f"train {key}", linewidth=1, alpha=0.6)
        if len(val_steps) > 0:
            axis.plot(val_steps, val_values, marker="o", label=f"validation {key}", linewidth=1.5)
        mark_best(axis, best_step)
        axis.set_xlabel("optimizer step")
        axis.set_ylabel(label)
        axis.set_title(f"{run_name}: {label}")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    args = parser.parse_args()

    records = read_jsonl(args.metrics_path)
    if len(records) == 0:
        raise SystemExit(f"No records found in {args.metrics_path}")
    os.makedirs(args.output_dir, exist_ok=True)

    best_step = find_best_step(records)
    plot_loss(records, args.run_name, best_step, os.path.join(args.output_dir, f"{args.run_name}_loss.png"))

    has_rewards = False
    for record in records:
        if record.get("reward_accuracy") is not None:
            has_rewards = True
            break
    if has_rewards:
        plot_rewards(records, args.run_name, best_step, os.path.join(args.output_dir, f"{args.run_name}_rewards.png"))


if __name__ == "__main__":
    main()
