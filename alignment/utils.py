import json
import math
import os
import random

import numpy as np
import torch
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except OSError as error:
        raise SystemExit(f"Could not read config file {path}: {error}")
    if not isinstance(config, dict):
        raise SystemExit(f"Config file {path} does not contain a mapping")
    return config


def make_parent_dir(path):
    parent = os.path.dirname(path)
    if parent != "":
        os.makedirs(parent, exist_ok=True)


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read JSON file {path}: {error}")
    return data


def save_json(path, data):
    make_parent_dir(path)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=str)


def read_jsonl(path):
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line == "":
                    continue
                rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Could not read JSONL file {path}: {error}")
    return rows


def write_jsonl(path, rows):
    make_parent_dir(path)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    make_parent_dir(path)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_line(log_path, text):
    print(text, flush=True)
    make_parent_dir(log_path)
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(text + "\n")


def make_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    def lr_factor(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_steps = total_steps - warmup_steps
        if decay_steps <= 0:
            return 1.0
        progress = (step - warmup_steps) / decay_steps
        if progress > 1.0:
            progress = 1.0
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
