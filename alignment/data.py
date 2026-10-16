import os
import random

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer

from alignment.core import format_prompt
from alignment.utils import read_json, save_json, write_jsonl


HUMAN_TAG = "\n\nHuman:"
ASSISTANT_TAG = "\n\nAssistant:"


def parse_ultrachat(example):
    messages = example.get("messages")
    if not messages:
        return None, "missing_messages"

    prompt = None
    response = None
    for message in messages:
        role = message.get("role")
        if prompt is None and role == "user":
            prompt = message.get("content")
            continue
        if prompt is not None and role == "assistant":
            response = message.get("content")
            break

    if prompt is None or prompt.strip() == "":
        return None, "missing_prompt"
    if response is None or response.strip() == "":
        return None, "missing_response"
    return {"prompt": prompt.strip(), "response": response.strip()}, None


def split_hh_transcript(text):
    if text.count(HUMAN_TAG) != 1:
        return None
    if text.count(ASSISTANT_TAG) != 1:
        return None
    human_index = text.find(HUMAN_TAG)
    assistant_index = text.find(ASSISTANT_TAG)
    if human_index > assistant_index:
        return None
    prompt = text[human_index + len(HUMAN_TAG):assistant_index].strip()
    response = text[assistant_index + len(ASSISTANT_TAG):].strip()
    return prompt, response


def parse_hh_rlhf(example):
    chosen_parts = split_hh_transcript(example.get("chosen", ""))
    rejected_parts = split_hh_transcript(example.get("rejected", ""))
    if chosen_parts is None or rejected_parts is None:
        return None, "not_single_turn"

    chosen_prompt, chosen_response = chosen_parts
    rejected_prompt, rejected_response = rejected_parts
    if chosen_prompt != rejected_prompt:
        return None, "prompt_mismatch"
    if chosen_prompt == "":
        return None, "empty_prompt"
    if chosen_response == "" or rejected_response == "":
        return None, "empty_response"
    if chosen_response == rejected_response:
        return None, "identical_responses"
    return {"prompt": chosen_prompt, "chosen": chosen_response, "rejected": rejected_response}, None


def single_turn_assistant_text(messages):
    if not messages or len(messages) != 2:
        return None
    if messages[0].get("role") != "user":
        return None
    if messages[1].get("role") != "assistant":
        return None
    content = messages[1].get("content")
    if content is None:
        return ""
    return content.strip()


def parse_ultrafeedback(example):
    chosen_response = single_turn_assistant_text(example.get("chosen"))
    rejected_response = single_turn_assistant_text(example.get("rejected"))
    if chosen_response is None or rejected_response is None:
        return None, "not_single_turn"

    prompt = example.get("prompt")
    if prompt is None or prompt.strip() == "":
        return None, "empty_prompt"
    if chosen_response == "" or rejected_response == "":
        return None, "empty_response"
    if chosen_response == rejected_response:
        return None, "identical_responses"
    return {"prompt": prompt.strip(), "chosen": chosen_response, "rejected": rejected_response}, None


def get_parser(source):
    if source == "ultrachat":
        return parse_ultrachat
    elif source == "hh_rlhf":
        return parse_hh_rlhf
    elif source == "ultrafeedback":
        return parse_ultrafeedback
    raise SystemExit(f"Unknown data source: {source}")


def sample_split(dataset, parse_function, tokenizer, num_wanted, max_prompt_tokens, seed):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    rows = []
    skipped = {}
    examined = 0
    for index in indices:
        if len(rows) >= num_wanted:
            break
        examined += 1

        parsed, reason = parse_function(dataset[index])
        if parsed is None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        prompt_text = format_prompt(tokenizer, parsed["prompt"])
        prompt_length = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        if prompt_length > max_prompt_tokens:
            skipped["prompt_too_long"] = skipped.get("prompt_too_long", 0) + 1
            continue

        parsed["source_index"] = index
        rows.append(parsed)

    stats = {
        "rows_in_split": len(dataset),
        "rows_examined": examined,
        "rows_kept": len(rows),
        "rows_requested": num_wanted,
        "skipped": skipped,
    }
    return rows, stats


def prepare_run_data(config, output_dir):
    run_name = config["run_name"]
    data_config = config["data"]
    source = data_config["source"]
    seed = config["seed"]
    parse_function = get_parser(source)

    if source == "ultrachat":
        max_prompt_tokens = config["max_length"] - 1
        length_rule = f"drop if templated prompt tokens >= {config['max_length']}"
    else:
        max_prompt_tokens = config["max_prompt_length"]
        length_rule = f"drop if templated prompt tokens > {config['max_prompt_length']}"

    try:
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    except Exception as error:
        raise SystemExit(f"Could not load tokenizer {config['tokenizer_name']}: {error}")

    all_stats = {
        "run_name": run_name,
        "dataset_name": data_config["dataset_name"],
        "dataset_config": data_config.get("dataset_config"),
        "source": source,
        "seed": seed,
        "tokenizer_name": config["tokenizer_name"],
        "procedure": "shuffle split indices with random.Random(seed), parse and filter in that order, keep the first rows that pass",
        "length_rule": length_rule,
        "splits": {},
    }

    split_plan = [
        ("train", data_config["train_split"], data_config["num_train"]),
        ("val", data_config["val_split"], data_config["num_val"]),
    ]
    for split_name, hf_split, num_wanted in split_plan:
        print(f"Loading {data_config['dataset_name']} split {hf_split}", flush=True)
        try:
            dataset = load_dataset(data_config["dataset_name"], data_config.get("dataset_config"), split=hf_split)
        except Exception as error:
            raise SystemExit(f"Could not load {data_config['dataset_name']} split {hf_split}: {error}")

        rows, stats = sample_split(dataset, parse_function, tokenizer, num_wanted, max_prompt_tokens, seed)
        stats["hf_split"] = hf_split
        if len(rows) < num_wanted:
            print(f"Warning: only {len(rows)} of {num_wanted} rows available for {run_name} {split_name}", flush=True)

        output_path = os.path.join(output_dir, f"{run_name}_{split_name}.jsonl")
        write_jsonl(output_path, rows)
        all_stats["splits"][split_name] = stats
        print(f"Wrote {len(rows)} rows to {output_path}: {stats}", flush=True)

    stats_path = os.path.join(output_dir, f"{run_name}_data_stats.json")
    save_json(stats_path, all_stats)
    return all_stats


def build_alpaca_eval_subset(output_path, num_prompts=300, seed=42):
    if os.path.exists(output_path):
        print(f"{output_path} already exists, keeping the saved subset", flush=True)
        return read_json(output_path)

    try:
        source_path = hf_hub_download(repo_id="tatsu-lab/alpaca_eval", filename="alpaca_eval.json", repo_type="dataset")
    except Exception as error:
        print(f"Could not download alpaca_eval.json: {error}", flush=True)
        try:
            files = list_repo_files("tatsu-lab/alpaca_eval", repo_type="dataset")
            print("Files in tatsu-lab/alpaca_eval:", flush=True)
            for name in files:
                print(name, flush=True)
        except Exception as list_error:
            print(f"Could not list repository files: {list_error}", flush=True)
        raise SystemExit("AlpacaEval subset was not created")

    records = read_json(source_path)
    sampled = random.Random(seed).sample(records, num_prompts)

    subset = []
    for position in range(len(sampled)):
        subset.append({
            "id": f"alpaca-eval-{seed}-{position:04d}",
            "instruction": sampled[position]["instruction"],
            "dataset": "alpaca_eval",
        })
    save_json(output_path, subset)
    print(f"Wrote {len(subset)} prompts to {output_path}", flush=True)
    return subset


def to_trl_record(row):
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def load_persona_split(path, num_eval, seed):
    rows = read_json(path)
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)

    train_rows = []
    eval_rows = []
    for position in range(len(indices)):
        row = rows[indices[position]]
        if position < num_eval:
            eval_rows.append(to_trl_record(row))
        else:
            train_rows.append(to_trl_record(row))
    return train_rows, eval_rows
