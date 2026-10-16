import os
import time

import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment import core
from alignment.utils import read_json, save_json


def get_stop_ids(tokenizer):
    stop_ids = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)
    vocab = tokenizer.get_vocab()
    for token in ["<|im_end|>", "<|endoftext|>"]:
        if token not in vocab:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id in stop_ids:
            continue
        stop_ids.append(token_id)
    return stop_ids


def get_dtype(name):
    if name == "float16":
        return torch.float16
    elif name == "float32":
        return torch.float32
    raise SystemExit(f"Unknown dtype in generation config: {name}")


def build_left_padded_batch(tokenizer, prompts, pad_id):
    all_ids = []
    longest = 1
    for prompt in prompts:
        prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        all_ids.append(prompt_ids)
        if len(prompt_ids) > longest:
            longest = len(prompt_ids)

    input_ids = torch.full((len(prompts), longest), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), longest), dtype=torch.long)
    for row in range(len(all_ids)):
        prompt_ids = all_ids[row]
        if len(prompt_ids) == 0:
            continue
        start = longest - len(prompt_ids)
        input_ids[row, start:] = torch.tensor(prompt_ids, dtype=torch.long)
        attention_mask[row, start:] = 1
    return input_ids, attention_mask


def settings_path_for(output_path):
    stem, extension = os.path.splitext(output_path)
    return stem + "_settings.json"


def run_generation(model_path, adapter_path, eval_set_path, generator, output_path, config, sanity):
    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping generation", flush=True)
        return

    eval_rows = read_json(eval_set_path)
    if sanity is not None and sanity > 0:
        eval_rows = eval_rows[:sanity]

    eval_set_name = os.path.splitext(os.path.basename(eval_set_path))[0]
    max_new_tokens = config["max_new_tokens"].get(eval_set_name)
    if max_new_tokens is None:
        raise SystemExit(f"No max_new_tokens entry for eval set {eval_set_name} in the generation config")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    dtype = get_dtype(config["dtype"])
    if device.type == "cpu":
        dtype = torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, adapter_path)
    except Exception as error:
        raise SystemExit(f"Could not load model {model_path} (adapter {adapter_path}): {error}")
    model.to(device)
    model.eval()

    stop_ids = get_stop_ids(tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    prompts = []
    for row in eval_rows:
        prompts.append(core.format_prompt(tokenizer, row["instruction"]))

    results = []
    batch_size = config["batch_size"]
    start_time = time.time()
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        input_ids, attention_mask = build_left_padded_batch(tokenizer, batch_prompts, pad_id)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=config["do_sample"],
                repetition_penalty=config["repetition_penalty"],
                no_repeat_ngram_size=config["no_repeat_ngram_size"],
                eos_token_id=stop_ids,
                pad_token_id=pad_id,
            )
        new_tokens = output_ids[:, input_ids.shape[1]:]

        for offset in range(len(batch_prompts)):
            row = eval_rows[start + offset]
            text = tokenizer.decode(new_tokens[offset], skip_special_tokens=True).strip()
            record = {
                "id": row["id"],
                "instruction": row["instruction"],
                "output": text,
                "generator": generator,
                "dataset": row.get("dataset", eval_set_name),
            }
            if "category" in row:
                record["category"] = row["category"]
            if "expected_behavior" in row:
                record["expected_behavior"] = row["expected_behavior"]
            results.append(record)

            if sanity is not None and sanity > 0:
                print("=" * 80, flush=True)
                print(f"[{row['id']}] {row['instruction']}", flush=True)
                print("-" * 80, flush=True)
                print(text, flush=True)

        print(f"{generator}: generated {len(results)}/{len(prompts)}", flush=True)

    save_json(output_path, results)

    settings = {
        "generator": generator,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "eval_set": eval_set_path,
        "num_prompts": len(results),
        "sanity": sanity,
        "decoding": "greedy" if not config["do_sample"] else "sampling",
        "do_sample": config["do_sample"],
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": config["repetition_penalty"],
        "no_repeat_ngram_size": config["no_repeat_ngram_size"],
        "temperature": None,
        "top_p": None,
        "batch_size": batch_size,
        "padding_side": "left",
        "dtype": str(dtype),
        "stop_token_ids": stop_ids,
        "stop_tokens": tokenizer.convert_ids_to_tokens(stop_ids),
        "pad_token_id": pad_id,
        "prompt_format": "tokenizer.apply_chat_template, one user message, add_generation_prompt=True",
        "rendered_prompt_template": core.format_prompt(tokenizer, "{instruction}"),
        "elapsed_seconds": round(time.time() - start_time, 1),
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    save_json(settings_path_for(output_path), settings)
    print(f"Wrote {len(results)} generations to {output_path}", flush=True)
