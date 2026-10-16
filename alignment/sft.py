import math
import os
import random
import shutil
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment import core
from alignment.utils import append_jsonl, log_line, make_warmup_cosine_scheduler, read_jsonl, save_json, set_seed


def make_micro_batches(num_rows, batch_size, seed):
    order = list(range(num_rows))
    random.Random(seed).shuffle(order)
    micro_batches = []
    for start in range(0, num_rows, batch_size):
        micro_batches.append(order[start:start + batch_size])
    return micro_batches


def save_checkpoint(model, tokenizer, path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def write_prompt_template(tokenizer, output_dir):
    rendered = core.format_prompt(tokenizer, "{instruction}")
    lines = [
        "Prompt format: tokenizer.apply_chat_template with one user message and add_generation_prompt=True.",
        "No system message is passed by the code; any system text below is inserted by the tokenizer's chat template.",
        "The response text follows the rendered prompt, then tokenizer.eos_token_id is appended during training.",
        f"eos_token={tokenizer.eos_token!r} eos_token_id={tokenizer.eos_token_id} pad_token={tokenizer.pad_token!r} pad_token_id={tokenizer.pad_token_id}",
        "Rendered template for instruction '{instruction}':",
        rendered,
    ]
    with open(os.path.join(output_dir, "prompt_template.txt"), "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def evaluate_sft(model, tokenizer, prompts, responses, config, device, use_fp16):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    batch_size = config["eval_batch_size"]
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = core.build_sft_batch(
                tokenizer,
                prompts[start:start + batch_size],
                responses[start:start + batch_size],
                config["max_length"],
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                token_logps, label_mask = core.response_token_logps(model, batch)
            total_nll += -token_logps.sum().item()
            total_tokens += label_mask.sum().item()
    model.train()
    if total_tokens == 0:
        return 0.0
    return total_nll / total_tokens


def run_sft(config, data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train_log.txt")
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    seed = config["seed"]
    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    use_fp16 = bool(config["fp16"]) and device.type == "cuda"

    run_name = config["run_name"]
    train_rows = read_jsonl(os.path.join(data_dir, f"{run_name}_train.jsonl"))
    val_rows = read_jsonl(os.path.join(data_dir, f"{run_name}_val.jsonl"))
    stats_path = os.path.join(data_dir, f"{run_name}_data_stats.json")
    if os.path.exists(stats_path):
        shutil.copy(stats_path, os.path.join(output_dir, "data_stats.json"))

    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        model = AutoModelForCausalLM.from_pretrained(config["model_name"], dtype=torch.float32)
    except Exception as error:
        raise SystemExit(f"Could not load model {config['model_name']}: {error}")
    model.to(device)
    model.train()
    write_prompt_template(tokenizer, output_dir)

    train_prompts = []
    train_responses = []
    for row in train_rows:
        train_prompts.append(core.format_prompt(tokenizer, row["prompt"]))
        train_responses.append(row["response"])
    val_prompts = []
    val_responses = []
    for row in val_rows:
        val_prompts.append(core.format_prompt(tokenizer, row["prompt"]))
        val_responses.append(row["response"])

    batch_size = config["per_device_batch_size"]
    accumulation = config["gradient_accumulation_steps"]
    num_epochs = config["num_epochs"]
    micro_batches_per_epoch = math.ceil(len(train_prompts) / batch_size)
    steps_per_epoch = math.ceil(micro_batches_per_epoch / accumulation)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(round(config["warmup_ratio"] * total_steps))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = make_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    config_used = dict(config)
    config_used["data_dir"] = data_dir
    config_used["output_dir"] = output_dir
    config_used["num_train_examples"] = len(train_prompts)
    config_used["num_val_examples"] = len(val_prompts)
    config_used["num_gpus_used"] = 1 if device.type == "cuda" else 0
    config_used["effective_batch_size"] = batch_size * accumulation
    config_used["steps_per_epoch"] = steps_per_epoch
    config_used["total_optimizer_steps"] = total_steps
    config_used["warmup_steps"] = warmup_steps
    config_used["precision"] = "fp32 weights, fp16 autocast with GradScaler" if use_fp16 else "fp32"
    config_used["loss_reduction"] = "mean over all unmasked response tokens in the batch"
    config_used["device"] = str(device)
    config_used["torch_version"] = torch.__version__
    config_used["transformers_version"] = transformers.__version__
    save_json(os.path.join(output_dir, "config_used.json"), config_used)

    log_line(log_path, f"SFT: {len(train_prompts)} train, {len(val_prompts)} val, {total_steps} optimizer steps, warmup {warmup_steps}")

    best_val_loss = None
    global_step = 0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        micro_batches = make_micro_batches(len(train_prompts), batch_size, seed + epoch)
        for group_start in range(0, len(micro_batches), accumulation):
            group = micro_batches[group_start:group_start + accumulation]
            group_loss = 0.0

            for indices in group:
                prompts = []
                responses = []
                for index in indices:
                    prompts.append(train_prompts[index])
                    responses.append(train_responses[index])
                batch = core.build_sft_batch(tokenizer, prompts, responses, config["max_length"])
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                    loss = core.sft_loss_from_batch(model, batch)
                scaled_loss = loss / len(group)
                scaler.scale(scaled_loss).backward()
                group_loss += loss.item() / len(group)

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if global_step % config["log_every_steps"] == 0 or global_step == total_steps:
                record = {
                    "split": "train",
                    "step": global_step,
                    "epoch": epoch,
                    "loss": group_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm),
                    "elapsed_seconds": round(time.time() - start_time, 1),
                }
                append_jsonl(metrics_path, record)
                log_line(log_path, f"step {global_step}/{total_steps} train_loss {group_loss:.4f} lr {record['lr']:.2e}")

            if global_step % config["eval_every_steps"] == 0 or global_step == total_steps:
                val_loss = evaluate_sft(model, tokenizer, val_prompts, val_responses, config, device, use_fp16)
                is_best = best_val_loss is None or val_loss < best_val_loss
                record = {
                    "split": "val",
                    "step": global_step,
                    "epoch": epoch,
                    "loss": val_loss,
                    "best": is_best,
                    "elapsed_seconds": round(time.time() - start_time, 1),
                }
                append_jsonl(metrics_path, record)
                log_line(log_path, f"step {global_step}/{total_steps} val_loss {val_loss:.4f} best {is_best}")
                if is_best:
                    best_val_loss = val_loss
                    save_checkpoint(model, tokenizer, os.path.join(output_dir, "best"))
                    log_line(log_path, f"saved best checkpoint at step {global_step}")

    save_checkpoint(model, tokenizer, os.path.join(output_dir, "final"))
    log_line(log_path, f"saved final checkpoint at step {global_step}, best val loss {best_val_loss}")
