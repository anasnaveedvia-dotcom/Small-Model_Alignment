import math
import os
import shutil
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment import core
from alignment.sft import make_micro_batches, save_checkpoint, write_prompt_template
from alignment.utils import append_jsonl, log_line, read_jsonl, save_json, set_seed


def load_pairs(tokenizer, path):
    rows = read_jsonl(path)
    prompts = []
    chosen = []
    rejected = []
    for row in rows:
        prompts.append(core.format_prompt(tokenizer, row["prompt"]))
        chosen.append(row["chosen"])
        rejected.append(row["rejected"])
    return prompts, chosen, rejected


def evaluate_dpo(policy_model, reference_model, tokenizer, prompts, chosen, rejected, config, device, use_fp16):
    policy_model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_margin = 0.0
    total_chosen_reward = 0.0
    total_rejected_reward = 0.0
    count = 0
    batch_size = config["eval_batch_size"]
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = core.build_dpo_batch(
                tokenizer,
                prompts[start:start + batch_size],
                chosen[start:start + batch_size],
                rejected[start:start + batch_size],
                config["max_length"],
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                losses, chosen_rewards, rejected_rewards = core.dpo_forward(
                    policy_model, reference_model, batch, config["beta"]
                )
            total_loss += losses.sum().item()
            total_correct += (chosen_rewards > rejected_rewards).float().sum().item()
            total_margin += (chosen_rewards - rejected_rewards).sum().item()
            total_chosen_reward += chosen_rewards.sum().item()
            total_rejected_reward += rejected_rewards.sum().item()
            count += losses.shape[0]
    policy_model.train()

    if count == 0:
        return None
    results = {
        "loss": total_loss / count,
        "reward_accuracy": total_correct / count,
        "reward_margin": total_margin / count,
        "chosen_reward": total_chosen_reward / count,
        "rejected_reward": total_rejected_reward / count,
    }
    return results


def run_dpo(config, sft_path, data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train_log.txt")
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    seed = config["seed"]
    set_seed(seed)

    if torch.cuda.is_available():
        policy_device = torch.device("cuda:0")
        if torch.cuda.device_count() > 1:
            reference_device = torch.device("cuda:1")
        else:
            reference_device = torch.device("cuda:0")
    else:
        policy_device = torch.device("cpu")
        reference_device = torch.device("cpu")
    use_fp16 = bool(config["fp16"]) and policy_device.type == "cuda"

    run_name = config["run_name"]
    stats_path = os.path.join(data_dir, f"{run_name}_data_stats.json")
    if os.path.exists(stats_path):
        shutil.copy(stats_path, os.path.join(output_dir, "data_stats.json"))

    try:
        tokenizer = AutoTokenizer.from_pretrained(sft_path)
        policy_model = AutoModelForCausalLM.from_pretrained(sft_path, dtype=torch.float32)
        reference_model = AutoModelForCausalLM.from_pretrained(sft_path, dtype=torch.float32)
    except Exception as error:
        raise SystemExit(f"Could not load SFT checkpoint from {sft_path}: {error}")

    policy_model.to(policy_device)
    policy_model.train()
    reference_model.to(reference_device)
    reference_model.eval()
    reference_model.requires_grad_(False)
    write_prompt_template(tokenizer, output_dir)

    train_prompts, train_chosen, train_rejected = load_pairs(tokenizer, os.path.join(data_dir, f"{run_name}_train.jsonl"))
    val_prompts, val_chosen, val_rejected = load_pairs(tokenizer, os.path.join(data_dir, f"{run_name}_val.jsonl"))

    batch_size = config["per_device_batch_size"]
    accumulation = config["gradient_accumulation_steps"]
    num_epochs = config["num_epochs"]
    micro_batches_per_epoch = math.ceil(len(train_prompts) / batch_size)
    steps_per_epoch = math.ceil(micro_batches_per_epoch / accumulation)
    total_steps = steps_per_epoch * num_epochs

    optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    config_used = dict(config)
    config_used["sft_path"] = sft_path
    config_used["reference_checkpoint"] = sft_path
    config_used["data_dir"] = data_dir
    config_used["output_dir"] = output_dir
    config_used["num_train_pairs"] = len(train_prompts)
    config_used["num_val_pairs"] = len(val_prompts)
    config_used["policy_device"] = str(policy_device)
    config_used["reference_device"] = str(reference_device)
    config_used["reference_dtype_used"] = str(torch.float32)
    config_used["reference_precision"] = "fp32 frozen weights, forward under the same fp16 autocast as the policy" if use_fp16 else "fp32"
    config_used["effective_batch_size"] = batch_size * accumulation
    config_used["steps_per_epoch"] = steps_per_epoch
    config_used["total_optimizer_steps"] = total_steps
    config_used["precision"] = "fp32 policy weights, fp16 autocast with GradScaler" if use_fp16 else "fp32"
    config_used["checkpoint_selection"] = "best validation reward accuracy (strict >), evaluations after step 0 only"
    config_used["torch_version"] = torch.__version__
    config_used["transformers_version"] = transformers.__version__
    save_json(os.path.join(output_dir, "config_used.json"), config_used)

    log_line(log_path, f"DPO {run_name}: {len(train_prompts)} train pairs, {len(val_prompts)} val pairs, {total_steps} optimizer steps")

    best_accuracy = None
    global_step = 0
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)
    results = evaluate_dpo(
        policy_model, reference_model, tokenizer, val_prompts, val_chosen, val_rejected, config, policy_device, use_fp16
    )
    if results is not None:
        record = {"split": "val", "step": 0, "epoch": 0, "best": False, "elapsed_seconds": round(time.time() - start_time, 1)}
        record.update(results)
        append_jsonl(metrics_path, record)
        log_line(log_path, f"step 0 val {results}")

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(num_epochs):
        micro_batches = make_micro_batches(len(train_prompts), batch_size, seed + epoch)
        for group_start in range(0, len(micro_batches), accumulation):
            group = micro_batches[group_start:group_start + accumulation]
            group_loss = 0.0
            group_correct = 0.0
            group_margin = 0.0
            group_count = 0

            for indices in group:
                prompts = []
                chosen = []
                rejected = []
                for index in indices:
                    prompts.append(train_prompts[index])
                    chosen.append(train_chosen[index])
                    rejected.append(train_rejected[index])
                batch = core.build_dpo_batch(tokenizer, prompts, chosen, rejected, config["max_length"])
                with torch.autocast(device_type=policy_device.type, dtype=torch.float16, enabled=use_fp16):
                    losses, chosen_rewards, rejected_rewards = core.dpo_forward(
                        policy_model, reference_model, batch, config["beta"]
                    )
                loss = losses.mean()
                scaler.scale(loss / len(group)).backward()

                group_loss += loss.item() / len(group)
                group_correct += (chosen_rewards > rejected_rewards).float().sum().item()
                group_margin += (chosen_rewards - rejected_rewards).sum().item()
                group_count += losses.shape[0]

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % config["log_every_steps"] == 0 or global_step == total_steps:
                record = {
                    "split": "train",
                    "step": global_step,
                    "epoch": epoch,
                    "loss": group_loss,
                    "reward_accuracy": group_correct / group_count,
                    "reward_margin": group_margin / group_count,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm),
                    "elapsed_seconds": round(time.time() - start_time, 1),
                }
                append_jsonl(metrics_path, record)
                log_line(log_path, f"step {global_step}/{total_steps} train_loss {group_loss:.4f} acc {record['reward_accuracy']:.3f}")

            if global_step % config["eval_every_steps"] == 0 or global_step == total_steps:
                optimizer.zero_grad(set_to_none=True)
                results = evaluate_dpo(
                    policy_model, reference_model, tokenizer, val_prompts, val_chosen, val_rejected, config, policy_device, use_fp16
                )
                if results is None:
                    continue
                is_best = best_accuracy is None or results["reward_accuracy"] > best_accuracy
                record = {"split": "val", "step": global_step, "epoch": epoch, "best": is_best, "elapsed_seconds": round(time.time() - start_time, 1)}
                record.update(results)
                append_jsonl(metrics_path, record)
                log_line(log_path, f"step {global_step}/{total_steps} val {results} best {is_best}")
                if is_best:
                    best_accuracy = results["reward_accuracy"]
                    save_checkpoint(policy_model, tokenizer, os.path.join(output_dir, "best"))
                    log_line(log_path, f"saved best checkpoint at step {global_step}")

    save_checkpoint(policy_model, tokenizer, os.path.join(output_dir, "final"))
    log_line(log_path, f"saved final checkpoint at step {global_step}, best val reward accuracy {best_accuracy}")
