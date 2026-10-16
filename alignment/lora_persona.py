import os
import shutil

import peft
import torch
import transformers
import trl
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from alignment.data import load_persona_split
from alignment.utils import append_jsonl, log_line, save_json, set_seed


def checkpoint_step(checkpoint_path):
    if checkpoint_path is None:
        return None
    name = os.path.basename(os.path.normpath(checkpoint_path))
    if not name.startswith("checkpoint-"):
        return None
    return int(name[len("checkpoint-"):])


def write_metrics(log_history, metrics_path, best_step):
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
    for entry in log_history:
        if "eval_loss" in entry:
            record = {
                "split": "val",
                "step": entry.get("step"),
                "epoch": entry.get("epoch"),
                "loss": entry["eval_loss"],
                "reward_accuracy": entry.get("eval_rewards/accuracies"),
                "reward_margin": entry.get("eval_rewards/margins"),
                "best": entry.get("step") == best_step,
            }
        elif "loss" in entry:
            record = {
                "split": "train",
                "step": entry.get("step"),
                "epoch": entry.get("epoch"),
                "loss": entry["loss"],
                "reward_accuracy": entry.get("rewards/accuracies"),
                "reward_margin": entry.get("rewards/margins"),
                "lr": entry.get("learning_rate"),
            }
        else:
            continue
        record["trainer_log"] = entry
        append_jsonl(metrics_path, record)


def copy_adapter(source_dir, target_dir, tokenizer):
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    for name in ["adapter_model.safetensors", "adapter_config.json", "README.md"]:
        source_path = os.path.join(source_dir, name)
        if not os.path.exists(source_path):
            continue
        shutil.copy(source_path, os.path.join(target_dir, name))
    tokenizer.save_pretrained(target_dir)


def run_lora_persona(config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train_log.txt")
    seed = config["seed"]
    set_seed(seed)

    train_rows, eval_rows = load_persona_split(config["data_path"], config["num_eval"], seed)
    log_line(log_path, f"LoRA persona: {len(train_rows)} train pairs, {len(eval_rows)} eval pairs")

    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        model = AutoModelForCausalLM.from_pretrained(config["model_name"], dtype=torch.float32)
    except Exception as error:
        raise SystemExit(f"Could not load model {config['model_name']}: {error}")

    use_fp16 = bool(config["fp16"]) and torch.cuda.is_available()

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=os.path.join(output_dir, "trainer"),
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_batch_size"],
        per_device_eval_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_epochs"],
        beta=config["beta"],
        max_length=config["max_length"],
        lr_scheduler_type=config["lr_schedule"],
        warmup_ratio=config["warmup_ratio"],
        fp16=use_fp16,
        bf16=False,
        logging_steps=config["log_every_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="rewards/accuracies",
        greater_is_better=True,
        report_to="none",
        seed=seed,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows),
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainable_params, all_params = trainer.model.get_nb_trainable_parameters()
    log_line(log_path, f"trainable parameters {trainable_params} of {all_params}")

    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    log_line(log_path, f"saved final adapter to {final_dir}")

    best_checkpoint = trainer.state.best_model_checkpoint
    best_step = checkpoint_step(best_checkpoint)
    if best_checkpoint is not None:
        copy_adapter(best_checkpoint, os.path.join(output_dir, "best"), tokenizer)
        log_line(log_path, f"saved best adapter from {best_checkpoint} (eval reward accuracy {trainer.state.best_metric})")
    else:
        copy_adapter(final_dir, os.path.join(output_dir, "best"), tokenizer)
        log_line(log_path, "no best checkpoint recorded, best is a copy of final")

    write_metrics(trainer.state.log_history, os.path.join(output_dir, "metrics.jsonl"), best_step)

    config_used = dict(config)
    config_used["output_dir"] = output_dir
    config_used["num_train_pairs"] = len(train_rows)
    config_used["num_eval_pairs"] = len(eval_rows)
    config_used["trainable_parameters"] = trainable_params
    config_used["all_parameters"] = all_params
    config_used["num_gpus_used"] = training_args.n_gpu
    config_used["effective_batch_size"] = config["per_device_batch_size"] * config["gradient_accumulation_steps"] * max(1, training_args.n_gpu)
    config_used["total_optimizer_steps"] = trainer.state.global_step
    config_used["precision"] = "fp32 weights, fp16 mixed precision" if use_fp16 else "fp32"
    config_used["reference_model"] = "base model with LoRA adapters disabled (TRL default when peft_config is passed)"
    config_used["best_checkpoint"] = best_checkpoint
    config_used["best_eval_reward_accuracy"] = trainer.state.best_metric
    config_used["lora_config"] = lora_config.to_dict()
    config_used["dpo_config"] = training_args.to_dict()
    config_used["torch_version"] = torch.__version__
    config_used["transformers_version"] = transformers.__version__
    config_used["trl_version"] = trl.__version__
    config_used["peft_version"] = peft.__version__
    save_json(os.path.join(output_dir, "config_used.json"), config_used)

    trainer_dir = os.path.join(output_dir, "trainer")
    if os.path.isdir(trainer_dir):
        shutil.rmtree(trainer_dir)
