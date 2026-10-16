import torch
import torch.nn.functional as F


def format_prompt(tokenizer, instruction):
    messages = [{"role": "user", "content": instruction}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt_text


def get_pad_id(tokenizer):
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    return pad_id


def get_model_device(model):
    for parameter in model.parameters():
        return parameter.device
    return torch.device("cpu")


def build_lm_batch(tokenizer, prompts, responses, max_length, pad_to_max_length=False):
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    if max_length < 1:
        raise ValueError("max_length must be at least 1")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = get_pad_id(tokenizer)

    all_input_ids = []
    all_prompt_lengths = []
    longest = 1
    for index in range(len(prompts)):
        prompt_ids = list(tokenizer.encode(prompts[index], add_special_tokens=False))
        response_ids = list(tokenizer.encode(responses[index], add_special_tokens=False))
        if eos_id is not None:
            response_ids.append(eos_id)

        input_ids = prompt_ids + response_ids
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        prompt_length = len(prompt_ids)
        if prompt_length > len(input_ids):
            prompt_length = len(input_ids)

        all_input_ids.append(input_ids)
        all_prompt_lengths.append(prompt_length)
        if len(input_ids) > longest:
            longest = len(input_ids)

    if pad_to_max_length:
        longest = max_length

    batch_size = len(prompts)
    input_tensor = torch.full((batch_size, longest), pad_id, dtype=torch.long)
    attention_tensor = torch.zeros((batch_size, longest), dtype=torch.long)
    labels_tensor = torch.full((batch_size, longest), -100, dtype=torch.long)

    for row in range(batch_size):
        input_ids = all_input_ids[row]
        prompt_length = all_prompt_lengths[row]
        sequence_length = len(input_ids)
        if sequence_length == 0:
            continue
        input_tensor[row, :sequence_length] = torch.tensor(input_ids, dtype=torch.long)
        attention_tensor[row, :sequence_length] = 1
        if prompt_length < sequence_length:
            labels_tensor[row, prompt_length:sequence_length] = input_tensor[row, prompt_length:sequence_length]

    batch = {
        "input_ids": input_tensor,
        "attention_mask": attention_tensor,
        "labels": labels_tensor,
    }
    return batch


def build_sft_batch(tokenizer, prompts, responses, max_length, pad_to_max_length=False):
    return build_lm_batch(tokenizer, prompts, responses, max_length, pad_to_max_length)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length, pad_to_max_length=False):
    if len(prompts) != len(chosen_responses):
        raise ValueError("prompts and chosen_responses must have the same length")
    if len(prompts) != len(rejected_responses):
        raise ValueError("prompts and rejected_responses must have the same length")

    chosen_batch = build_lm_batch(tokenizer, prompts, chosen_responses, max_length, pad_to_max_length)
    rejected_batch = build_lm_batch(tokenizer, prompts, rejected_responses, max_length, pad_to_max_length)
    return {"chosen": chosen_batch, "rejected": rejected_batch}


def response_token_logps(model, batch):
    device = get_model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits.float()

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    label_mask = shifted_labels.ne(-100)
    safe_labels = torch.where(label_mask, shifted_labels, torch.zeros_like(shifted_labels))

    log_probs = F.log_softmax(shifted_logits, dim=-1)
    token_logps = torch.gather(log_probs, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * label_mask.float()
    return token_logps, label_mask


def logprobs_from_batch(model, batch):
    token_logps, label_mask = response_token_logps(model, batch)
    return token_logps.sum(dim=1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length, pad_to_max_length=False):
    batch = build_lm_batch(tokenizer, prompts, responses, max_length, pad_to_max_length)
    return logprobs_from_batch(model, batch)


def sft_loss_from_batch(model, batch):
    token_logps, label_mask = response_token_logps(model, batch)
    total_logp = token_logps.sum()
    token_count = label_mask.sum().clamp_min(1)
    loss = -total_logp / token_count
    return loss


def compute_sft_loss(model, tokenizer, prompts, responses, max_length, pad_to_max_length=False):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length, pad_to_max_length)
    return sft_loss_from_batch(model, batch)


def dpo_losses_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    reference_chosen_logps = reference_chosen_logps.to(policy_chosen_logps.device)
    reference_rejected_logps = reference_rejected_logps.to(policy_chosen_logps.device)

    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps
    per_example_losses = -F.logsigmoid(beta * (policy_gap - reference_gap))

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return per_example_losses, chosen_rewards, rejected_rewards


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    per_example_losses, chosen_rewards, rejected_rewards = dpo_losses_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )
    return per_example_losses.mean()


def dpo_metrics(chosen_rewards, rejected_rewards):
    correct = (chosen_rewards > rejected_rewards).float()
    margins = chosen_rewards - rejected_rewards
    metrics = {
        "reward_accuracy": correct.mean().item(),
        "reward_margin": margins.mean().item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
    }
    return metrics


def dpo_forward(policy_model, reference_model, batch, beta):
    policy_chosen_logps = logprobs_from_batch(policy_model, batch["chosen"])
    policy_rejected_logps = logprobs_from_batch(policy_model, batch["rejected"])

    with torch.no_grad():
        reference_chosen_logps = logprobs_from_batch(reference_model, batch["chosen"])
        reference_rejected_logps = logprobs_from_batch(reference_model, batch["rejected"])

    per_example_losses, chosen_rewards, rejected_rewards = dpo_losses_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )
    return per_example_losses, chosen_rewards, rejected_rewards


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
    pad_to_max_length=False,
):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length, pad_to_max_length)
    per_example_losses, chosen_rewards, rejected_rewards = dpo_forward(policy_model, reference_model, batch, beta)
    return per_example_losses.mean()
