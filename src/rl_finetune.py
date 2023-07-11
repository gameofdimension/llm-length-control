import json
import random

import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from tqdm.autonotebook import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import create_reference_model, AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


def rl_collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return '\n'.join([
        f"trainable model parameters: {trainable_model_params}",
        f"all model parameters: {all_model_params}",
        f"percentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%",
        "\n",
    ])


def build_ppo_model():
    peft_model_id = "felixdae/lora-cs324-length-control"
    config = PeftConfig.from_pretrained(peft_model_id)

    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    peft_model = PeftModel.from_pretrained(model, peft_model_id, is_trainable=True)
    print(f'PEFT model parameters to be updated:')
    print(print_number_of_trainable_model_parameters(peft_model))

    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        peft_model, is_trainable=True)
    print(f'ppo model parameters to be updated:')
    print(print_number_of_trainable_model_parameters(ppo_model))

    ref_model = create_reference_model(ppo_model)
    print(f'reference model parameters to be updated:')
    print(print_number_of_trainable_model_parameters(ref_model))

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, ppo_model, ref_model


def make_ppo_config():
    learning_rate = 1.41e-5
    max_ppo_epochs = 1
    mini_batch_size = 4
    batch_size = 16

    model_name = "rl-cs324-length-control"
    config = PPOConfig(
        model_name=model_name,
        learning_rate=learning_rate,
        ppo_epochs=max_ppo_epochs,
        mini_batch_size=mini_batch_size,
        batch_size=batch_size,
        # log_with='tensorboard',
        # project_kwargs={"logging_dir": PATH_TO_LOGS},
    )
    return config


def make_data(n: int, path: str):
    lines = []
    for _ in range(n):
        num = random.randint(1, 128)
        inst = {
            'query': f'<len> {num} <text>)',
        }
        lines.append(json.dumps(inst))

    with open(path, 'w') as fp:
        for row in lines:
            print(row, file=fp)


def tk(tokenizer, sample):
    sample['input_ids'] = tokenizer(sample['query'])['input_ids']
    return sample


def prep_data(tokenizer, sample: int, path):
    data = load_dataset('json', data_files=path, split="train")
    data = data.shuffle().select(range(sample))
    data = data.map(lambda x: tk(tokenizer, x), batched=False)
    data.set_format(type="torch")

    return data


def make_ppo_trainer(data_path):
    tokenizer, ppo_model, ref_model = build_ppo_model()
    config = make_ppo_config()

    dataset = prep_data(tokenizer, 1_000, data_path)

    ppo_trainer = PPOTrainer(
        config=config,
        model=ppo_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=rl_collator,
    )
    return tokenizer, ppo_model, ppo_trainer


def compute_reward(tokenizer, prompt: str, output_ids) -> float:
    target = float(prompt.split()[1])
    prefix_token_number = len(tokenizer(prompt)['input_ids'])
    return -(len(output_ids) - (prefix_token_number + 1) - target) ** 2


def train(data_path, max_ppo_steps):
    tokenizer, ppo_model, ppo_trainer = make_ppo_trainer(data_path)

    generation_kwargs = {
        "do_sample": True,
        "top_k": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 1000,
        'pad_token_id': tokenizer.pad_token_id,
    }
    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        # Break when you reach max_steps.
        if step >= max_ppo_steps:
            break

        prompts = batch['query']
        prompt_tensors = batch["input_ids"]

        summary_tensors = []
        reward_tensors = []

        for prompt, prompt_tensor in zip(prompts, prompt_tensors):
            summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)

            output_ids = summary.squeeze()
            summary_tensors.append(output_ids)
            reward = compute_reward(tokenizer, prompt, output_ids)
            reward_tensors.append(torch.tensor(reward))

        # This needs to be called "response".
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]

        # Run PPO step.
        stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
        ppo_trainer.log_stats(stats, batch, reward_tensors)

        print(f'step: {step}, batch size: {len(prompts)}, stats: {stats}')
        print(f'objective/kl: {stats["objective/kl"]}')
        print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
        print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
        print('-' * 100)
        if (1 + step) % 10 == 0:
            ppo_model.push_to_hub("felixdae/rl-cs324-length-control")

    ppo_model.push_to_hub("felixdae/rl-cs324-length-control")
