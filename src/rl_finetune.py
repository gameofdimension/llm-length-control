import json
import random

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import create_reference_model, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler

from src.common import prepare_data

from datasets import load_dataset


def rl_collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def build_ppo_model():
    peft_model_id = "felixdae/lora-cs324-length-control"
    config = PeftConfig.from_pretrained(peft_model_id)

    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    peft_model = PeftModel.from_pretrained(model, peft_model_id, is_trainable=True)
    ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        peft_model, is_trainable=True)
    ref_model = create_reference_model(ppo_model)
    # model = model.to(device)
    # model.eval()

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
        batch_size=batch_size
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
    sample['input_ids'] = tokenizer(sample['query'])
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

    # context_length = 128
    # tokenized_datasets = prepare_data(context_length, tokenizer, 100000, 1000)
    dataset = prep_data(tokenizer, 100_000, data_path)

    ppo_trainer = PPOTrainer(
        config=config,
        model=ppo_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=rl_collator,
    )
    return tokenizer, ppo_trainer


def compute_reward(tokenizer, prompt: str, output_ids):
    target = float(prompt.split()[1])
    prefix_token_number = len(tokenizer(prompt)['input_ids'])
    return -(len(output_ids) - (prefix_token_number + 1) - target) ** 2


def train(data_path):
    # output_min_length = 100
    # output_max_length = 400
    # output_length_sampler = LengthSampler(output_min_length, output_max_length)

    # generation_kwargs = {
    #     "min_length": 5,
    #     "top_k": 0.0,
    #     "top_p": 1.0,
    #     "do_sample": True
    # }

    generation_kwargs = {
        "do_sample": True,
        "top_k": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 1000,
    }

    # reward_kwargs = {
    #     "top_k": None,  # Return all scores.
    #     "function_to_apply": "none",  # You want the raw logits without softmax.
    #     "batch_size": 16
    # }

    max_ppo_steps = 10
    tokenizer, ppo_trainer = make_ppo_trainer(data_path)
    for step, batch in enumerate(ppo_trainer.dataloader):
        # Break when you reach max_steps.
        if step >= max_ppo_steps:
            break

        prompts = batch['query']
        prompt_tensors = batch["input_ids"]

        # Get response from FLAN-T5/PEFT LLM.
        summary_tensors = []
        reward_tensors = []

        for prompt, prompt_tensor in zip(prompts, prompt_tensors):
            # max_new_tokens = output_length_sampler()

            # generation_kwargs["max_new_tokens"] = max_new_tokens
            print(prompt, prompt_tensor)
            summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)

            # summary_tensors.append(summary.squeeze()[-max_new_tokens:])
            output_ids = summary.squeeze()
            # reward = len(output_ids)
            summary_tensors.append(output_ids)
            reward_tensors = [torch.tensor(compute_reward(tokenizer, prompt, output_ids))]

        # This needs to be called "response".
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]

        # Compute reward outputs.
        query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]
        # rewards = sentiment_pipe(query_response_pairs, **reward_kwargs)

        # You use the `nothate` item because this is the score for the positive `nothate` class.
        # reward_tensors = [torch.tensor(reward[not_hate_index]["score"]) for reward in rewards]

        # Run PPO step.
        stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
        ppo_trainer.log_stats(stats, batch, reward_tensors)

        print(f'objective/kl: {stats["objective/kl"]}')
        print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
        print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
        print('-' * 100)

# if __name__ == '__main__':
#     tokenizer = AutoTokenizer.from_pretrained('gpt2')
#     tokenizer.pad_token = tokenizer.eos_token
#     data = prep_data(tokenizer, 100, './dd.jsonl')
#     print(data)
