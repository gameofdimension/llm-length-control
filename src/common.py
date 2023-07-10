from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments


def tokenize(element, context_length, tokenizer):
    outputs = tokenizer(
        element["sentence"],
        truncation=True,
        max_length=context_length,
    )
    return {"input_ids": outputs["input_ids"]}


def prepare_data(context_length, tokenizer, train_num: int, valid_num: int):
    dataset = load_dataset("felixdae/openwebtext-wordlength")
    # dataset = dataset.filter(lambda x: len(nltk.word_tokenize(x['sentence'])) < context_length)
    dataset = DatasetDict(
        {
            "train": dataset['train'].shuffle().select(range(train_num)),
            # .shuffle().select(range(50000)),
            "valid": dataset['valid'].shuffle().select(range(valid_num)),
            # .shuffle().select(range(500))
            "test": dataset['test'].shuffle().select(range(valid_num))
        })
    tokenized_datasets = dataset.map(
        lambda x: tokenize(x, context_length, tokenizer), batched=True,
        remove_columns=dataset["train"].column_names
    )
    return tokenized_datasets


def get_train_args(output_dir: str, save_steps: int, warmup_steps: int):
    args = TrainingArguments(
        # output_dir="cs324-length-control",
        output_dir=output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        gradient_accumulation_steps=8,
        # eval_steps=5_000,
        # logging_steps=5_000,
        # save_steps=5_000,
        # warmup_steps=1_000,
        eval_steps=save_steps,
        logging_steps=save_steps,
        save_steps=save_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=1,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        push_to_hub=True,
        fp16=True,
    )
    return args


def make_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return model
