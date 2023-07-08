import nltk
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


def tokenize(element, context_length, tokenizer):
    outputs = tokenizer(
        element["sentence"],
        truncation=True,
        max_length=context_length,
    )
    return {"input_ids": outputs["input_ids"]}


def make_data(context_length, tokenizer):
    dataset = load_dataset("felixdae/openwebtext-wordlength")
    # dataset = dataset.filter(lambda x: len(nltk.word_tokenize(x['sentence'])) < context_length)
    dataset = DatasetDict(
        {
            "train": dataset['train'].shuffle().select(range(4_000)),  # .shuffle().select(range(50000)),
            "valid": dataset['valid'].shuffle().select(range(450)),  # .shuffle().select(range(500))
            "test": dataset['test'].shuffle().select(range(450))
        })
    tokenized_datasets = dataset.map(
        lambda x: tokenize(x, context_length, tokenizer), batched=True,
        remove_columns=dataset["train"].column_names
    )
    return tokenized_datasets


def get_train_args():
    args = TrainingArguments(
        output_dir="cs324-length-control",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        push_to_hub=True,
        fp16=True,
    )
    return args


def make_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return tokenizer


def build_model():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    # ###
    # peft_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     inference_mode=False,
    #     r=8,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    # )
    #
    # model = get_peft_model(model, peft_config)
    # model.is_parallelizable = True
    # model.model_parallel = True
    # ###
    return model


def train():
    context_length = 128

    tokenizer = make_tokenizer()
    model = build_model()

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    tokenized_datasets = make_data(context_length, tokenizer)

    args = get_train_args()
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )
    trainer.train()
