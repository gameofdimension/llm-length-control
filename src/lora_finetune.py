from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer

from src.common import make_tokenizer, build_model, prepare_data, get_train_args


def build_lora_model():
    model = build_model()
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    model = get_peft_model(model, peft_config)
    model.is_parallelizable = True
    model.model_parallel = True
    return model


def train():
    context_length = 128

    tokenizer = make_tokenizer()
    model = build_lora_model()

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    tokenized_datasets = prepare_data(context_length, tokenizer, 100000, 1000)

    args = get_train_args(
        gradient_accumulation_steps=1,
        output_dir="lora-cs324-length-control",
        save_steps=500, warmup_steps=100)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )
    trainer.train()
    trainer.push_to_hub()
