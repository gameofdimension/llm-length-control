from transformers import DataCollatorForLanguageModeling
from transformers import Trainer

from src.common import make_tokenizer, build_model, prepare_data, get_train_args


def train():
    context_length = 128

    tokenizer = make_tokenizer()
    model = build_model()

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    tokenized_datasets = prepare_data(context_length, tokenizer, 50000, 500)

    args = get_train_args(
        gradient_accumulation_steps=8,
        output_dir="cs324-length-control",
        save_steps=5000, warmup_steps=1000)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )
    trainer.train()
