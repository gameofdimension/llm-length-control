import torch
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer

from src.common import make_tokenizer, build_model, prepare_data, get_train_args


def train():
    context_length = 128

    tokenizer = make_tokenizer()
    model = build_model()

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    tokenized_datasets = prepare_data(context_length, tokenizer, 2000000, 2000)

    args = get_train_args(
        gradient_accumulation_steps=8,
        output_dir="cs324-length-control",
        save_steps=1500, warmup_steps=1000)

    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )
    trainer.train()
    trainer.push_to_hub()


class Sampler:
    def __init__(self, device='cpu'):
        model_id = "felixdae/cs324-length-control"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, n: int):
        inputs = self.tokenizer(f"<len> {n} <text>", return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"].to(self.device),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=1000)
            output_ids = outputs.detach().cpu().numpy()
            return output_ids, self.tokenizer.batch_decode(output_ids)[0]
