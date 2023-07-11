import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class Sampler:
    def __init__(self, model_id: str, is_peft: bool, device='cpu'):
        # load the model from the Hub
        if is_peft:
            # peft_model_id = "felixdae/rl-cs324-length-control"
            # peft_model_id = "felixdae/lora-cs324-length-control"
            config = PeftConfig.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
            model = PeftModel.from_pretrained(model, model_id)
            model = model.to(device)
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        else:
            # model_id = "felixdae/cs324-length-control"
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
