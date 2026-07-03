import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class LoRAInference:
    def __init__(self, base_model: str, lora_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float32,
            device_map="cpu",
        )

        self.model = PeftModel.from_pretrained(
            base,
            lora_path,
        )

        self.model.eval()

    def generate(
        self,
        system: str,
        user: str,
        max_new_tokens: int = 150,
    ) -> str:

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Remove the prompt tokens
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return response.strip()