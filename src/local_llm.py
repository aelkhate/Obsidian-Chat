import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

class LocalLLM:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
        )

        # 4-bit quantization config (QLoRA-style runtime loading)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,   # ✅ correct way
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        self.model.eval()

    @torch.inference_mode()
    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 400) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]   # ✅ length of the prompt in tokens

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.05,
        )

        # ✅ Slice off the prompt tokens, keep only the model's new text
        gen_ids = out[0][input_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

