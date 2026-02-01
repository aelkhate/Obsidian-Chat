import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

class LocalLLM:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # 4-bit quantization to fit comfortably on 16GB VRAM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 400) -> str:
        # Qwen chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.05,
        )

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Remove the prompt part if it got included
        # (Simple approach: return tail after user_prompt; good enough for now)
        if user_prompt in decoded:
            return decoded.split(user_prompt, 1)[-1].strip()
        return decoded.strip()
