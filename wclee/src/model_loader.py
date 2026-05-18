"""
모델 통합 로딩 인터페이스.
여러 로컬 LLM을 동일한 API로 사용할 수 있도록 추상화.
"""

import sys
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "models.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class LLMWrapper:
    """단일 LLM에 대한 통합 인터페이스."""

    def __init__(self, model_key: str, config: dict = None):
        self.model_key = model_key
        cfg = config or load_config()
        self.model_cfg = cfg["models"][model_key]
        self.gen_cfg = cfg["generation"]
        self.model = None
        self.tokenizer = None

    def load(self):
        path = self.model_cfg["path"]
        dtype = getattr(torch, self.model_cfg["dtype"])
        trust = self.model_cfg["trust_remote_code"]

        print(f"[model_loader] Loading {self.model_cfg['name']} from {path}")

        tok_path = self.model_cfg.get("tokenizer_path", path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_path,
            trust_remote_code=trust,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            device_map=self.model_cfg["device_map"],
            trust_remote_code=trust,
        )
        self.model.eval()
        print(f"[model_loader] Loaded. Device: {next(self.model.parameters()).device}")
        return self

    def format_prompt(self, question: str, system: str = None) -> str:
        """채팅 템플릿에 맞게 프롬프트 포맷."""
        template = self.model_cfg["chat_template"]
        system = system or "You are a helpful assistant. Answer the question concisely."

        if template == "chatml":
            return (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        if template == "openchat":
            # OpenChat-3.5 / Starling-LM format (C-RLFT / RLHF on Mistral-7B)
            # Uses apply_chat_template if available; falls back to hardcoded format
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                try:
                    messages = [{"role": "user", "content": f"{system}\n\n{question}"}]
                    return self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    return (
                        f"GPT4 Correct System: {system}<|end_of_turn|>"
                        f"GPT4 Correct User: {question}<|end_of_turn|>"
                        f"GPT4 Correct Assistant:"
                    )

        if template in ("llama3", "qwen", "mistral", "smollm"):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback 1: system role not supported — merge into user turn
                try:
                    messages = [{"role": "user", "content": f"{system}\n\n{question}"}]
                    return self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    # Fallback 2: tokenizer has no chat template (e.g. base model used as
                    # tokenizer workaround) — use hardcoded Mistral [INST] format
                    return f"[INST] {system}\n\n{question} [/INST]"

        elif template == "exaone":
            return (
                f"[|system|]\n{system}\n\n"
                f"[|user|]\n{question}\n\n"
                f"[|assistant|]\n"
            )

        elif template == "falcon":
            return (
                f">>INTRODUCTION<<\n{system}\n"
                f">>QUESTION<<\n{question}\n"
                f">>ANSWER<<\n"
            )

        elif template == "opt":
            # OPT is a base model — no instruction format; use simple Q&A style
            return f"Q: {question}\nA:"

        else:
            return f"### Question:\n{question}\n\n### Answer:\n"

    def tokenize(self, text: str):
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    @property
    def max_new_tokens(self):
        return self.model_cfg["max_new_tokens"]

    def unload(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
        print(f"[model_loader] Unloaded {self.model_cfg['name']}")


def get_available_models():
    cfg = load_config()
    return list(cfg["models"].keys())


def load_model(model_key: str) -> LLMWrapper:
    wrapper = LLMWrapper(model_key)
    wrapper.load()
    return wrapper
