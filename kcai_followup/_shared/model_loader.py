"""Model + tokenizer loader. FP16/8-bit 옵션. eval 모드.

VRAM 안전:
- 7B FP16: ~14GB (5070 12GB로 빠듯) → device_map="auto" + 일부 CPU offload
- 7B 8-bit: ~7GB → 여유
- 1.5-3B FP16: ~3-6GB → 여유
"""
from __future__ import annotations

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name: str, dtype: str = "fp16", device_map: str = "auto", trust_remote_code: bool = True):
    """모델 + 토크나이저 로드. eval 모드. cache_dir는 HF default 사용."""
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs = dict(trust_remote_code=trust_remote_code, device_map=device_map)
    if dtype == "fp16":
        kwargs["dtype"] = torch.float16
    elif dtype == "bf16":
        kwargs["dtype"] = torch.bfloat16
    elif dtype == "8bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
        )
    elif dtype == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        raise ValueError(f"unknown dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, tokenizer


def unload(model) -> None:
    """모델 메모리 해제. 다음 모델 로딩 전 호출."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def model_n_layers(model) -> int:
    """모델 layer 수. 아키텍처별로 다름."""
    cfg = model.config
    for k in ("num_hidden_layers", "n_layer", "n_layers"):
        if hasattr(cfg, k):
            return getattr(cfg, k)
    raise ValueError(f"cannot detect n_layers for {cfg}")


def model_hidden_dim(model) -> int:
    cfg = model.config
    for k in ("hidden_size", "n_embd", "d_model"):
        if hasattr(cfg, k):
            return getattr(cfg, k)
    raise ValueError(f"cannot detect hidden_dim for {cfg}")
