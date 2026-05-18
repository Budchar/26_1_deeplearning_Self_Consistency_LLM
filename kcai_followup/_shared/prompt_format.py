"""Prompt 포맷터. 데이터셋·모델 chat template 적용.

free-form QA: zero-shot 또는 few-shot. Phase 1과 동일 패턴 (간단·재현 가능).
"""
from __future__ import annotations


# Phase 1 (sample_generator.py) 와 완전 동일. hidden state 재현성 보장.
SYSTEM_QA = "You are a helpful assistant that answers questions concisely."


def format_qa_prompt(tokenizer, question: str, system: str = SYSTEM_QA, force_plain: bool = False) -> str:
    """chat template 적용. Phase 1과 일치. force_plain=True면 모든 모델에 같은 plain format."""
    if force_plain:
        return f"{system}\n\nQuestion: {question}\nGive a short factual answer."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\nGive a short factual answer."},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{system}\n\nQuestion: {question}\nGive a short factual answer."


def format_record(tokenizer, record: dict, force_plain: bool = False) -> str:
    return format_qa_prompt(tokenizer, record["question"], force_plain=force_plain)
