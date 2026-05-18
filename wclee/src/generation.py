"""
Logit/score 캡처를 포함한 텍스트 생성 유틸리티.
모든 실험의 핵심 — 생성과 동시에 불확실성 신호를 수집한다.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from src.model_loader import LLMWrapper


@dataclass
class GenerationResult:
    prompt: str
    generated_text: str
    token_ids: List[int]
    token_texts: List[str]

    # Per-token 확률 정보
    token_log_probs: List[float]          # log P(token_t | context)
    token_entropies: List[float]          # H(vocab distribution) at each step
    token_top_probs: List[float]          # argmax prob at each step

    # 집계 스코어
    sequence_log_prob: float              # sum of log probs (unnormalized)
    mean_entropy: float
    max_entropy: float

    # 메타
    model_key: str = ""
    extra: dict = field(default_factory=dict)


def generate_with_scores(
    wrapper: LLMWrapper,
    question: str,
    system: str = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> GenerationResult:
    """
    단일 생성 + 모든 step의 logit score 캡처.
    output_scores=True 로 각 스텝의 vocab logit 전체를 가져온다.
    """
    prompt = wrapper.format_prompt(question, system)
    inputs = wrapper.tokenize(prompt)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = wrapper.model.generate(
            **inputs,
            max_new_tokens=wrapper.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=wrapper.tokenizer.pad_token_id,
        )

    generated_ids = output.sequences[0, input_len:]
    scores = output.scores  # tuple of (vocab_size,) tensors, one per generated token

    token_log_probs = []
    token_entropies = []
    token_top_probs = []

    for step_idx, (token_id, step_scores) in enumerate(zip(generated_ids, scores)):
        # step_scores: (batch_size, vocab_size) tensor normally.
        # Some models (e.g. Falcon with trust_remote_code) may return
        # non-tensor objects (DynamicCache etc.) — skip those tokens.
        if not isinstance(step_scores, torch.Tensor):
            continue
        logits = step_scores[0] if step_scores.dim() > 1 else step_scores
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        token_log_prob = log_probs[token_id].item()
        entropy = torch.special.entr(probs).sum().item()
        top_prob = probs.max().item()

        token_log_probs.append(token_log_prob)
        token_entropies.append(entropy)
        token_top_probs.append(top_prob)

    token_texts = [
        wrapper.tokenizer.decode([tid], skip_special_tokens=True)
        for tid in generated_ids.tolist()
    ]
    generated_text = wrapper.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return GenerationResult(
        prompt=prompt,
        generated_text=generated_text,
        token_ids=generated_ids.tolist(),
        token_texts=token_texts,
        token_log_probs=token_log_probs,
        token_entropies=token_entropies,
        token_top_probs=token_top_probs,
        sequence_log_prob=sum(token_log_probs),
        mean_entropy=float(np.mean(token_entropies)) if token_entropies else 0.0,
        max_entropy=float(np.max(token_entropies)) if token_entropies else 0.0,
        model_key=wrapper.model_key,
    )


def generate_multiple(
    wrapper: LLMWrapper,
    question: str,
    n: int = 10,
    system: str = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> List[GenerationResult]:
    """
    동일 질문에 대해 n번 샘플링.
    Semantic Entropy / Self-Consistency 실험에 사용.
    """
    results = []
    for i in range(n):
        r = generate_with_scores(
            wrapper, question, system=system,
            temperature=temperature, top_p=top_p, do_sample=True,
        )
        results.append(r)
    return results


def greedy_generate(wrapper: LLMWrapper, question: str, system: str = None) -> GenerationResult:
    """Greedy (deterministic) 생성 — calibration baseline으로 사용."""
    return generate_with_scores(
        wrapper, question, system=system,
        temperature=1.0, top_p=1.0, do_sample=False,
    )
