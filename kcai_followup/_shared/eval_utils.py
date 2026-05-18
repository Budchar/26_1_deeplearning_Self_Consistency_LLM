"""정답 판정 유틸. Phase 1 metrics.py 호환 minimal 재구현.

정답 판정 (free-form QA): 모델 답이 gold answer set 중 하나를 substring으로 포함하면 정답.
normalize: lowercase, strip articles, punctuation, whitespace 정규화.
"""
from __future__ import annotations

import re
import string


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization."""
    s = s.lower().strip()
    s = _ARTICLE_RE.sub(" ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_correct(prediction: str, gold_answers: list[str]) -> bool:
    """예측이 정답 set 중 하나와 일치 (정규화 후 EM 또는 substring)."""
    if not prediction or not gold_answers:
        return False
    pred = normalize_answer(prediction)
    for ga in gold_answers:
        gold = normalize_answer(ga)
        if not gold:
            continue
        if pred == gold or gold in pred:
            return True
    return False


def extract_first_line(text: str) -> str:
    """모델 답에서 첫 줄·첫 문장 추출 (긴 generation 정리)."""
    text = text.strip()
    # 첫 줄
    line = text.split("\n", 1)[0].strip()
    # 첫 문장 (마침표·물음표·느낌표 기준)
    for stop in [". ", "? ", "! ", "."]:
        if stop in line:
            line = line.split(stop)[0].strip()
            break
    return line
