"""
TriviaQA / TruthfulQA / HaluEval / MMLU / NaturalQuestions 데이터셋 로더.
각 데이터셋을 {"question": str, "answers": List[str], "source": str} 형태로 통일.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "datasets"


def _normalize_answer(text: str) -> str:
    return text.strip().lower()


def check_correctness(prediction: str, gold_answers: List[str]) -> bool:
    """예측 답변이 정답 목록 중 하나를 포함하는지 확인."""
    pred = _normalize_answer(prediction)
    for ans in gold_answers:
        ans_norm = _normalize_answer(ans)
        if ans_norm in pred or pred in ans_norm:
            return True
    return False


# ─── TriviaQA ────────────────────────────────────────────────────────────────

def load_triviaqa(split: str = "validation", n_samples: int = None, seed: int = 42) -> List[Dict]:
    """
    HuggingFace datasets에서 TriviaQA 로드.
    각 샘플: {question, answers, source, dataset}
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("trivia_qa", "rc.nocontext", split=split,
                          cache_dir=str(DATA_DIR))
    except Exception as e:
        print(f"[loader] TriviaQA 로드 실패: {e}")
        print("[loader] scripts/download_datasets.py 를 먼저 실행하세요.")
        return []

    samples = []
    for item in ds:
        answers = item["answer"]["aliases"] if "aliases" in item["answer"] else [item["answer"]["value"]]
        samples.append({
            "question": item["question"],
            "answers": answers,
            "source": "triviaqa",
            "dataset": "triviaqa",
        })

    if n_samples and n_samples < len(samples):
        random.seed(seed)
        samples = random.sample(samples, n_samples)

    return samples


# ─── TruthfulQA ──────────────────────────────────────────────────────────────

def load_truthfulqa(split: str = "validation", n_samples: int = None, seed: int = 42) -> List[Dict]:
    """
    TruthfulQA (generation task) 로드.
    각 샘플: {question, answers (correct only), incorrect_answers, source}
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "generation", split=split,
                          cache_dir=str(DATA_DIR))
    except Exception as e:
        print(f"[loader] TruthfulQA 로드 실패: {e}")
        return []

    samples = []
    for item in ds:
        correct = item.get("correct_answers", [])
        incorrect = item.get("incorrect_answers", [])
        if not correct:
            continue
        samples.append({
            "question": item["question"],
            "answers": correct,
            "incorrect_answers": incorrect,
            "source": "truthfulqa",
            "dataset": "truthfulqa",
        })

    if n_samples and n_samples < len(samples):
        random.seed(seed)
        samples = random.sample(samples, n_samples)

    return samples


# ─── HaluEval ────────────────────────────────────────────────────────────────

def load_halueval(task: str = "qa", n_samples: int = None, seed: int = 42) -> List[Dict]:
    """
    HaluEval 로드. task: "qa" | "summarization" | "dialogue"
    각 샘플에 is_hallucination (bool) 레이블 포함.
    """
    try:
        from datasets import load_dataset
        name_map = {
            "qa": "qa_samples",
            "summarization": "summarization_samples",
            "dialogue": "dialogue_samples",
        }
        ds = load_dataset("pminervini/HaluEval", name_map[task],
                          cache_dir=str(DATA_DIR))
        split_data = ds["data"] if "data" in ds else ds[list(ds.keys())[0]]
    except Exception as e:
        print(f"[loader] HaluEval 로드 실패: {e}")
        return []

    samples = []
    for item in split_data:
        if task == "qa":
            samples.append({
                "question": item.get("question", ""),
                "answers": [item.get("right_answer", "")],
                "hallucinated_answer": item.get("hallucinated_answer", ""),
                "is_hallucination": False,
                "source": "halueval_qa",
                "dataset": "halueval",
            })
            # hallucinated version도 추가
            samples.append({
                "question": item.get("question", ""),
                "answers": [item.get("right_answer", "")],
                "hallucinated_answer": item.get("hallucinated_answer", ""),
                "is_hallucination": True,
                "source": "halueval_qa",
                "dataset": "halueval",
                "prefilled_answer": item.get("hallucinated_answer", ""),
            })
        else:
            samples.append({
                "question": item.get("user_query", item.get("document", "")),
                "answers": [item.get("right_response", item.get("right_summary", ""))],
                "is_hallucination": False,
                "source": f"halueval_{task}",
                "dataset": "halueval",
            })

    if n_samples and n_samples < len(samples):
        random.seed(seed)
        samples = random.sample(samples, n_samples)

    return samples


# ─── MMLU ────────────────────────────────────────────────────────────────────

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

_MMLU_CHOICE_LABELS = ["A", "B", "C", "D"]


def load_mmlu(
    subjects: List[str] = None,
    split: str = "test",
    n_samples: int = None,
    seed: int = 42,
) -> List[Dict]:
    """
    MMLU (Massive Multitask Language Understanding) 로드.
    4지선다 형식 → 정답 텍스트 + 보기 포함.
    subjects가 None이면 전체 57개 subject에서 균등 샘플링.
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        print("[loader] datasets 패키지가 없습니다.")
        return []

    target_subjects = subjects or MMLU_SUBJECTS
    all_samples = []

    for subj in target_subjects:
        try:
            ds = hf_load("cais/mmlu", subj, split=split, cache_dir=str(DATA_DIR))
            for item in ds:
                choices = item["choices"]
                answer_idx = item["answer"]  # 0-3
                answer_text = choices[answer_idx]
                question_with_choices = (
                    item["question"] + "\n"
                    + "\n".join(f"{_MMLU_CHOICE_LABELS[i]}. {c}" for i, c in enumerate(choices))
                )
                all_samples.append({
                    "question": question_with_choices,
                    "question_only": item["question"],
                    "choices": choices,
                    "answer_idx": answer_idx,
                    "answers": [answer_text] + [_MMLU_CHOICE_LABELS[answer_idx]],
                    "subject": subj,
                    "source": "mmlu",
                    "dataset": "mmlu",
                })
        except Exception as e:
            print(f"[loader] MMLU subject {subj} 로드 실패: {e}")
            continue

    if n_samples and n_samples < len(all_samples):
        random.seed(seed)
        all_samples = random.sample(all_samples, n_samples)

    return all_samples


# ─── NaturalQuestions ────────────────────────────────────────────────────────

def load_naturalquestions(
    split: str = "validation",
    n_samples: int = None,
    seed: int = 42,
) -> List[Dict]:
    """
    NaturalQuestions (open-domain) 로드.
    단답형 정답(short_answers)만 사용.
    """
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load(
            "google-research-datasets/natural_questions",
            "default",
            split=split,
            cache_dir=str(DATA_DIR),
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[loader] NaturalQuestions 로드 실패: {e}")
        return []

    samples = []
    for item in ds:
        annotations = item.get("annotations", {})
        short_answers_list = annotations.get("short_answers", [])
        answers = []
        for sa in short_answers_list:
            tokens = sa.get("text", [])
            if isinstance(tokens, list):
                answers.extend(tokens)
            elif isinstance(tokens, str) and tokens:
                answers.append(tokens)
        # yes/no answers
        yn = annotations.get("yes_no_answer", [])
        for v in yn:
            if v in (0, 1):
                answers.append("yes" if v == 1 else "no")

        if not answers:
            continue

        question = item["question"]["text"]
        if not question.endswith("?"):
            question += "?"

        samples.append({
            "question": question,
            "answers": list(set(answers)),
            "source": "naturalquestions",
            "dataset": "naturalquestions",
        })

    if n_samples and n_samples < len(samples):
        random.seed(seed)
        samples = random.sample(samples, n_samples)

    return samples


# ─── 통합 로더 ────────────────────────────────────────────────────────────────

def load_dataset_by_name(
    name: str,
    n_samples: int = None,
    seed: int = 42,
    **kwargs,
) -> List[Dict]:
    loaders = {
        "triviaqa": load_triviaqa,
        "truthfulqa": load_truthfulqa,
        "halueval": load_halueval,
        "mmlu": load_mmlu,
        "naturalquestions": load_naturalquestions,
        "nq": load_naturalquestions,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders.keys())}")
    return loaders[name](n_samples=n_samples, seed=seed, **kwargs)
