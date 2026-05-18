"""
Layer별 Hidden State 추출 및 Probe 분석.

각 레이어에서 마지막 토큰의 hidden state를 추출하여:
1. 어느 레이어가 hallucination 정보를 가장 잘 인코딩하는지
2. 레이어 깊이에 따른 uncertainty 신호 변화
를 분석한다.

두 가지 추출 방식:
- output_hidden_states=True  (표준 HF 모델)
- Forward Hook                (EXAONE 등 custom 모델)
"""

import re
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from src.model_loader import LLMWrapper


@dataclass
class HiddenStateResult:
    question: str
    generated_text: str
    is_correct: Optional[bool]

    # shape: (n_layers+1, hidden_dim)
    hidden_states_last_token: np.ndarray

    mean_entropy: float
    sequence_log_prob: float
    n_layers: int
    hidden_dim: int
    model_key: str


# ─── Layer 탐지 ───────────────────────────────────────────────────

LAYER_PATTERNS = [
    r"^model\.layers\.(\d+)$",         # Llama, Qwen, Mistral
    r"^transformer\.h\.(\d+)$",        # EXAONE, GPT-2 style
    r"^model\.decoder\.layers\.(\d+)$", # OPT
    r"^transformer\.blocks\.(\d+)$",   # Falcon
]


def _find_transformer_layers(model) -> List[tuple]:
    """모델에서 transformer block 모듈 목록 반환 (순서 보장)."""
    named_mods = dict(model.named_modules())
    for pattern in LAYER_PATTERNS:
        hits = []
        for name, mod in named_mods.items():
            m = re.match(pattern, name)
            if m:
                hits.append((int(m.group(1)), name, mod))
        if hits:
            hits.sort(key=lambda x: x[0])
            return [(name, mod) for _, name, mod in hits]
    return []


def _extract_with_output_hidden_states(wrapper, inputs):
    """표준 output_hidden_states=True 방식."""
    with torch.no_grad():
        out = wrapper.model(**inputs, output_hidden_states=True, return_dict=True)
    if out.hidden_states is None:
        return None
    return [h[0, -1, :].cpu().float().numpy() for h in out.hidden_states]


def _extract_with_hooks(wrapper, inputs):
    """Forward hook 방식 — custom 모델(EXAONE 등)용."""
    layers = _find_transformer_layers(wrapper.model)
    if not layers:
        raise RuntimeError("레이어 패턴을 찾을 수 없습니다. LAYER_PATTERNS 확인 필요.")

    captured = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output: (hidden_state, ...) 또는 tensor
            if isinstance(output, (tuple, list)):
                h = output[0]
            else:
                h = output
            captured[layer_idx] = h[0, -1, :].detach().cpu().float().numpy()
        return hook

    handles = []
    for idx, (name, mod) in enumerate(layers):
        handles.append(mod.register_forward_hook(make_hook(idx)))

    # 입력 임베딩 레이어 후크 (레이어 0 이전)
    embed_captured = {}
    embed_mod = None
    for attr in ["embed_tokens", "wte", "word_embeddings"]:
        for mod_name, mod in wrapper.model.named_modules():
            if mod_name.endswith(attr):
                embed_mod = mod
                break
        if embed_mod:
            break

    if embed_mod:
        def embed_hook(module, input, output):
            embed_captured["embed"] = output[0, -1, :].detach().cpu().float().numpy()
        handles.append(embed_mod.register_forward_hook(embed_hook))

    with torch.no_grad():
        wrapper.model(**inputs, return_dict=True)

    for h in handles:
        h.remove()

    # 임베딩 + 레이어 순서로 쌓기
    result = []
    if "embed" in embed_captured:
        result.append(embed_captured["embed"])
    for idx in range(len(layers)):
        if idx in captured:
            result.append(captured[idx])

    return result if result else None


def extract_hidden_states(
    wrapper: LLMWrapper,
    question: str,
    system: str = None,
) -> HiddenStateResult:
    """
    질문 forward pass → 각 레이어의 마지막 토큰 hidden state 추출.
    output_hidden_states=True 먼저 시도, 실패 시 hook 방식 fallback.
    """
    prompt = wrapper.format_prompt(question, system)
    inputs = wrapper.tokenize(prompt)

    # Step 1: hidden states
    hs_list = _extract_with_output_hidden_states(wrapper, inputs)
    if hs_list is None:
        hs_list = _extract_with_hooks(wrapper, inputs)
    if not hs_list:
        raise RuntimeError(f"Hidden state 추출 실패: {wrapper.model_key}")

    hs_last = np.stack(hs_list)  # (n_layers[+1], hidden_dim)

    # Step 2: generation (entropy + log_prob 수집)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen_output = wrapper.model.generate(
            **inputs,
            max_new_tokens=wrapper.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=wrapper.tokenizer.pad_token_id,
        )

    generated_ids = gen_output.sequences[0, input_len:]
    token_log_probs, token_entropies = [], []
    for token_id, step_scores in zip(generated_ids, gen_output.scores):
        logits = step_scores[0]
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs.append(log_probs[token_id].item())
        token_entropies.append(torch.special.entr(probs).sum().item())

    generated_text = wrapper.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return HiddenStateResult(
        question=question,
        generated_text=generated_text,
        is_correct=None,
        hidden_states_last_token=hs_last,
        mean_entropy=float(np.mean(token_entropies)) if token_entropies else 0.0,
        sequence_log_prob=float(sum(token_log_probs)),
        n_layers=hs_last.shape[0] - 1,
        hidden_dim=hs_last.shape[1],
        model_key=wrapper.model_key,
    )


# ─── Layer Probe ──────────────────────────────────────────────────

def train_layer_probes(
    hidden_states_list: List[np.ndarray],
    labels: List[int],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """각 레이어 hidden state → 정답 여부 로지스틱 회귀 probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.model_selection import train_test_split

    X = np.stack(hidden_states_list)
    y = np.array(labels)
    n_layers_plus_1 = X.shape[1]

    stratify = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=stratify
    )

    layer_auroc, layer_f1 = [], []
    for layer_idx in range(n_layers_plus_1):
        X_l_train = X_train[:, layer_idx, :]
        X_l_test  = X_test[:, layer_idx, :]
        scaler = StandardScaler()
        X_l_train = scaler.fit_transform(X_l_train)
        X_l_test  = scaler.transform(X_l_test)
        try:
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=seed)
            clf.fit(X_l_train, y_train)
            probs = clf.predict_proba(X_l_test)[:, 1]
            preds = clf.predict(X_l_test)
            auroc = roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else 0.5
            f1 = f1_score(y_test, preds, zero_division=0)
        except Exception:
            auroc, f1 = 0.5, 0.0
        layer_auroc.append(auroc)
        layer_f1.append(f1)

    best_layer = int(np.argmax(layer_auroc))
    return {
        "layer_auroc": layer_auroc,
        "layer_f1": layer_f1,
        "best_layer": best_layer,
        "best_auroc": layer_auroc[best_layer],
        "n_layers": n_layers_plus_1 - 1,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def compute_layer_entropy_profile(
    hidden_states_list: List[np.ndarray],
    labels: List[int],
) -> dict:
    """레이어별 L2 norm, cosine similarity 분석."""
    X = np.stack(hidden_states_list)
    y = np.array(labels)
    correct_mask = y == 1
    wrong_mask   = y == 0
    n_layers_plus_1 = X.shape[1]

    metrics = {
        "layer_norm_correct": [], "layer_norm_wrong": [], "layer_norm_diff": [],
        "layer_cosine_within_correct": [], "layer_cosine_within_wrong": [],
        "layer_cosine_between": [],
    }

    def mean_cosine(A):
        if len(A) < 2:
            return 1.0
        A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        sim = A_n @ A_n.T
        idx = np.triu_indices(len(A), k=1)
        return float(sim[idx].mean())

    for layer_idx in range(n_layers_plus_1):
        h_c = X[correct_mask, layer_idx, :] if correct_mask.any() else X[:1, layer_idx, :]
        h_w = X[wrong_mask, layer_idx, :]   if wrong_mask.any()   else X[:1, layer_idx, :]
        metrics["layer_norm_correct"].append(float(np.linalg.norm(h_c, axis=1).mean()))
        metrics["layer_norm_wrong"].append(float(np.linalg.norm(h_w, axis=1).mean()))
        metrics["layer_norm_diff"].append(float(np.linalg.norm(h_w, axis=1).mean()
                                                - np.linalg.norm(h_c, axis=1).mean()))
        metrics["layer_cosine_within_correct"].append(mean_cosine(h_c))
        metrics["layer_cosine_within_wrong"].append(mean_cosine(h_w))
        c_mean = h_c.mean(0); w_mean = h_w.mean(0)
        cos_bw = float(np.dot(c_mean, w_mean) /
                       (np.linalg.norm(c_mean) * np.linalg.norm(w_mean) + 1e-9))
        metrics["layer_cosine_between"].append(cos_bw)

    return metrics
