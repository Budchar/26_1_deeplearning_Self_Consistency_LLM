"""실험 3 Step 5: 정답·오답 그룹별 layer-wise 통계 계산 (extras).

입력:
  _data/logit_attn/{model}/{dataset}/extra_metrics.npz  — Step 4 출력
  _data/hidden_states/{model}/{dataset}/labels.jsonl    — Phase 1 정답 라벨

지표 (layer ℓ별):
  - logit_entropy   : correct mean, wrong mean, gap (wrong - correct)
  - attn_entropy    : 동일
  - answer_logit    : 동일 (final layer만, scalar이므로 trajectory 아님)
  - answer_prob     : 동일

추가:
  - peak_layer (correct·wrong 분기 최대 |gap|)
  - rel_depth (layer / (n_layers - 1))

출력:
  03_multi_metric_trajectory/results/{model}__{dataset}_extras.json

실행:
    python 05_compute_extras.py --models all --datasets all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "_shared"))

from paths import DATA, DATASETS, EXP3_TRAJECTORY, hidden_cell_dir
from resumable import atomic_write_json


LOGIT_ATTN_CACHE = DATA / "logit_attn"
RESULTS = EXP3_TRAJECTORY / "results"

TARGET_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]


def logit_attn_cell_dir(model: str, dataset: str) -> Path:
    return LOGIT_ATTN_CACHE / model.replace("/", "__") / dataset


def load_labels(cell: Path) -> dict:
    """labels.jsonl → {id: bool}."""
    p = cell / "labels.jsonl"
    out: dict[str, bool] = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["id"])] = bool(rec["correct"])
    return out


def _group_stats(values: np.ndarray, y: np.ndarray) -> dict:
    """1-D vector (N,)에 대해 correct·wrong group mean/std/gap."""
    out = {}
    if y.sum() > 0:
        v = values[y]
        v = v[~np.isnan(v)]
        out["correct_mean"] = float(v.mean()) if len(v) else float("nan")
        out["correct_std"] = float(v.std()) if len(v) else float("nan")
        out["correct_n"] = int(len(v))
    else:
        out["correct_mean"] = float("nan")
        out["correct_std"] = float("nan")
        out["correct_n"] = 0
    if (~y).sum() > 0:
        v = values[~y]
        v = v[~np.isnan(v)]
        out["wrong_mean"] = float(v.mean()) if len(v) else float("nan")
        out["wrong_std"] = float(v.std()) if len(v) else float("nan")
        out["wrong_n"] = int(len(v))
    else:
        out["wrong_mean"] = float("nan")
        out["wrong_std"] = float("nan")
        out["wrong_n"] = 0
    cm, wm = out["correct_mean"], out["wrong_mean"]
    if not (np.isnan(cm) or np.isnan(wm)):
        out["gap"] = float(wm - cm)  # 일반적으로 wrong이 더 큰 entropy
        out["abs_gap"] = float(abs(wm - cm))
    else:
        out["gap"] = float("nan")
        out["abs_gap"] = float("nan")
    return out


def _find_peak_layer(layer_stats: list[dict], key: str = "abs_gap") -> dict:
    """layer trajectory 중 abs_gap 최대 layer 찾기."""
    arr = np.array([ls.get(key, float("nan")) for ls in layer_stats], dtype=float)
    arr = np.where(np.isnan(arr), -np.inf, arr)
    if (arr == -np.inf).all():
        return {"peak_layer": -1, "peak_rel_depth": float("nan"), "peak_value": float("nan")}
    idx = int(arr.argmax())
    n = len(layer_stats)
    return {
        "peak_layer": idx,
        "peak_rel_depth": idx / max(1, n - 1),
        "peak_value": float(layer_stats[idx].get(key, float("nan"))),
    }


def run_one_cell(model: str, dataset: str, force: bool = False) -> dict:
    cell = logit_attn_cell_dir(model, dataset)
    npz_path = cell / "extra_metrics.npz"
    if not npz_path.exists():
        print(f"[skip] {model}/{dataset} — extra_metrics.npz not found", flush=True)
        return {"status": "skipped_no_npz"}

    out_path = RESULTS / f"{model.replace('/', '__')}__{dataset}_extras.json"
    if out_path.exists() and not force:
        print(f"[skip] {model}/{dataset} — extras json exists", flush=True)
        return {"status": "skipped_already", "path": str(out_path)}

    print(f"\n[load] {model}/{dataset}", flush=True)
    data = np.load(npz_path, allow_pickle=True)
    ids = [str(x) for x in data["ids"].tolist()]
    logit_entropy = data["logit_entropy"]  # (N, n_layers+1)
    attn_entropy = data["attn_entropy"]    # (N, n_layers)
    answer_logit = data["answer_logit"]    # (N,)
    answer_prob = data["answer_prob"]      # (N,)
    answer_tid = data["answer_token_id"]   # (N,)

    # 라벨 (hidden_states cell에서)
    hcell = hidden_cell_dir(model, dataset)
    labels_map = load_labels(hcell)
    y = np.array([labels_map.get(i, False) for i in ids], dtype=bool)
    n_correct = int(y.sum())
    n_total = len(y)
    print(f"  n={n_total}, correct={n_correct} ({n_correct / max(1, n_total) * 100:.1f}%)", flush=True)

    t0 = time.time()

    # logit_entropy per layer (n_layers + 1)
    n_layers_p1 = logit_entropy.shape[1]
    logit_layer_stats = []
    for li in range(n_layers_p1):
        s = _group_stats(logit_entropy[:, li], y)
        s["layer"] = li
        s["rel_depth"] = li / max(1, n_layers_p1 - 1)
        logit_layer_stats.append(s)

    # attn_entropy per layer (n_layers)
    n_layers = attn_entropy.shape[1]
    attn_layer_stats = []
    for li in range(n_layers):
        s = _group_stats(attn_entropy[:, li], y)
        s["layer"] = li
        s["rel_depth"] = li / max(1, n_layers - 1)
        attn_layer_stats.append(s)

    # answer_logit·answer_prob: scalar (final-layer only). pseudo-trajectory 만들지 않고 단일 통계로.
    valid_mask = answer_tid >= 0  # 정답 토큰을 못 찾은 prompt 제외
    valid_y = y & valid_mask
    invalid_n = int((~valid_mask).sum())

    def _scalar_group(vals: np.ndarray) -> dict:
        s = _group_stats(vals, valid_y)
        s["n_valid"] = int(valid_mask.sum())
        s["n_invalid"] = invalid_n
        return s

    # 단, 유효한 prompt만 사용하므로 vals[valid]·y[valid]로 다시 계산
    vmask = valid_mask
    if vmask.sum() > 0:
        logit_scalar = _group_stats(answer_logit[vmask], y[vmask])
        prob_scalar = _group_stats(answer_prob[vmask], y[vmask])
        logit_scalar["n_valid"] = int(vmask.sum())
        logit_scalar["n_invalid"] = invalid_n
        prob_scalar["n_valid"] = int(vmask.sum())
        prob_scalar["n_invalid"] = invalid_n
    else:
        logit_scalar = {"correct_mean": float("nan"), "wrong_mean": float("nan"), "gap": float("nan"),
                        "abs_gap": float("nan"), "n_valid": 0, "n_invalid": invalid_n}
        prob_scalar = dict(logit_scalar)

    # peak layer
    logit_peak = _find_peak_layer(logit_layer_stats, key="abs_gap")
    attn_peak = _find_peak_layer(attn_layer_stats, key="abs_gap")

    result = {
        "model": model,
        "dataset": dataset,
        "n_prompts": n_total,
        "n_correct": n_correct,
        "n_layers": int(n_layers),
        "n_layers_p1": int(n_layers_p1),
        "logit_entropy_layer_stats": logit_layer_stats,
        "attn_entropy_layer_stats": attn_layer_stats,
        "answer_logit_final": logit_scalar,
        "answer_prob_final": prob_scalar,
        "logit_entropy_peak": logit_peak,
        "attn_entropy_peak": attn_peak,
        "compute_time_sec": time.time() - t0,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, result)
    print(f"  saved → {out_path} ({time.time() - t0:.1f}s)", flush=True)
    print(f"  logit_entropy peak rel_depth = {logit_peak['peak_rel_depth']:.3f} (gap={logit_peak['peak_value']:.4f})", flush=True)
    print(f"  attn_entropy  peak rel_depth = {attn_peak['peak_rel_depth']:.3f} (gap={attn_peak['peak_value']:.4f})", flush=True)
    return {"status": "ok", "path": str(out_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--datasets", nargs="+", default=["all"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    models = TARGET_MODELS if args.models == ["all"] else args.models
    datasets = DATASETS if args.datasets == ["all"] else args.datasets

    summary = []
    for model in models:
        for dataset in datasets:
            try:
                r = run_one_cell(model, dataset, force=args.force)
                summary.append({"model": model, "dataset": dataset, **r})
            except Exception as e:
                import traceback
                traceback.print_exc()
                summary.append({"model": model, "dataset": dataset, "error": f"{type(e).__name__}: {e}"})

    print("\n=== SUMMARY ===")
    for s in summary:
        print(json.dumps(s, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
