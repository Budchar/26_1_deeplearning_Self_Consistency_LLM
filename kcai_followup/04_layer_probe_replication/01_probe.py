"""실험 4 Step 1: layer별 hallucination probe 재학습. wclee 결과 재현·검증.

각 (모델, 데이터셋, layer)에 대해:
  - Hidden state → Logistic Regression → correct/wrong 분류
  - 5-fold CV AUROC
  - peak layer 추출

Phase 1 5 모델 + Mistral 3 모델 (다운 완료 후) 적용 가능.
hidden state cache 재사용 (실험 3과 공유).

입력: _data/hidden_states/{model}/{dataset}/{hidden.npz, labels.jsonl}
출력: 04_layer_probe_replication/results/{model}__{dataset}_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "_shared"))

from paths import PHASE1_MODELS, DATASETS, hidden_cell_dir, EXP4_PROBE
from hidden_state import load_hidden_cell
from resumable import is_done, atomic_write_json


RESULTS = EXP4_PROBE / "results"


def load_labels(cell: Path) -> dict:
    out = {}
    with open(cell / "labels.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["id"]] = bool(rec["correct"])
    return out


def probe_layer(H: np.ndarray, y: np.ndarray, n_folds: int = 5, seed: int = 42) -> dict:
    """단일 layer probe. 5-fold CV AUROC."""
    H = H.astype(np.float32)
    # 정답·오답 둘 다 최소 n_folds 이상 있어야 stratified CV 가능
    if y.sum() < n_folds or (~y).sum() < n_folds:
        return {"auroc_mean": float("nan"), "auroc_std": float("nan"), "n_folds": 0, "error": "insufficient class balance"}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aurocs = []
    for tr, te in skf.split(H, y):
        scaler = StandardScaler()
        Htr = scaler.fit_transform(H[tr])
        Hte = scaler.transform(H[te])
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed, n_jobs=1)
        clf.fit(Htr, y[tr])
        proba = clf.predict_proba(Hte)[:, 1]
        try:
            auroc = roc_auc_score(y[te], proba)
        except ValueError:
            auroc = float("nan")
        aurocs.append(auroc)
    arr = np.array(aurocs)
    return {
        "auroc_mean": float(np.nanmean(arr)),
        "auroc_std": float(np.nanstd(arr)),
        "n_folds": n_folds,
        "auroc_per_fold": [float(a) for a in arr],
    }


def run_one_cell(model: str, dataset: str, force: bool = False) -> dict:
    cell = hidden_cell_dir(model, dataset)
    if not is_done(cell):
        return {"status": "skipped_no_hidden", "model": model, "dataset": dataset}

    out_path = RESULTS / f"{model.replace('/', '__')}__{dataset}_probe.json"
    if out_path.exists() and not force:
        return {"status": "skipped_already", "path": str(out_path)}

    print(f"\n[load] {model}/{dataset}", flush=True)
    by_layer, meta, _ = load_hidden_cell(cell)
    labels = load_labels(cell)
    ids = meta["ids"]
    y = np.array([labels[i] for i in ids], dtype=bool)
    n_correct = int(y.sum())
    n_total = len(y)
    print(f"  n={n_total}, correct={n_correct} ({n_correct / n_total * 100:.1f}%)", flush=True)

    layer_keys = sorted([k for k in by_layer.keys() if k.startswith("layer_")])
    n_layers = len(layer_keys)
    print(f"  probing {n_layers} layers...", flush=True)

    t0 = time.time()
    layer_results = []
    for i, k in enumerate(layer_keys):
        H = by_layer[k]
        r = probe_layer(H, y)
        r["layer"] = i
        r["rel_depth"] = i / max(1, n_layers - 1)
        layer_results.append(r)

    aurocs = np.array([r["auroc_mean"] for r in layer_results])
    aurocs_safe = np.where(np.isnan(aurocs), -np.inf, aurocs)
    peak_layer = int(aurocs_safe.argmax())
    peak_auroc = float(aurocs[peak_layer])
    peak_rel_depth = peak_layer / max(1, n_layers - 1)

    result = {
        "model": model,
        "dataset": dataset,
        "n_prompts": n_total,
        "n_correct": n_correct,
        "n_layers": n_layers,
        "peak_layer": peak_layer,
        "peak_rel_depth": peak_rel_depth,
        "peak_auroc": peak_auroc,
        "layer_results": layer_results,
        "compute_time_sec": time.time() - t0,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, result)
    print(f"  peak: layer {peak_layer} (rel_d={peak_rel_depth:.2f}), AUROC={peak_auroc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return {"status": "ok", "path": str(out_path), "peak_layer": peak_layer, "peak_auroc": peak_auroc, "peak_rel_depth": peak_rel_depth}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--datasets", nargs="+", default=["all"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    models = PHASE1_MODELS if args.models == ["all"] else args.models
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
