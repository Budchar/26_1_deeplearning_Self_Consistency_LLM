"""실험 3 Step 2: layer별 6 지표 계산.

지표:
  1. Mutual Information (hidden ↔ correct)
  2. Fisher discriminant ratio (representation separability)
  3. Silhouette score (separability 보조)
  4. Residual norm (||h_l - h_{l-1}||)
  5. Hidden norm (||h_l||)
  6. Layer-wise correct/wrong mean distance (||μ_c - μ_w||)

* logit entropy·attention entropy는 별도 forward 필요 → Step 3 별도 처리
* answer-token probability도 답 토큰 추적 필요 → Step 3

입력: _data/hidden_states/{model}/{dataset}/{hidden.npz, labels.jsonl, meta.json}
출력: 03_multi_metric_trajectory/results/{model}__{dataset}_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "_shared"))

from paths import HIDDEN_CACHE, PHASE1_MODELS, DATASETS, hidden_cell_dir, EXP3_TRAJECTORY
from hidden_state import load_hidden_cell
from resumable import atomic_write_json, is_done, mark_done


RESULTS = EXP3_TRAJECTORY / "results"


def load_labels(cell: Path) -> dict:
    """labels.jsonl → {id: bool}."""
    p = cell / "labels.jsonl"
    out = {}
    with open(p) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["id"]] = bool(rec["correct"])
    return out


def compute_layer_metrics(H: np.ndarray, y: np.ndarray, prev_H: np.ndarray | None = None, sample_n: int = 500) -> dict:
    """단일 layer 지표 계산.
    H: (N, hidden_dim) fp16 hidden state
    y: (N,) bool 정답 여부
    prev_H: (N, hidden_dim) 이전 layer hidden (residual norm용, 없으면 None)
    sample_n: PCA·silhouette 등 비싼 연산 sampling
    """
    H = H.astype(np.float32)
    n = len(H)
    out = {}

    # 1. Mutual Information — PCA 축소 후 (속도)
    n_pca = min(50, H.shape[1], n - 1)
    if n_pca >= 2:
        try:
            pca = PCA(n_components=n_pca, random_state=42)
            Hp = pca.fit_transform(H)
            mi = mutual_info_classif(Hp, y, random_state=42, n_neighbors=3)
            out["mutual_info_mean"] = float(mi.mean())
            out["mutual_info_max"] = float(mi.max())
        except Exception as e:
            out["mutual_info_mean"] = float("nan")
            out["mutual_info_max"] = float("nan")
            out["mi_error"] = str(e)[:100]
    else:
        out["mutual_info_mean"] = float("nan")

    # 2. Fisher discriminant ratio (between-class variance / within-class variance)
    if y.sum() >= 5 and (~y).sum() >= 5:
        mu_c = H[y].mean(axis=0)
        mu_w = H[~y].mean(axis=0)
        mu_all = H.mean(axis=0)
        between = ((mu_c - mu_all) ** 2 * y.mean() + (mu_w - mu_all) ** 2 * (1 - y.mean())).sum()
        within_c = ((H[y] - mu_c) ** 2).mean(axis=0).sum()
        within_w = ((H[~y] - mu_w) ** 2).mean(axis=0).sum()
        within = within_c * y.mean() + within_w * (1 - y.mean())
        out["fisher_ratio"] = float(between / (within + 1e-8))
        out["mean_class_distance"] = float(np.linalg.norm(mu_c - mu_w))
    else:
        out["fisher_ratio"] = float("nan")
        out["mean_class_distance"] = float("nan")

    # 3. Silhouette score (sample subset)
    if y.sum() >= 5 and (~y).sum() >= 5:
        try:
            n_s = min(sample_n, n)
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=n_s, replace=False)
            sil = silhouette_score(H[idx], y[idx].astype(int), metric="euclidean")
            out["silhouette"] = float(sil)
        except Exception as e:
            out["silhouette"] = float("nan")
            out["sil_error"] = str(e)[:100]
    else:
        out["silhouette"] = float("nan")

    # 4. Residual norm
    if prev_H is not None:
        delta = H - prev_H.astype(np.float32)
        out["residual_norm_mean"] = float(np.linalg.norm(delta, axis=1).mean())
        out["residual_norm_correct"] = float(np.linalg.norm(delta[y], axis=1).mean()) if y.sum() > 0 else float("nan")
        out["residual_norm_wrong"] = float(np.linalg.norm(delta[~y], axis=1).mean()) if (~y).sum() > 0 else float("nan")
    else:
        out["residual_norm_mean"] = float("nan")

    # 5. Hidden norm
    out["hidden_norm_mean"] = float(np.linalg.norm(H, axis=1).mean())
    out["hidden_norm_correct"] = float(np.linalg.norm(H[y], axis=1).mean()) if y.sum() > 0 else float("nan")
    out["hidden_norm_wrong"] = float(np.linalg.norm(H[~y], axis=1).mean()) if (~y).sum() > 0 else float("nan")

    return out


def run_one_cell(model: str, dataset: str, force: bool = False) -> dict:
    cell = hidden_cell_dir(model, dataset)
    if not is_done(cell):
        print(f"[skip] {model}/{dataset} — hidden state not extracted yet", flush=True)
        return {"status": "skipped_no_hidden"}

    out_path = RESULTS / f"{model.replace('/', '__')}__{dataset}_metrics.json"
    if out_path.exists() and not force:
        print(f"[skip] {model}/{dataset} — metrics already computed", flush=True)
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
    print(f"  computing 5 metrics x {n_layers} layers...", flush=True)

    t0 = time.time()
    layer_metrics = []
    prev_H = None
    for i, k in enumerate(layer_keys):
        H = by_layer[k]
        m = compute_layer_metrics(H, y, prev_H=prev_H)
        m["layer"] = i
        m["rel_depth"] = i / max(1, n_layers - 1)
        layer_metrics.append(m)
        prev_H = H
        if (i + 1) % 5 == 0:
            print(f"    layer {i + 1}/{n_layers} ({time.time() - t0:.0f}s)", flush=True)

    result = {
        "model": model,
        "dataset": dataset,
        "n_prompts": n_total,
        "n_correct": n_correct,
        "n_layers": n_layers,
        "layer_metrics": layer_metrics,
        "compute_time_sec": time.time() - t0,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, result)
    print(f"  saved → {out_path} ({time.time() - t0:.0f}s)", flush=True)
    return {"status": "ok", "path": str(out_path), "n_layers": n_layers, "time_sec": time.time() - t0}


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
