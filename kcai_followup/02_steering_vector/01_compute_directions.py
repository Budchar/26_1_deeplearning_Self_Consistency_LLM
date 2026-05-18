"""실험 2 Step 1: hallucination direction vector 계산.

각 (model, dataset, layer)에 대해:
  d_l = mean(h_l | wrong) - mean(h_l | correct)
  ||d_l|| 정규화 옵션
  layer별 d_l을 .npz로 저장

입력: _data/hidden_states/{model}/{dataset}/
출력: 02_steering_vector/directions/{model}__{dataset}_directions.npz
       (key: layer_00 ~ layer_NN, shape=(hidden_dim,), fp16)
       _meta.json: n_wrong, n_correct, norm per layer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "_shared"))

from paths import PHASE1_MODELS, DATASETS, hidden_cell_dir, EXP2_STEERING
from hidden_state import load_hidden_cell
from resumable import is_done, atomic_write_json


DIRECTIONS = EXP2_STEERING / "directions"


def load_labels(cell: Path) -> dict:
    out = {}
    with open(cell / "labels.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["id"]] = bool(rec["correct"])
    return out


def compute_one(model: str, dataset: str, force: bool = False) -> dict:
    cell = hidden_cell_dir(model, dataset)
    if not is_done(cell):
        return {"status": "skipped_no_hidden"}

    out_npz = DIRECTIONS / f"{model.replace('/', '__')}__{dataset}_directions.npz"
    out_meta = DIRECTIONS / f"{model.replace('/', '__')}__{dataset}_directions_meta.json"
    if out_npz.exists() and not force:
        return {"status": "skipped_already", "path": str(out_npz)}

    by_layer, meta, _ = load_hidden_cell(cell)
    labels = load_labels(cell)
    ids = meta["ids"]
    y = np.array([labels[i] for i in ids], dtype=bool)
    n_c = int(y.sum())
    n_w = int((~y).sum())
    if n_c < 5 or n_w < 5:
        return {"status": "skipped_class_imbalance", "n_correct": n_c, "n_wrong": n_w}

    layer_keys = sorted([k for k in by_layer.keys() if k.startswith("layer_")])
    n_layers = len(layer_keys)

    directions = {}
    norms = {}
    cos_to_norm = {}  # direction과 평균 hidden norm 관계
    for k in layer_keys:
        H = by_layer[k].astype(np.float32)
        mu_w = H[~y].mean(axis=0)
        mu_c = H[y].mean(axis=0)
        d = mu_w - mu_c
        directions[k] = d.astype(np.float16)
        norms[k] = float(np.linalg.norm(d))

    DIRECTIONS.mkdir(parents=True, exist_ok=True)
    tmp = out_npz.parent / (out_npz.stem + ".tmp.npz")
    np.savez_compressed(tmp, **directions)
    import os
    os.replace(tmp, out_npz)

    meta_out = {
        "model": model,
        "dataset": dataset,
        "n_correct": n_c,
        "n_wrong": n_w,
        "n_layers": n_layers,
        "direction_norm_per_layer": norms,
        "direction_norm_mean": float(np.mean(list(norms.values()))),
        "direction_norm_max_layer": max(norms, key=norms.get),
        "direction_norm_max": max(norms.values()),
    }
    atomic_write_json(out_meta, meta_out)
    return {"status": "ok", "n_layers": n_layers, "norm_max": meta_out["direction_norm_max"], "norm_max_layer": meta_out["direction_norm_max_layer"]}


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
                r = compute_one(model, dataset, force=args.force)
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
