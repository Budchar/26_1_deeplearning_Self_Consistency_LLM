"""Sweep C/D 평가 — from-scratch GPT 모델로 환각 검출 신호 측정.

Sweep C (depth ∈ {4,8,12,16,24,32}, width=512) + Sweep D (depth=12, width ∈ {256,
384,512,768,1024}) 의 trained checkpoint을 로드해, TriviaQA 1000q 추론 + hidden
state layer별 추출 + SEPs probe 학습 + peak layer 측정.

H3 검증: peak layer를 relative depth (= peak / total_layers) 로 정규화했을 때,
모든 모델에서 ~0.68 부근에 머무르는가?

출력: 05_sweep_c_depth/results/sweep_cd_h3_eval.json
       + plots/peak_rel_depth_vs_total_depth.png
       + plots/peak_rel_depth_vs_width.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Reuse Phase 2 GPT class
sys.path.insert(0, "/home/kcai/experiments/dl_team_v2/02_c2_sinks/code")
from model import GPT, GPTConfig  # type: ignore

sys.path.insert(0, "/home/kcai/experiments/dl_team_v2/01_se_seps/code")
from data_loader import load_dataset_by_name  # type: ignore


def _config_from_path(ckpt_path: Path) -> Dict:
    """체크포인트 경로 (예: .../softmax_d4_w512/step050000.pt) 에서 config 파싱.
    train.py가 config 안 저장해서 path에서 추론. n_head는 head_dim=64 가정.
    """
    import re
    parent = ckpt_path.parent.name  # softmax_d{D}_w{W}
    m = re.match(r"^([a-z_]+)_d(\d+)_w(\d+)$", parent)
    if not m:
        raise ValueError(f"cannot parse config from path: {parent}")
    variant, depth, width = m.group(1), int(m.group(2)), int(m.group(3))
    if width % 64 != 0:
        raise ValueError(f"width {width} not divisible by 64 (head_dim convention)")
    return {
        "block_size": 1024,
        "attn_variant": variant,
        "n_layer": depth,
        "n_embd": width,
        "n_head": width // 64,
        "grad_checkpoint": False,
    }


def load_checkpoint(ckpt_path: Path, device: str = "cuda") -> Tuple[GPT, Dict]:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = state.get("config") or state.get("cfg") or {}
    if not cfg_dict:
        # train.py가 config 안 저장. path에서 파싱.
        cfg_dict = _config_from_path(ckpt_path)
    cfg = GPTConfig(**cfg_dict)
    model = GPT(cfg)
    sd = state.get("model") or state.get("state_dict") or state
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, cfg_dict


def get_tokenizer():
    """GPT-2 BPE via tiktoken (Phase 2 dataset uses GPT-2 BPE)."""
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def build_prompt(q: str) -> str:
    """Same minimal prompt structure across all models for fair comparison."""
    return f"Question: {q}\nAnswer:"


@torch.no_grad()
def hidden_states_per_layer(model: GPT, idx: torch.Tensor) -> np.ndarray:
    """Return (n_layers+1, hidden_size) — last-token hidden state per layer (incl. embedding)."""
    B, T = idx.shape
    pos = torch.arange(T, device=idx.device)
    if T > model.pos_emb.num_embeddings:
        pos = pos.clamp(max=model.pos_emb.num_embeddings - 1)
    x = model.tok_emb(idx) + model.pos_emb(pos)
    x = model.drop(x)
    layers = [x[0, -1].detach().to(torch.float32).cpu().numpy()]
    for blk in model.blocks:
        x = blk(x)
        layers.append(x[0, -1].detach().to(torch.float32).cpu().numpy())
    x = model.ln_f(x)
    layers.append(x[0, -1].detach().to(torch.float32).cpu().numpy())
    return np.stack(layers, axis=0)  # (L+2, H)


@torch.no_grad()
def greedy_generate(model: GPT, idx: torch.Tensor, max_new: int = 32, eos_id: Optional[int] = None) -> torch.Tensor:
    """Pure greedy decoding using forward()."""
    out = idx
    for _ in range(max_new):
        if out.shape[1] >= model.pos_emb.num_embeddings:
            break
        logits, _ = model(out)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break
    return out


def is_match(pred: str, gold_list: List[str]) -> bool:
    p = pred.strip().lower()
    for g in gold_list:
        g = g.strip().lower()
        if not g:
            continue
        if g in p or p.startswith(g):
            return True
    return False


def eval_one_model(ckpt_path: Path, dataset_records: List[Dict], device: str, max_new: int = 32) -> Dict:
    print(f"[eval] {ckpt_path}", flush=True)
    model, cfg_dict = load_checkpoint(ckpt_path, device)
    tok = get_tokenizer()
    n_layers = cfg_dict.get("n_layer", len(model.blocks))

    all_hidden: List[np.ndarray] = []
    all_correct: List[int] = []
    correct_count = 0

    t0 = time.time()
    for i, rec in enumerate(dataset_records):
        prompt = build_prompt(rec["question"])
        ids = tok.encode(prompt)
        # truncate prompt to fit pos_emb
        max_prompt = max(8, model.pos_emb.num_embeddings - max_new - 1)
        ids = ids[-max_prompt:]
        idx = torch.tensor([ids], device=device, dtype=torch.long)

        h = hidden_states_per_layer(model, idx)  # (L+2, H)
        all_hidden.append(h)

        out = greedy_generate(model, idx, max_new=max_new)
        gen_ids = out[0, idx.shape[1]:].tolist()
        # decode and trim at first newline (Q&A boundary)
        text = tok.decode(gen_ids).split("\n")[0].strip()
        ok = 1 if is_match(text, rec.get("answers") or []) else 0
        all_correct.append(ok)
        correct_count += ok

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(dataset_records)}] acc={correct_count/(i+1):.3f} ({time.time()-t0:.1f}s)", flush=True)

    H = np.stack(all_hidden, axis=0)  # (N, L+2, D)
    y = np.array(all_correct)  # 1=correct, 0=wrong (=hallucination)
    halluc = 1 - y

    # Train per-layer probe
    n_total_layers = H.shape[1]
    auroc_per_layer = np.full(n_total_layers, np.nan)
    if halluc.sum() >= 10 and (1 - halluc).sum() >= 10:
        Xtr_idx, Xte_idx = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0, stratify=halluc)
        for L in range(n_total_layers):
            Xtr = H[Xtr_idx, L]
            ytr = halluc[Xtr_idx]
            Xte = H[Xte_idx, L]
            yte = halluc[Xte_idx]
            try:
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
                p = clf.predict_proba(Xte)[:, 1]
                auroc_per_layer[L] = roc_auc_score(yte, p)
            except Exception:
                continue

    peak_layer = int(np.nanargmax(auroc_per_layer)) if np.isfinite(auroc_per_layer).any() else -1
    peak_auroc = float(np.nanmax(auroc_per_layer)) if np.isfinite(auroc_per_layer).any() else float("nan")
    peak_rel = peak_layer / max(1, n_total_layers - 1)

    return {
        "ckpt": str(ckpt_path),
        "n_layer_blocks": n_layers,
        "n_layer_probe_levels": n_total_layers,
        "n_records": len(y),
        "greedy_acc": float(y.mean()),
        "hallucination_rate": float(halluc.mean()),
        "auroc_per_layer": auroc_per_layer.tolist(),
        "peak_layer": peak_layer,
        "peak_auroc": peak_auroc,
        "peak_rel_depth": peak_rel,
        "wall_sec": time.time() - t0,
    }


def find_checkpoints() -> List[Path]:
    out: List[Path] = []
    for root in [
        Path("/home/kcai/experiments/dl_team_v2/05_sweep_c_depth/checkpoints"),
        Path("/home/kcai/experiments/dl_team_v2/07_sweep_d_width/checkpoints"),
    ]:
        if not root.exists():
            continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir() and not run_dir.is_symlink():
                continue
            ckpts = sorted(run_dir.glob("step050000*.pt")) or sorted(run_dir.glob("step*_full.pt"))
            if ckpts:
                out.append(ckpts[-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--dataset", default="triviaqa")
    ap.add_argument("--out", default="/home/kcai/experiments/dl_team_v2/05_sweep_c_depth/results/sweep_cd_h3_eval.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpts = find_checkpoints()
    if not ckpts:
        print("[eval] no checkpoints found — Sweep C/D not yet trained. Exiting cleanly.")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"status": "no_checkpoints", "checked": True}, indent=2))
        return
    print(f"[eval] found {len(ckpts)} checkpoints")

    recs = load_dataset_by_name(args.dataset, n=args.n)

    results = []
    for ckpt in ckpts:
        try:
            r = eval_one_model(ckpt, recs, args.device)
        except Exception as e:
            r = {"ckpt": str(ckpt), "error": str(e)}
        results.append(r)
        # write incrementally
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2))

    # H3 check: are peak_rel_depth values clustered around 0.68?
    rels = [r["peak_rel_depth"] for r in results if "peak_rel_depth" in r and np.isfinite(r["peak_rel_depth"])]
    if rels:
        h3 = {
            "n_models": len(rels),
            "peak_rel_depth_mean": float(np.mean(rels)),
            "peak_rel_depth_std": float(np.std(rels)),
            "matches_h3_target_0.68": (abs(float(np.mean(rels)) - 0.68) < 0.10),
        }
    else:
        h3 = {"n_models": 0, "note": "no valid probes (likely accuracy too low or unbalanced)"}
    Path(args.out).write_text(json.dumps({"results": results, "h3_summary": h3}, ensure_ascii=False, indent=2))
    print(f"[eval] done: {args.out}")
    print(f"[eval] H3: {h3}")


if __name__ == "__main__":
    main()
