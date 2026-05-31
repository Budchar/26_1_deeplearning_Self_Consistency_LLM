"""
Exp12: Sublayer Probe — MLP vs Attention Stream Analysis

각 레이어 내부를 두 구간으로 분리해 probe:
  - attn_stream[i]: layer_input + attention_output  (MLP 이전 잔차 스트림)
  - mlp_stream[i]:  full layer output               (MLP 이후 잔차 스트림, 기존 Exp06과 동일)

해석:
  attn AUROC >> mlp AUROC at layer L  → attention이 hallucination 정보를 주도
  mlp  AUROC >> attn AUROC at layer L → MLP(key-value memory)가 주도
  둘 다 높음                           → 두 sublayer 모두 기여

Mistral Type I 가설 검증:
  L4에서 attn_stream AUROC가 낮고 mlp_stream AUROC가 높으면 →
  "MLP key-value memory"가 조기 인코딩의 원인

Usage:
  python experiments/12_sublayer_probe/run.py --model mistral --n_samples 150
  python experiments/12_sublayer_probe/run.py --model mistral_base --n_samples 150
  python experiments/12_sublayer_probe/run.py --model exaone --n_samples 150
"""

import sys, json, argparse, gc, re, torch
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.model_loader import load_model
from src.datasets.loader import load_triviaqa, check_correctness


LAYER_PATTERNS = [
    r"^model\.layers\.(\d+)$",
    r"^transformer\.h\.(\d+)$",
    r"^model\.decoder\.layers\.(\d+)$",
    r"^transformer\.blocks\.(\d+)$",
]

ATTN_ATTRS = ["self_attn", "attn", "attention", "self_attention"]


def find_transformer_layers(model):
    named = dict(model.named_modules())
    for pattern in LAYER_PATTERNS:
        hits = []
        for name, mod in named.items():
            m = re.match(pattern, name)
            if m:
                hits.append((int(m.group(1)), mod))
        if hits:
            hits.sort(key=lambda x: x[0])
            return [mod for _, mod in hits]
    return []


def find_attn_submodule(layer_mod):
    for attr in ATTN_ATTRS:
        sub = getattr(layer_mod, attr, None)
        if sub is not None:
            return sub
    return None


def extract_sublayer_streams(wrapper, prompt):
    """
    단일 forward pass에서 각 레이어의 attn_stream과 mlp_stream을 캡처.

    attn_stream[i] = layer_input[i] + attn_output[i]   (MLP 전)
    mlp_stream[i]  = full_layer_output[i]              (MLP 후)
    """
    model = wrapper.model
    device = next(model.parameters()).device
    inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(device)

    layers = find_transformer_layers(model)
    n = len(layers)

    layer_inputs  = {}  # idx → (hidden_dim,) np.float32
    attn_raw_outs = {}  # idx → attention sublayer output (no residual)
    layer_outputs = {}  # idx → full layer output (= mlp_stream)

    hooks = []

    for i, layer_mod in enumerate(layers):
        # 1. layer 입력 캡처 (pre-hook)
        def _pre(module, args, idx=i):
            h = args[0] if isinstance(args, (tuple, list)) else args
            if isinstance(h, torch.Tensor):
                layer_inputs[idx] = h[0, -1, :].detach().cpu().float().numpy()
        hooks.append(layer_mod.register_forward_pre_hook(_pre))

        # 2. attention sublayer 출력 캡처
        attn_mod = find_attn_submodule(layer_mod)
        if attn_mod is not None:
            def _attn(module, inp, out, idx=i):
                o = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(o, torch.Tensor):
                    attn_raw_outs[idx] = o[0, -1, :].detach().cpu().float().numpy()
            hooks.append(attn_mod.register_forward_hook(_attn))

        # 3. 전체 레이어 출력 캡처 (= mlp_stream)
        def _layer(module, inp, out, idx=i):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(o, torch.Tensor):
                layer_outputs[idx] = o[0, -1, :].detach().cpu().float().numpy()
        hooks.append(layer_mod.register_forward_hook(_layer))

    try:
        with torch.no_grad():
            model(**inputs, return_dict=True)
    finally:
        for h in hooks:
            h.remove()

    attn_streams = []
    mlp_streams  = []
    for i in range(n):
        # attn_stream = layer_input + attn_raw_out  (잔차 연결 수동 복원)
        if i in layer_inputs and i in attn_raw_outs:
            attn_streams.append(layer_inputs[i] + attn_raw_outs[i])
        elif i in layer_outputs:
            attn_streams.append(layer_outputs[i])   # fallback
        else:
            attn_streams.append(np.zeros(64, dtype=np.float32))

        mlp_streams.append(
            layer_outputs.get(i, np.zeros(64, dtype=np.float32))
        )

    return attn_streams, mlp_streams  # each: list of (hidden_dim,) arrays


def probe_all_layers(states_list, labels, seed=42):
    """
    states_list: List[List[np.ndarray]]  — (n_samples, n_layers, hidden_dim)
    Returns list of AUROC per layer.
    """
    labels = np.array(labels)
    n_layers = len(states_list[0])

    # 차원이 일치하는 레이어만 처리
    aurocs = []
    for li in range(n_layers):
        vecs = [s[li] for s in states_list]
        # 모두 같은 차원인지 확인
        dims = set(v.shape[0] for v in vecs)
        if len(dims) > 1 or list(dims)[0] < 2:
            aurocs.append(0.5)
            continue
        X = np.stack(vecs)
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, labels, test_size=0.2, random_state=seed,
                stratify=labels if len(set(labels)) > 1 else None
            )
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=seed)
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_te)[:, 1]
            auroc = roc_auc_score(y_te, proba) if len(set(y_te)) > 1 else 0.5
        except Exception:
            auroc = 0.5
        aurocs.append(float(auroc))

    return aurocs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--n_samples", type=int, default=150)
    args = parser.parse_args()

    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"12_sublayer_{args.model}_{ts}.json"

    print(f"=== Exp12: Sublayer Probe | model={args.model} n={args.n_samples} ===")

    # Exp06 best_layer 참조
    probe_files = sorted(
        (ROOT / "results" / "raw").glob(f"06_layer_probe_{args.model}_*.json"),
        key=lambda x: x.stat().st_mtime,
    )
    # 정확한 모델 키 매칭 (mistral이 mistral_base 파일을 잡지 않도록)
    exp06_best = None
    for pf in reversed(probe_files):
        pd = json.load(open(pf))
        if pd.get("model", "") == args.model:
            exp06_best = pd["probe"]["best_layer"]
            print(f"  Exp06 best_layer = {exp06_best}")
            break

    samples = load_triviaqa(n_samples=args.n_samples)
    wrapper = load_model(args.model)

    attn_all, mlp_all, labels = [], [], []

    for i, item in enumerate(samples[:args.n_samples]):
        if i % 20 == 0:
            print(f"  [{i}/{args.n_samples}]", flush=True)
        try:
            prompt = wrapper.format_prompt(item["question"])
            attn_s, mlp_s = extract_sublayer_streams(wrapper, prompt)

            # 생성은 별도 forward
            inputs = wrapper.tokenizer(prompt, return_tensors="pt").to(
                next(wrapper.model.parameters()).device
            )
            with torch.no_grad():
                gen_ids = wrapper.model.generate(
                    **inputs,
                    max_new_tokens=wrapper.max_new_tokens,
                    do_sample=False,
                    pad_token_id=wrapper.tokenizer.pad_token_id,
                )
            generated = wrapper.tokenizer.decode(
                gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            is_correct = check_correctness(generated, item["answers"])
            attn_all.append(attn_s)
            mlp_all.append(mlp_s)
            labels.append(int(is_correct))
        except Exception as e:
            print(f"  [WARN] {i}: {e}")
            continue

    wrapper.unload()
    gc.collect()
    torch.cuda.empty_cache()

    if not labels:
        print("No samples collected. Exiting.")
        return

    print(f"  Collected {len(labels)} samples. Training probes...", flush=True)
    attn_auroc = probe_all_layers(attn_all, labels)
    mlp_auroc  = probe_all_layers(mlp_all,  labels)

    n_layers = len(attn_auroc)
    ba_idx = int(np.argmax(attn_auroc))
    bm_idx = int(np.argmax(mlp_auroc))

    summary = {
        "model":           args.model,
        "n_samples":       len(labels),
        "n_layers":        n_layers,
        "accuracy":        sum(labels) / len(labels),
        "exp06_best_layer": exp06_best,
        "attn_stream_auroc": attn_auroc,
        "mlp_stream_auroc":  mlp_auroc,
        "best_attn_layer":   ba_idx,
        "best_attn_auroc":   attn_auroc[ba_idx],
        "best_mlp_layer":    bm_idx,
        "best_mlp_auroc":    mlp_auroc[bm_idx],
    }

    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")
    print(f"  Attn stream: best L{ba_idx}/{n_layers} ({ba_idx/n_layers*100:.0f}%) AUROC={attn_auroc[ba_idx]:.3f}")
    print(f"  MLP stream:  best L{bm_idx}/{n_layers} ({bm_idx/n_layers*100:.0f}%) AUROC={mlp_auroc[bm_idx]:.3f}")
    if exp06_best is not None and exp06_best < n_layers:
        print(f"  @ Exp06 best_layer L{exp06_best}:")
        print(f"    attn_auroc={attn_auroc[exp06_best]:.3f}  mlp_auroc={mlp_auroc[exp06_best]:.3f}")
        diff = mlp_auroc[exp06_best] - attn_auroc[exp06_best]
        driver = "MLP" if diff > 0.02 else ("Attention" if diff < -0.02 else "Both equal")
        print(f"    MLP - Attn = {diff:+.3f}  → driver: {driver}")


if __name__ == "__main__":
    main()
