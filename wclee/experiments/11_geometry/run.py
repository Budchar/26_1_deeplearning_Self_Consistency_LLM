"""
Exp11: Hidden State Geometry + Logit Lens (Layer Vocabulary Entropy)

각 레이어 hidden state를 추출하여:
1. t-SNE용 key layer hidden states 저장
2. Logit Lens: 각 레이어 hidden state를 LM head로 투영 → vocab entropy 계산
   (logit lens 기법: 중간 레이어 표현이 "출력 어휘 공간"에서 어떻게 보이는지)

Usage:
  python experiments/11_geometry/run.py --model mistral --n_samples 150
  python experiments/11_geometry/run.py --model exaone  --n_samples 150
  python experiments/11_geometry/run.py --model qwen_7b --n_samples 150
"""

import sys, json, argparse, gc, torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.model_loader import load_model
from src.hidden_states import extract_hidden_states
from src.datasets.loader import load_triviaqa, check_correctness


def get_final_norm_and_lm_head(model):
    """Find the final LayerNorm and LM head projection of a transformer model."""
    lm_head = getattr(model, "lm_head", None)
    norm = None
    for attr_path in [
        "model.norm",               # Qwen, Mistral, Llama
        "transformer.ln_f",         # EXAONE, GPT-2 style
        "model.decoder.final_layer_norm",  # OPT
        "transformer.norm_f",       # Falcon
    ]:
        try:
            obj = model
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            norm = obj
            break
        except AttributeError:
            continue
    return lm_head, norm


def compute_logit_lens_entropy(hidden_states_np, model, device):
    """
    Logit Lens: 각 레이어 hidden state → (final_norm) → lm_head → softmax → entropy.
    중간 레이어가 출력 어휘 공간에서 얼마나 불확실한지 측정.
    """
    lm_head, final_norm = get_final_norm_and_lm_head(model)
    if lm_head is None:
        return None

    n_layers_plus_1 = hidden_states_np.shape[0]
    entropies = []
    dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for i in range(n_layers_plus_1):
            h = torch.tensor(hidden_states_np[i], dtype=dtype).unsqueeze(0).to(device)
            if final_norm is not None:
                try:
                    h = final_norm(h)
                except Exception:
                    pass
            try:
                logits = lm_head(h).squeeze(0).float()
                probs = F.softmax(logits, dim=-1)
                entropy = torch.special.entr(probs).sum().item()
                entropies.append(entropy)
            except Exception:
                entropies.append(0.0)

    return entropies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--n_samples", type=int, default=150)
    args = parser.parse_args()

    out_dir = ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"11_geometry_{args.model}_{ts}.json"

    print(f"=== Exp11: Geometry + Logit Lens | model={args.model} n={args.n_samples} ===")

    # Best probe layer from existing Exp06 results
    probe_files = sorted(
        (ROOT / "results" / "raw").glob(f"06_layer_probe_{args.model}_*.json"),
        key=lambda x: x.stat().st_mtime,
    )
    best_layer = None
    if probe_files:
        pd = json.load(open(probe_files[-1]))
        best_layer = pd["probe"]["best_layer"]
        n_layers_total = pd["n_layers"]
        print(f"  best_probe_layer={best_layer}/{n_layers_total} from Exp06")

    samples = load_triviaqa(n_samples=args.n_samples)
    wrapper = load_model(args.model)
    device = next(wrapper.model.parameters()).device

    records = []
    for i, item in enumerate(samples[:args.n_samples]):
        if i % 20 == 0:
            print(f"  [{i}/{args.n_samples}]", flush=True)
        try:
            result = extract_hidden_states(wrapper, item["question"])
            is_correct = check_correctness(result.generated_text, item["answers"])
            hs = result.hidden_states_last_token  # (n_layers+1, hidden_dim)
            n_L = hs.shape[0] - 1

            # Logit lens: vocab entropy at every layer
            logit_entropy = compute_logit_lens_entropy(hs, wrapper.model, device)

            # Store hidden states at key layers only (to keep file size reasonable)
            key_layers = sorted(set(filter(lambda x: 0 <= x < hs.shape[0], [
                0,
                n_L // 4,
                n_L // 2,
                3 * n_L // 4,
                best_layer if best_layer is not None else n_L - 1,
                n_L,
            ])))
            key_hidden = {str(l): hs[l].tolist() for l in key_layers}

            records.append({
                "question": item["question"],
                "generated": result.generated_text,
                "is_correct": is_correct,
                "logit_lens_entropy": logit_entropy,
                "key_hidden_states": key_hidden,
                "n_layers": n_L,
            })
        except Exception as e:
            print(f"  [WARN] {i}: {e}")
            continue

    wrapper.unload()
    gc.collect()
    torch.cuda.empty_cache()

    # Aggregate logit lens per correct/wrong
    correct_entropy = [r["logit_lens_entropy"] for r in records if r["is_correct"] and r["logit_lens_entropy"]]
    wrong_entropy   = [r["logit_lens_entropy"] for r in records if not r["is_correct"] and r["logit_lens_entropy"]]

    summary = {
        "model": args.model,
        "n_samples": len(records),
        "best_probe_layer": best_layer,
        "accuracy": sum(r["is_correct"] for r in records) / len(records) if records else 0,
        "logit_lens_mean_correct": np.mean(correct_entropy, axis=0).tolist() if correct_entropy else [],
        "logit_lens_mean_wrong":   np.mean(wrong_entropy,   axis=0).tolist() if wrong_entropy   else [],
        "logit_lens_std_correct":  np.std(correct_entropy,  axis=0).tolist() if correct_entropy else [],
        "logit_lens_std_wrong":    np.std(wrong_entropy,    axis=0).tolist() if wrong_entropy   else [],
        "records": records,
    }

    json.dump(summary, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    print(f"  n_correct={len(correct_entropy)}, n_wrong={len(wrong_entropy)}")
    if correct_entropy and wrong_entropy:
        mean_c = np.mean(correct_entropy, axis=0)
        mean_w = np.mean(wrong_entropy, axis=0)
        if best_layer is not None:
            print(f"  Logit lens entropy @ best_layer={best_layer}: "
                  f"correct={mean_c[best_layer]:.3f}, wrong={mean_w[best_layer]:.3f}, "
                  f"gap={mean_w[best_layer]-mean_c[best_layer]:+.3f}")


if __name__ == "__main__":
    main()
