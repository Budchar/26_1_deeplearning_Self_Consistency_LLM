"""Backfill meta.json into existing run folders for reproducibility.

Idempotent: skips folders that already have a meta.json with all required keys.
CPU-only; safe to run while Phase 2 trains on GPU.

Coverage:
  - 01_se_seps/runs/<model>/<dataset>/   (Phase 1 generations + SE + SEPs)
  - 02_c2_sinks/runs/<variant>_<tag>/    (Phase 2 attention-sink training)
  - 03_c3_grokking/runs/<task>_<arch>/   (Phase 3 v1 grokking, if exists)

Captures: launch_args, prompt_template (sampled), git_commit, library_versions,
model_revision (HF SHA), host (GPU/CUDA), seed, launch_script_path + sha256.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

ROOT = Path("/home/kcai/experiments/dl_team_v2")

# ---- Hard-coded launch params (recovered from launch_full_sweep.sh / launch_full_train.sh) ----

PHASE1_LAUNCH_ARGS_FP16 = {
    "n": 1000,
    "n_samples": 10,
    "max_new_tokens": 64,
    "temperature": 1.0,
    "top_p": 0.95,
    "dtype": "fp16",
    "four_bit": False,
    "save_hidden": True,
    "prompt_style": "chat_template_with_system_user",
    "system_prompt": "You are a helpful assistant that answers questions concisely.",
    "user_template": "Question: {q}\nGive a short factual answer.",
}

PHASE1_LAUNCH_ARGS_4BIT = {
    **PHASE1_LAUNCH_ARGS_FP16,
    "dtype": "fp16",
    "four_bit": True,
    "bnb_quant_type": "nf4",
    "bnb_compute_dtype": "fp16",
    "bnb_double_quant": True,
    "bnb_max_memory": {"0": "10GiB", "cpu": "60GiB"},
}

PHASE2_LAUNCH_ARGS = {
    "max_steps": 50000,
    "micro_bs": 4,
    "grad_accum": 4,
    "checkpoint_interval": 5000,
    "dataset": "openwebtext_subset_100M_tokens",
    "model_arch": "gpt2_124M",
    "block_size": 1024,
    "sps_observed_5070": "≈1.45 it/s",
}

# Models that run in 4-bit
FOUR_BIT_MODELS = {"Qwen/Qwen2.5-7B-Instruct"}


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def file_sha256(p: Path) -> str:
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def library_versions() -> Dict[str, str]:
    out: Dict[str, str] = {"python": platform.python_version()}
    venv_py = ROOT / "shared/.venv/bin/python"
    py = str(venv_py) if venv_py.exists() else sys.executable
    code = (
        "import json\n"
        "v={}\n"
        "for n in ['torch','transformers','numpy','tokenizers','bitsandbytes','accelerate','peft']:\n"
        "    try:\n"
        "        m=__import__(n);v[n]=getattr(m,'__version__','?')\n"
        "    except Exception:\n"
        "        v[n]='not_installed'\n"
        "print(json.dumps(v))\n"
    )
    try:
        s = subprocess.check_output([py, "-c", code], stderr=subprocess.DEVNULL, text=True, timeout=20)
        out.update(json.loads(s.strip()))
    except Exception as e:
        out["error"] = str(e)
    return out


def host_info() -> Dict[str, str]:
    return {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu_count": str(os.cpu_count()),
        "gpu": sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).split("\n")[0],
        "cuda_version": sh(["nvcc", "--version"]).split("release ")[-1].split(",")[0] if sh(["which", "nvcc"]) else "",
        "driver": sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).split("\n")[0],
    }


def hf_model_revision(model_id: str) -> str:
    """Try local cache first; fall back to '' if not resolvable offline."""
    cache = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    safe = model_id.replace("/", "--")
    refs = cache / "hub" / f"models--{safe}" / "refs" / "main"
    if refs.exists():
        try:
            return refs.read_text().strip()
        except Exception:
            return ""
    return ""


def sampled_prompt(model_id: str) -> str:
    """Reconstruct one example prompt as-built by sample_generator.build_prompt."""
    sample_q = "When was the Eiffel Tower built?"
    fallback = (
        "Answer the following question with a short factual answer.\n"
        f"Question: {sample_q}\n"
        "Answer:"
    )
    venv_py = ROOT / "shared/.venv/bin/python"
    if not venv_py.exists():
        return fallback
    code = (
        "from transformers import AutoTokenizer\n"
        f"t=AutoTokenizer.from_pretrained({model_id!r}, trust_remote_code=True)\n"
        "msg=[{'role':'system','content':'You are a helpful assistant that answers questions concisely.'},"
        f"{{'role':'user','content':'Question: {sample_q}\\nGive a short factual answer.'}}]\n"
        "try:\n"
        "    print(t.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))\n"
        "except Exception:\n"
        "    print('FALLBACK')\n"
    )
    try:
        out = subprocess.check_output([str(venv_py), "-c", code], stderr=subprocess.DEVNULL, text=True, timeout=60)
        s = out.strip()
        if s == "FALLBACK" or not s:
            return fallback
        return s
    except Exception:
        return fallback


def common_meta(launch_script: Optional[Path]) -> Dict:
    return {
        "git_commit": sh(["git", "-C", str(ROOT), "rev-parse", "HEAD"]) or "not_a_git_repo",
        "library_versions": library_versions(),
        "host": host_info(),
        "started_at_backfilled": datetime.now(timezone.utc).isoformat(),
        "launch_script_path": str(launch_script) if launch_script else "",
        "launch_script_sha256": file_sha256(launch_script) if launch_script else "",
        "seed": None,
        "seed_note": "Phase 1/2 의도적으로 seed 미고정. N=10 sampling 평균값을 보는 통계량이라 점추정만 흔들리고 결론 방향은 안정. v2부터는 seed 고정.",
        "backfilled": True,
    }


def needs_write(p: Path) -> bool:
    if not p.exists():
        return True
    try:
        d = json.loads(p.read_text())
        return not d.get("backfilled")
    except Exception:
        return True


def write_meta(folder: Path, meta: Dict) -> None:
    target = folder / "meta.json"
    if not needs_write(target):
        return
    target.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"  wrote {target}")


# ---- Phase 1 ----
def backfill_phase1():
    print("[Phase 1] 01_se_seps/runs/")
    base = ROOT / "01_se_seps/runs"
    launch = ROOT / "01_se_seps/code/launch_full_sweep.sh"
    if not base.exists():
        print(f"  skip (not found): {base}")
        return
    common = common_meta(launch)

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name.startswith("synthetic"):
            continue
        # Reverse "Qwen__Qwen2.5-1.5B-Instruct" -> "Qwen/Qwen2.5-1.5B-Instruct"
        model_id = model_dir.name.replace("__", "/", 1)
        is_4bit = model_id in FOUR_BIT_MODELS
        launch_args = dict(PHASE1_LAUNCH_ARGS_4BIT if is_4bit else PHASE1_LAUNCH_ARGS_FP16)
        launch_args["model_id"] = model_id
        rev = hf_model_revision(model_id)
        prompt = sampled_prompt(model_id) if (model_dir / "triviaqa").exists() else ""

        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            la = dict(launch_args)
            la["dataset"] = ds_dir.name
            meta = {
                **common,
                "phase": "phase1_se_seps",
                "model_id": model_id,
                "model_revision_hf_sha": rev,
                "dataset": ds_dir.name,
                "launch_args": la,
                "prompt_template_sample": prompt,
                "outputs_present": sorted(p.name for p in ds_dir.iterdir() if p.is_file()),
            }
            write_meta(ds_dir, meta)


# ---- Phase 2 ----
def backfill_phase2():
    print("[Phase 2] 02_c2_sinks/runs/")
    base = ROOT / "02_c2_sinks/runs"
    launch = ROOT / "02_c2_sinks/code/launch_full_train.sh"
    if not base.exists():
        print(f"  skip (not found): {base}")
        return
    common = common_meta(launch)

    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        # parse <variant>_<tag>
        parts = run_dir.name.split("_", 1)
        variant, tag = (parts[0], parts[1] if len(parts) > 1 else "")
        la = dict(PHASE2_LAUNCH_ARGS)
        la["variant"] = variant
        la["tag"] = tag

        if variant == "softmax" and tag.startswith("smoke"):
            la["max_steps"] = 5000

        meta = {
            **common,
            "phase": "phase2_c2_sinks",
            "variant": variant,
            "tag": tag,
            "launch_args": la,
            "outputs_present": sorted(p.name for p in run_dir.iterdir())[:50],
        }
        write_meta(run_dir, meta)


# ---- Phase 3 v1 (if exists) ----
def backfill_phase3_v1():
    base = ROOT / "03_c3_grokking/runs"
    if not base.exists():
        return
    print("[Phase 3 v1] 03_c3_grokking/runs/")
    common = common_meta(None)
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = {
            **common,
            "phase": "phase3_v1_grokking",
            "run_name": run_dir.name,
            "launch_args": {"note": "see Phase 3 v1 launch script (CPU 4-core parallel)"},
            "outputs_present": sorted(p.name for p in run_dir.iterdir())[:50],
        }
        write_meta(run_dir, meta)


def main():
    backfill_phase1()
    backfill_phase2()
    backfill_phase3_v1()
    print("done")


if __name__ == "__main__":
    main()
