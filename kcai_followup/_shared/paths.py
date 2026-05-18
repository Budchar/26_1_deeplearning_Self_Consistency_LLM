"""Central path config for followup experiments. Resumable·portable."""
from pathlib import Path
import os

ROOT = Path("/home/kcai/experiments/dl_team_followup")
SHARED = ROOT / "_shared"
DATA = ROOT / "_data"
DOCS = ROOT / "_docs"

# 실험별 디렉토리
EXP1_PATCHING = ROOT / "01_activation_patching"
EXP2_STEERING = ROOT / "02_steering_vector"
EXP3_TRAJECTORY = ROOT / "03_multi_metric_trajectory"
EXP4_PROBE = ROOT / "04_layer_probe_replication"

# 외부 의존
DL_V2 = Path("/home/kcai/experiments/dl_team_v2")
DL_V2_SHARED = DL_V2 / "shared"
DL_V2_DATA = DL_V2 / "_data" / "cache"  # 데이터셋 cache 재사용
TEAM_REPO = Path("/home/kcai/Nextcloud/2. 계속관리/[공부] AI대학원/딥러닝/팀프로젝트/_team_github_repo")
PHASE1 = TEAM_REPO / "Phase1_SE_SEPs_full_20260501"

# Phase 1 5 모델 (이미 다운된 것)
PHASE1_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

# Mistral 3 모델 (다운로드 중)
MISTRAL_MODELS = [
    "mistralai/Mistral-7B-v0.1",
    "teknium/OpenHermes-2.5-Mistral-7B",
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
]

DATASETS = ["triviaqa", "nq_open", "squad"]

# Hidden state cache 위치 (실험 3/4 공유)
HIDDEN_CACHE = DATA / "hidden_states"


def cell_dir(exp_dir: Path, model: str, dataset: str) -> Path:
    """모델·데이터셋별 cell 디렉토리. 슬래시는 더블 언더스코어로."""
    return exp_dir / "runs" / model.replace("/", "__") / dataset


def hidden_cell_dir(model: str, dataset: str) -> Path:
    return HIDDEN_CACHE / model.replace("/", "__") / dataset


# dl_team_v2 shared util import 가능하게
import sys
if str(DL_V2_SHARED) not in sys.path:
    sys.path.insert(0, str(DL_V2_SHARED))
# data_loader 직접 import 가능하게 (01_se_seps/code)
DL_V2_DATALOADER_DIR = DL_V2 / "01_se_seps" / "code"
if str(DL_V2_DATALOADER_DIR) not in sys.path:
    sys.path.insert(0, str(DL_V2_DATALOADER_DIR))
