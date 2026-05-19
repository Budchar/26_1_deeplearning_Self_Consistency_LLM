# 실험 5 — 동일 base SFT vs SFT+DPO 통제 학습

> 목적: paper §Causal Analysis 인과 결론 확정
> 담당: 우창님 (A100 40GB)
> 부담: QLoRA 학습 1.5일 + 평가 2-3시간 (백그라운드 가능)

---

## 왜 이 실험이 필요한가

지금까지 우리가 가진 결과:
- OpenHermes-2.5-Mistral-7B (SFT만): peak L21
- Nous-Hermes-2-Mistral-7B-DPO (SFT + DPO): peak L2
- 19 layer 차이 → "DPO가 인코딩 위치 결정" 주장

문제: 두 모델은 base만 같고 SFT 데이터·학습 step·seed·DPO 데이터가 모두 다름. peak 이동이 진짜 DPO 때문인지 다른 변수 때문인지 단정 불가.

해결: 동일 조건에서 DPO 한 단계만 차이 나는 두 모델을 직접 학습해서 비교 → 차이의 원인을 DPO 한 가지로 좁힘.

---

## 실험 명세

### Base 모델 (공통 출발점)
**`mistralai/Mistral-7B-v0.1`** (OpenHermes·NousHermes의 공통 base이기도 함)

### 학습할 두 모델

| 모델 | 학습 내용 | 시작점 |
|---|---|---|
| **B (SFT only)** | OpenHermes-2.5 SFT 1 epoch | Mistral-7B-v0.1 base |
| **C (SFT + DPO)** | Argilla DPO-mix-7k DPO 1 epoch | **B 모델** 위에 |

> 핵심: C는 B 위에 DPO 한 단계만 추가한 것. 두 모델의 유일한 차이가 DPO.

### 학습 데이터

| 단계 | 데이터셋 | 샘플 수 |
|---|---|---|
| SFT (B 학습) | `teknium/OpenHermes-2.5` | 100,000 samples (전체 1M 중 일부) |
| DPO (C 학습) | `argilla/dpo-mix-7k` | 약 7,000 preference pairs |

### 학습 방법

- **QLoRA** (4-bit base + LoRA adapter)
- LoRA config: r=64, alpha=16, target=all linear modules
- 1 epoch, batch_size=1, gradient_accumulation_steps=16
- Adam 8-bit optimizer, learning_rate=2e-4 (SFT) / 5e-7 (DPO)
- gradient_checkpointing=True

### 자원

- **A100 40GB 1장** (실제 사용 약 12-15GB)
- 디스크 약 30GB (모델 + 데이터 cache)
- 시간:
  - SFT 100k samples 1 epoch: 약 15-20시간
  - DPO 7k samples 1 epoch: 약 12-15시간
  - **학습 합계 약 1.5일** (백그라운드)
- 평가 (hidden state + probe): 약 2-3시간

### 평가 (학습 후)

본 코드 `evaluate.py`가 자동 실행:
1. B와 C 각 모델에서 hidden state 추출 (TriviaQA·NQ-Open·SQuAD, 각 1000 prompt)
2. 각 layer에 logistic regression probe 학습 (5-fold CV)
3. peak layer 추출
4. B와 C의 peak layer 비교 plot

**기대 결과**: B는 peak ~L21, C는 peak ~L2-L4로 이동 → 가설 지지.

---

## 실행 절차

### 1) Clone + 환경 설치

```bash
git clone https://github.com/Budchar/26_1_deeplearning_Self_Consistency_LLM.git
cd 26_1_deeplearning_Self_Consistency_LLM/kcai_followup/exp5
pip install -r requirements.txt
```

### 2) B 모델 학습 (SFT, 약 15-20시간)

```bash
nohup python train_sft.py \
  --base mistralai/Mistral-7B-v0.1 \
  --data teknium/OpenHermes-2.5 \
  --n_samples 100000 \
  --output ./B_adapter \
  > train_sft.log 2>&1 &
```

### 3) C 모델 학습 (DPO, B 위에 추가, 약 12-15시간)

B 학습이 끝난 후 실행:

```bash
nohup python train_dpo.py \
  --base mistralai/Mistral-7B-v0.1 \
  --sft_adapter ./B_adapter \
  --data argilla/dpo-mix-7k \
  --output ./C_adapter \
  > train_dpo.log 2>&1 &
```

### 4) 평가 (약 2-3시간)

```bash
python evaluate.py \
  --base mistralai/Mistral-7B-v0.1 \
  --adapters B_adapter C_adapter \
  --datasets triviaqa nq_open squad \
  --n_prompts 1000 \
  --output ./results_exp5
```

### 5) 결과 GitHub push

```bash
git add kcai_followup/exp5/B_adapter/
git add kcai_followup/exp5/C_adapter/
git add kcai_followup/exp5/results_exp5/
git commit -m "exp5: B (SFT) + C (SFT+DPO) adapters + evaluation results"
git push
```

> adapter 파일은 약 100MB 각각. GitHub LFS 없이도 OK.

---

## 산출물 (push 후 확인용)

```
kcai_followup/exp5/
├── B_adapter/                    # SFT only LoRA adapter (~100MB)
├── C_adapter/                    # SFT + DPO LoRA adapter (~100MB)
├── train_sft.log
├── train_dpo.log
└── results_exp5/
    ├── B_hidden_states/          # B 모델 hidden state cache
    ├── C_hidden_states/          # C 모델 hidden state cache
    ├── B_probe_results.json      # B의 layer별 probe AUROC + peak
    ├── C_probe_results.json      # C의 layer별 probe AUROC + peak
    ├── comparison_plot.png       # B vs C peak layer 비교
    └── summary.json              # 최종 결과 요약
```

---

## 문제 발생 시

- OOM: `--batch_size 1 --gradient_accumulation_steps 32`로 변경
- 데이터 다운로드 느림: `huggingface-cli login`으로 인증 후 재시도
- 학습 도중 중단: 스크립트는 resumable 아니므로 처음부터 재실행

질문이나 문제 있으면 카톡으로 알려주세요. 코드 수정해서 다시 push하겠습니다.
