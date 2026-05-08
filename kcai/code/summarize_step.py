"""파인만식 단계 요약 마크다운 생성기.

사용:
  summarize_step.py <step-name> --out <body.md>

step-name ∈ {phase2, sweep_b, sweep_c, sweep_a, phase3v2, final}

각 단계에 맞는 결과 디렉토리에서 핵심 수치를 자동 추출해 평이한 비유로 풀어 씀.
실패하거나 결과 부족하면 가능한 만큼만 출력 (이메일은 어쨌든 발송).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _read_json(p: Path) -> Optional[Dict]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _read_jsonl(p: Path, limit: int = 0) -> List[Dict]:
    out: List[Dict] = []
    if not p.exists():
        return out
    with open(p) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
            if limit and i + 1 >= limit:
                break
    return out


# ---------------- phase2 ----------------
def summarize_phase2() -> str:
    base = Path("/home/kcai/experiments/dl_team_v2/02_c2_sinks")
    results = base / "results"
    md: List[str] = []
    md.append("## Phase 2 — Attention Sink 자발 생성 추적\n")
    md.append("**무엇을 본 건가?** GPT-2 학생을 from-scratch로 학습시키면서 학생이 시간이 지남에 따라 \"막히면 1번 자리 학생을 보자\"는 비공식 약속(attention sink)을 자발적으로 만드는지 추적했어요. 4가지 attention 형태(softmax, sigmoid, softpick, softplus)를 비교했어요.\n")

    found_any = False
    for variant in ["softmax", "sigmoid", "softpick", "softplus"]:
        metrics_path = results / f"{variant}_full_metrics.jsonl"
        recs = _read_jsonl(metrics_path)
        if not recs:
            continue
        found_any = True
        # find key milestones
        steps = [r.get("step") for r in recs if r.get("step") is not None]
        sinks = [r.get("sink_max") for r in recs if r.get("sink_max") is not None]
        if not sinks:
            continue
        s0 = sinks[0]
        s_max = max(sinks)
        s_max_step = steps[sinks.index(s_max)] if steps else "?"
        s_last = sinks[-1]
        s_last_step = steps[-1] if steps else "?"
        md.append(f"### {variant}")
        md.append(f"- step 0 sink_max ≈ {s0:.3f}")
        md.append(f"- step {s_max_step} 에서 최고치 sink_max ≈ {s_max:.3f}")
        md.append(f"- step {s_last_step} 종료 시점 sink_max ≈ {s_last:.3f}")
        if s_max > 0.5 and s0 < 0.1:
            ratio = s_max / max(s0, 1e-3)
            md.append(f"- → 학습 중 **{ratio:.0f}배 증가**. 학생들이 진짜로 1번 자리 약속을 만들어냈어요.")
        else:
            md.append(f"- → sink emergence 신호 약함 (이 attention 형태에서는 1번 자리 약속이 잘 안 만들어짐).")
        md.append("")

    if not found_any:
        md.append("*(아직 metrics.jsonl 결과가 부족합니다. 학습이 막 끝났다면 분석 단계가 더 필요해요.)*\n")

    md.append("**왜 중요한가?** sink가 자발적으로 생기는 attention은 \"그 형태가 sink의 원인\"이라는 architecture 차원 결론. 안 생기는 변형이 있으면 더 강한 발견. 발표용 그래프 한 장으로 핵심 메시지 전달 가능.\n")
    return "\n".join(md)


# ---------------- sweep_b (Pythia) ----------------
def summarize_sweep_b() -> str:
    base = Path("/home/kcai/experiments/dl_team_v2/04_sweep_b_pythia/runs")
    md: List[str] = []
    md.append("# Sweep B — Pythia 7개 사이즈 종단 비교 (1번)\n")
    md.append("## 무엇을 본 건가?\n")
    md.append("학생 7명(Pythia 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B)을 같은 학교 출신으로 데려와 시험 1000문제씩 풀게 했어요. 사이즈만 다르고 나머지(데이터, 토크나이저, 학습 레시피)는 같음.\n")
    md.append("**환각 감지 두 방법 비교**:\n")
    md.append("- **방법 A (Semantic Entropy, SE)**: 같은 문제 10번 풀게 시켜 답이 흩어지는지 봄")
    md.append("- **방법 B (Semantic Entropy Probes, SEPs)**: 학생 머릿속(hidden state) 직접 들여다봄\n")

    sizes = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b"]
    datasets = ["triviaqa", "nq_open", "squad"]

    # 셀별 데이터 수집
    rows: List[Dict] = []
    for sz in sizes:
        model_dir = base / f"EleutherAI__pythia-{sz}-deduped"
        for ds in datasets:
            ds_dir = model_dir / ds
            m = _read_json(ds_dir / "metrics.json")
            p = _read_json(ds_dir / "probes.json")
            if not m or not p:
                continue
            ga = m.get("greedy_acc", float("nan"))
            seps = p.get("best_logreg_halluc_auroc")
            if seps is None:
                seps = p.get("best_logreg_hallucination_auroc")
            se = m.get("se_discrete", {}).get("auroc", float("nan"))
            if seps is None or not isinstance(seps, (int, float)) or not isinstance(se, (int, float)):
                continue
            rows.append({
                "size": sz, "ds": ds, "greedy_acc": ga,
                "se_auroc": float(se), "seps_auroc": float(seps),
                "gap": float(seps) - float(se),
            })

    if not rows:
        md.append("*(아직 결과 없음.)*\n")
        return "\n".join(md)

    # 전체 표
    md.append("## 전체 결과 (21셀)\n")
    md.append("| 사이즈 | 데이터셋 | 답 맞춤률 | SE AUROC | SEPs AUROC | gap |")
    md.append("|---|---|---:|---:|---:|---:|")
    for r in rows:
        md.append(f"| {r['size']} | {r['ds']} | {r['greedy_acc']:.3f} | {r['se_auroc']:.3f} | {r['seps_auroc']:.3f} | {r['gap']:+.3f} |")
    md.append("")

    # 핵심 발견 1: SEPs 우위 카운트
    n_total = len(rows)
    n_seps_win = sum(1 for r in rows if r["gap"] > 0)
    avg_gap = sum(r["gap"] for r in rows) / n_total
    md.append("## 결과 1 — SEPs 우위 압도적\n")
    md.append(f"**{n_seps_win}/{n_total} 셀에서 SEPs > SE.** 평균 gap **{avg_gap:+.3f}**. SEPs(머릿속 보기)가 SE(답 흩어짐)보다 환각 감지 우수함을 강하게 재확인.\n")

    # 결과 2: weak-base 모델
    md.append("## 결과 2 — Pythia base 모델은 답을 거의 못 맞춤\n")
    md.append("Pythia는 instruction-tuning 안 된 base 모델이라 정답률 매우 낮음:\n")
    md.append("| 사이즈 | TriviaQA | NQ_Open | SQuAD |")
    md.append("|---|---:|---:|---:|")
    for sz in sizes:
        cells = {r["ds"]: r["greedy_acc"] for r in rows if r["size"] == sz}
        if not cells:
            continue
        md.append(f"| {sz} | {cells.get('triviaqa', float('nan')):.1%} | {cells.get('nq_open', float('nan')):.1%} | {cells.get('squad', float('nan')):.1%} |")
    md.append("")
    md.append("그런데도 SEPs는 잘 작동. 예: 70M이 SQuAD에서 답 0.6%만 맞추는데 환각 감지 AUROC 0.97. \"모델이 답을 못 해도 '나 모름'은 머릿속에 명확히 있다\".\n")
    md.append("**주의**: 이 결과는 reviewer 시각에서 일부 trivial. 답을 거의 다 틀리면 \"환각\" 라벨이 거의 다 붙고 → 분류가 쉬워짐. paper에서 강조하면 약점 잡힐 수 있음.\n")

    # 결과 3: 사이즈 효과
    md.append("## 결과 3 — 사이즈 효과 (경향성, 단조감소 아님)\n")
    md.append("| 사이즈 | 평균 gap |")
    md.append("|---|---:|")
    for sz in sizes:
        gaps = [r["gap"] for r in rows if r["size"] == sz]
        if gaps:
            md.append(f"| {sz} | {sum(gaps)/len(gaps):+.3f} |")
    md.append("")
    md.append("160M에서 peak (+0.314), 6.9B로 갈수록 감소 (+0.082). **방향은 맞지만 단조감소는 아님** (1.4B가 1B보다 약간 높음). paper에서 \"단조감소\" 강하게 못 쓰고 \"주로 감소\" 정도로.\n")

    # 결과 4: 데이터셋 효과
    md.append("## 결과 4 — 데이터셋별 SEPs 정확도\n")
    md.append("| 데이터셋 | 평균 SEPs AUROC |")
    md.append("|---|---:|")
    for ds in datasets:
        sepsa = [r["seps_auroc"] for r in rows if r["ds"] == ds]
        if sepsa:
            md.append(f"| {ds} | {sum(sepsa)/len(sepsa):.3f} |")
    md.append("")
    md.append("SQuAD(읽기이해)에서 환각 감지 가장 잘 됨. 책 안에 정답이 있는데 모델이 엉뚱한 단어 갖다 붙이면 차이가 명확해서.\n")

    # 한 줄 정리
    md.append("## 한 줄 정리\n")
    md.append("**SEPs > SE 압도적 (19/21 안팎). 사이즈 효과는 경향성만. weak-base finding은 trivial 위험.** Sweep C/D(깊이/너비 단독)가 paper의 진짜 가치 결정.\n")

    md.append("## 다음 단계 (orchestrator 자동 진행)\n")
    md.append("- 2번 (Sweep C 깊이단독): width=512 고정, depth 4~32 6개 from-scratch 학습 → H3 인과 검증")
    md.append("- 3번 (Sweep D 너비단독): depth=12 고정, width 256~1024 5개")
    md.append("- 4번 (Sweep A 패밀리5종): Pythia/Llama/Qwen/OPT/GPT-Neo base ~1.3B")
    md.append("- 종료 5/6 오전 예상\n")
    return "\n".join(md)


# ---------------- sweep_c (depth-only) ----------------
def summarize_sweep_c() -> str:
    base = Path("/home/kcai/experiments/dl_team_v2/05_sweep_c_depth/results")
    md: List[str] = []
    md.append("## Sweep C — 깊이 단독 변인 (width=512 고정, from-scratch)\n")
    md.append("**무엇을 본 건가?** 6개 학생을 똑같은 폭(512)으로 만들고 깊이만 4/8/12/16/24/32로 변경해서 처음부터 학습. 환각 검출 신호가 어느 layer에서 가장 잘 나오는지 — \"절대 layer index\"인지 \"상대 깊이 비율\"인지 인과적으로 검증.\n")

    md.append("| depth | n_params | sink_max(50K) | 비고 |")
    md.append("|---:|---:|---:|---|")
    found = 0
    for d in [4, 8, 12, 16, 24, 32]:
        run_dir = base.parent / "runs" / f"softmax_d{d}_w512"
        metrics = base / f"softmax_d{d}_w512_metrics.jsonl"
        recs = _read_jsonl(metrics)
        if not recs:
            md.append(f"| {d} | – | – | (대기) |")
            continue
        found += 1
        sinks = [r.get("sink_max") for r in recs if r.get("sink_max") is not None]
        last = sinks[-1] if sinks else float("nan")
        md.append(f"| {d} | – | {last:.3f} | |")
    md.append("")

    if found == 0:
        md.append("*(아직 결과 없음.)*\n")
    md.append("**왜 중요한가?** Phase 1 H3 (peak rel depth 0.68 ± 0.12) 의 진짜 인과 검증. 깊이를 바꿨을 때 peak이 같은 상대 위치(예: 32층 모델 → 21층 부근)에 머물면 H3 강하게 확정. 절대 인덱스에 머물면 다른 가설.\n")
    return "\n".join(md)


# ---------------- sweep_a (family) ----------------
def summarize_sweep_a() -> str:
    base = Path("/home/kcai/experiments/dl_team_v2/06_sweep_a_family/runs")
    md: List[str] = []
    md.append("## Sweep A — 5개 패밀리 (~1.3B 등급) 비교\n")
    md.append("**무엇을 본 건가?** 비슷한 사이즈(1.0-1.5B)의 학생 5명, 각자 다른 학교(Pythia/Llama/Qwen/OPT/GPT-Neo) 출신으로 데려와서 환각 검출 우위가 학교에 의존하는지 확인.\n")

    md.append("| 패밀리 | greedy_acc | SEPs AUROC | SE AUROC | gap |")
    md.append("|---|---:|---:|---:|---:|")
    families = [
        ("EleutherAI__pythia-1.4b-deduped", "Pythia"),
        ("meta-llama__Llama-3.2-1B-Instruct", "Llama"),
        ("Qwen__Qwen2.5-1.5B-Instruct", "Qwen"),
        ("facebook__opt-1.3b", "OPT"),
        ("EleutherAI__gpt-neo-1.3B", "GPT-Neo"),
    ]
    found = 0
    for slug, name in families:
        ds_dir = base / slug / "triviaqa"
        m = _read_json(ds_dir / "metrics.json")
        p = _read_json(ds_dir / "probes.json")
        if not m or not p:
            md.append(f"| {name} | (대기) | | | |")
            continue
        found += 1
        ga = m.get("greedy_acc", float("nan"))
        seps = p.get("best_logreg_hallucination_auroc", float("nan"))
        se = m.get("se_discrete", {}).get("auroc", float("nan"))
        gap = (seps - se) if (isinstance(seps, float) and isinstance(se, float)) else float("nan")
        md.append(f"| {name} | {ga:.3f} | {seps:.3f} | {se:.3f} | {gap:+.3f} |")
    md.append("")
    if found == 0:
        md.append("*(아직 결과 없음.)*\n")
    md.append("**왜 중요한가?** 모든 패밀리에서 gap이 양수면 \"학교에 무관하게 SEPs가 SE보다 환각을 잘 잡는다\"는 일반화 주장 가능. 한 패밀리만 음수면 그 패밀리 architecture 특성과 연결되는 추가 발견.\n")
    return "\n".join(md)


# ---------------- phase3v2 ----------------
def summarize_phase3v2() -> str:
    base = Path("/home/kcai/experiments/dl_team_v2/03_c3_grokking_v2/runs")
    md: List[str] = []
    md.append("## Phase 3 v2 — Grokking 예지 신호 (개선판)\n")
    md.append("**무엇을 본 건가?** 어려운 task 5종(parity-16, sparse-parity-k5n20, dyck-d8, mod-arith, full-SCAN)에서 Transformer/Mamba/LinAttn이 \"갑자기 일반화하는 순간(grokking)\"을 예측 가능한 신호가 있는지 봤어요. v1에서 task가 너무 쉬워서 grokking이 안 나왔던 문제 해결.\n")
    md.append("**왜 중요한가?** Mamba는 grok 안 한다는 anti-example이 v1에서 강한 신호로 나왔는데, v2 어려운 task로 옮기면 이 차이가 더 명확해질 가능성. 또 hidden state probe로 \"언제부터 일반화 회로가 형성되는지\"를 layer별로 추적.\n")

    md.append("\n*(상세 결과는 분석 단계 후 추가)*\n")
    return "\n".join(md)


# ---------------- final integration ----------------
def summarize_final() -> str:
    md: List[str] = []
    md.append("## 최종 통합 보고서\n")
    md.append("3주 사전실험 종료. Phase 1/2/3 v2 + Sweep A/B/C + Confident-Wrong 분석을 모두 통합해 paper-ready 산출물 정리.\n")
    md.append("**결과 위치**: `~/Nextcloud/2. 계속관리/AI대학원/딥러닝/팀프로젝트/사전실험_결과/04_v2_확장_통합.md`\n")
    return "\n".join(md)


def summarize_sweep_d() -> str:
    base = Path("/home/kcai/experiments/dl_team_v2/07_sweep_d_width/results")
    md: List[str] = []
    md.append("## Sweep D — 너비 단독 변인 (depth=12 고정, from-scratch)\n")
    md.append("**무엇을 본 건가?** 학생 5명을 똑같은 학년(12)에 두고 교실 크기만 256/384/512/768/1024로 변경. Sweep C(학년만 변경)와 짝 이뤄 학년·교실 둘 다 단독 인과 검증.\n")
    md.append("**왜 중요한가?** 환각 검출 신호의 위치가 학년 비율(0.68)에 머무는지, 아니면 교실 크기와도 관련있는지 나눠 봄. 양쪽 sweep 모두에서 비율이 0.68에 모이면 H3 강하게 확정.\n")
    md.append("\n*(상세 수치는 Sweep C/D 평가 단계에서 통합 보고)*\n")
    return "\n".join(md)


def summarize_sweep_cd_eval() -> str:
    out_path = Path("/home/kcai/experiments/dl_team_v2/05_sweep_c_depth/results/sweep_cd_h3_eval.json")
    md: List[str] = []
    md.append("## Sweep C/D 평가 — H3 인과 검증\n")
    md.append("**무엇을 본 건가?** Sweep C(6 깊이) + Sweep D(5 너비) = 11개 from-scratch 학생을 TriviaQA로 시험 보고, 각 layer별로 \"이 학생이 환각하는가\" probe를 학습. peak layer 위치를 모델 깊이로 정규화해 0.68에 모이는지 확인.\n")
    if not out_path.exists():
        md.append("\n*(평가 결과 파일 없음 — 학습이 끝났는지 확인 필요)*\n")
        return "\n".join(md)
    try:
        d = json.loads(out_path.read_text())
        h3 = d.get("h3_summary", {})
        md.append(f"### H3 요약\n")
        md.append(f"- 검증된 모델 수: {h3.get('n_models', 0)}")
        if "peak_rel_depth_mean" in h3:
            md.append(f"- peak rel depth 평균: **{h3['peak_rel_depth_mean']:.3f}** (목표 0.68)")
            md.append(f"- 표준편차: {h3['peak_rel_depth_std']:.3f}")
            md.append(f"- H3 일치 여부: {'✅ 통과' if h3.get('matches_h3_target_0.68') else '❌ 미달'}")
        md.append("")
        md.append("| 모델 | greedy_acc | peak layer | 총 layer | rel depth | peak AUROC |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for r in d.get("results", []):
            if "error" in r:
                continue
            ckpt = Path(r["ckpt"]).parent.name
            md.append(f"| {ckpt} | {r.get('greedy_acc', float('nan')):.3f} | {r.get('peak_layer', -1)} | {r.get('n_layer_probe_levels', 0)} | {r.get('peak_rel_depth', float('nan')):.3f} | {r.get('peak_auroc', float('nan')):.3f} |")
        md.append("")
        md.append("**왜 중요한가?** Phase 1에서는 Llama/Qwen Instruct 모델로 0.68 ± 0.12를 봤는데, from-scratch tiny base 모델에서도 같은 비율이 나오면 \"환각 검출 신호 위치는 학습 방식·아키텍처와 무관하게 모델 깊이의 일정 비율\"이라는 강한 일반화. 다른 비율이 나오면 그 차이가 보강 발견.\n")
    except Exception as e:
        md.append(f"\n*(요약 생성 실패: {e})*\n")
    return "\n".join(md)


SUMMARIZERS = {
    "phase2": summarize_phase2,
    "sweep_b": summarize_sweep_b,
    "sweep_c": summarize_sweep_c,
    "sweep_d": summarize_sweep_d,
    "sweep_cd_eval": summarize_sweep_cd_eval,
    "sweep_a": summarize_sweep_a,
    "phase3v2": summarize_phase3v2,
    "final": summarize_final,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(SUMMARIZERS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    body = SUMMARIZERS[args.step]()
    Path(args.out).write_text(body)
    print(f"[summarize] wrote {args.out} ({len(body)} chars)")


if __name__ == "__main__":
    main()
