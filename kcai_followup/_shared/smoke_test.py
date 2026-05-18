"""_shared util smoke test. Qwen2.5-1.5B + 3 prompt로 hidden state 추출.

목적:
- model_loader.load_model 동작 확인
- hidden_state.extract_for_cell 동작 확인
- resumable 재시작 동작 확인 (1차 부분 처리 → 2차 완료)

실행:
    python smoke_test.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from model_loader import load_model, unload, model_n_layers, model_hidden_dim
from hidden_state import extract_for_cell, load_hidden_cell
from resumable import is_done
from prompt_format import format_record
from paths import hidden_cell_dir


def main():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    test_cell = Path("/tmp/smoke_test_cell")
    test_cell.mkdir(parents=True, exist_ok=True)

    # 3 prompt 가짜 데이터
    prompts = [
        {"id": "q1", "question": "What is the capital of France?", "answers": ["Paris"], "dataset": "smoke"},
        {"id": "q2", "question": "Who wrote Hamlet?", "answers": ["William Shakespeare", "Shakespeare"], "dataset": "smoke"},
        {"id": "q3", "question": "What year did World War 2 end?", "answers": ["1945"], "dataset": "smoke"},
    ]

    print(f"\n[1/3] Loading {model_name}...", flush=True)
    model, tokenizer = load_model(model_name, dtype="fp16")
    print(f"  n_layers: {model_n_layers(model)}, hidden_dim: {model_hidden_dim(model)}")

    print(f"\n[2/3] Extracting hidden states (3 prompts)...", flush=True)
    result = extract_for_cell(
        model, tokenizer, prompts, test_cell,
        prompt_format_fn=lambda r: format_record(tokenizer, r),
        batch_save_every=2,
    )
    print(f"  result: {result}")
    print(f"  done.marker: {is_done(test_cell)}")

    print(f"\n[3/3] Loading cell back to verify...", flush=True)
    by_layer, meta, prompts_loaded = load_hidden_cell(test_cell)
    print(f"  layers found: {len(by_layer)} (first key: {list(by_layer.keys())[0]})")
    print(f"  hidden shape per layer: {list(by_layer.values())[0].shape}")
    print(f"  meta n_prompts: {meta['n_prompts']}, n_layers_p1: {meta['n_layers_p1']}")
    print(f"  prompts loaded: {len(prompts_loaded)}")

    print(f"\n[4/4] Resume test — call extract_for_cell again (should skip)...", flush=True)
    result2 = extract_for_cell(
        model, tokenizer, prompts, test_cell,
        prompt_format_fn=lambda r: format_record(tokenizer, r),
    )
    print(f"  result2: {result2}")
    assert result2["status"] == "skipped_already_done", "resume skip failed"

    unload(model)
    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
