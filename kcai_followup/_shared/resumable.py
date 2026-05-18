"""Resumable execution helpers. done.marker · JSONL append · checkpoint pattern.

핵심 원칙:
- 모든 작업 단위는 단일 cell (예: model × dataset). cell 디렉토리 내에 done.marker 두면 skip.
- 부분 결과는 JSONL append 모드로 저장. 매 N 항목마다 flush.
- 모든 함수는 idempotent — 같은 입력으로 반복 호출해도 같은 결과·중복 작업 없음.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


DONE_FILE = "done.marker"
PROGRESS_FILE = "progress.json"
LOG_FILE = "run.log"


def is_done(cell: Path) -> bool:
    return (cell / DONE_FILE).exists()


def mark_done(cell: Path, meta: dict | None = None) -> None:
    cell.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": time.time(), "timestamp_iso": time.strftime("%Y-%m-%d %H:%M:%S"), **(meta or {})}
    (cell / DONE_FILE).write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def read_progress(cell: Path) -> dict:
    p = cell / PROGRESS_FILE
    if not p.exists():
        return {"processed_ids": [], "n_done": 0}
    return json.loads(p.read_text())


def save_progress(cell: Path, progress: dict) -> None:
    cell.mkdir(parents=True, exist_ok=True)
    (cell / PROGRESS_FILE).write_text(json.dumps(progress, indent=2, ensure_ascii=False))


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def processed_ids(jsonl_path: Path, id_key: str = "id") -> set:
    """JSONL에서 이미 처리된 id set 읽기. 재시작 시 skip 판단용."""
    if not jsonl_path.exists():
        return set()
    ids = set()
    for rec in read_jsonl(jsonl_path):
        if id_key in rec:
            ids.add(rec[id_key])
    return ids


@contextmanager
def cell_log(cell: Path, stage: str):
    """Cell 진행 로그. 시작·종료·에러 기록."""
    cell.mkdir(parents=True, exist_ok=True)
    log_path = cell / LOG_FILE
    t0 = time.time()
    with open(log_path, "a") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {stage}\n")
    try:
        yield log_path
        with open(log_path, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DONE {stage} ({time.time() - t0:.1f}s)\n")
    except Exception as e:
        with open(log_path, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR {stage}: {type(e).__name__}: {e}\n")
        raise


def atomic_write_json(path: Path, data: Any) -> None:
    """원자적 JSON 쓰기 (중단 안전). tmp → rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, path)
