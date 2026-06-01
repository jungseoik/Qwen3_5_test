#!/usr/bin/env python3
"""description 분석 Web UI 경로/포트 설정 — 이 파일만 고쳐서 기본값을 바꾼다.

describe_server.py 가 기본값으로 여기 상수들을 읽는다. CLI 인자(--path / --port)를
주면 그쪽이 우선한다.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

# 분석 대상: labeled 평가 결과 폴더 (각 <Model>/describe/describe.jsonl 보유).
# 생략 시 describe_server 가 results/ 아래 최신 sweep_labeled_* 를 자동 선택한다.
DEFAULT_PATH = None

# Web UI 포트 (라벨링 도구 8800 과 충돌 회피).
PORT = 8801


def find_latest_sweep(base: Path = RESULTS_DIR):
    """results/ 아래 가장 최근 sweep_labeled_* 폴더를 반환 (없으면 None)."""
    if not base.is_dir():
        return None
    cands = sorted(p for p in base.glob("sweep_labeled_*") if p.is_dir())
    return cands[-1] if cands else None
