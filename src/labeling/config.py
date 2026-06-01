#!/usr/bin/env python3
"""라벨링 도구 경로 설정 — 이 파일만 고쳐서 입력/출력 위치를 바꾼다.

label_server.py 가 기본값으로 여기 상수들을 읽는다. CLI 인자
(--input / --output / --session)를 주면 그쪽이 우선한다.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# 입력: by_category/{category}/thumbnail/{date}/*.jpg 를 포함하는 데이터셋 루트
INPUT_ROOT = REPO_ROOT / "nas/data/khonkaen"

# 출력: 라벨 상태(labels.json)와 재정렬 복사본(export/)을 둘 위치.
# 추후 다른 디스크/경로로 옮기려면 여기만 바꾸면 된다.
OUTPUT_ROOT = REPO_ROOT / "results"

# 한 라벨링 세션의 산출물이 모이는 폴더 이름.
# 실제 경로 = OUTPUT_ROOT / SESSION_NAME  (예: results/labeling/)
SESSION_NAME = "labeling"
