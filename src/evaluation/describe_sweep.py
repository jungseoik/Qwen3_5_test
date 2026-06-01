#!/usr/bin/env python3
"""다중 모델 description sweep (진단용).

기존 labeled sweep 이 만든 폴더(`results/sweep_labeled_*`)를 입력으로 받아, 그 안의 각 모델
폴더(`<Model>/manifest.csv`)에 대해 모델을 차례로 재기동하며 describe_eval.py 를 실행한다.
결과는 각 모델 폴더 안 describe/ 서브폴더에 제자리(co-locate) 저장된다 — 별도 결과 폴더를 만들지 않는다.

오케스트레이션(모델 교체·헬스체크·모델 단축 매칭)은 sweep.py 의 헬퍼를 그대로 재사용한다
(labeled_eval ↔ fp_reduction 과 동일한 의존 방향).

흐름 (모델마다):
  1) docker compose down                        (이전 모델 종료)
  2) VLLM_MODEL override 하여 docker compose up -d
  3) /health 폴링                               (모델 로딩 대기)
  4) subprocess 로 describe_eval.py 실행         (그 모델의 오분류 case description)
  5) <Model>/describe/ 채움
종료 정책: 정상 종료 시 마지막 모델 띄워둠 / Ctrl+C 시 docker compose down.

전제
    - <from> 폴더가 labeled 평가 결과여야 한다 (각 <Model>/manifest.csv 보유).
    - describe 대상 모델의 weight 가 HF 캐시에 있어야 한다 (없으면 자동 스킵).
의존성: 표준 라이브러리만 사용.

사용 예
    python src/evaluation/describe_sweep.py --from results/sweep_labeled_20260529_095220
    python src/evaluation/describe_sweep.py --from <폴더> --models 4B,9B --kind fn
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from models import MODELS
from fp_reduction import load_env
from sweep import (
    compose, wait_healthy, parse_models, model_dirname, cache_has_weights,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DESCRIBE_EVAL = REPO_ROOT / "src" / "evaluation" / "describe_eval.py"
ENV_PATH = REPO_ROOT / ".env"
DESCRIBE_ENV_PATH = REPO_ROOT / ".env.describe"


def build_overlay(args) -> dict:
    """.env.describe(있으면) + CLI 오버라이드 → compose 에 넘길 서버 env override.

    우선순위: CLI 플래그 > .env.describe > (.env / compose 기본값).
    여기 담긴 키만 .env 를 덮고 나머지는 .env 상속.
    """
    overlay = load_env(DESCRIBE_ENV_PATH) if DESCRIBE_ENV_PATH.is_file() else {}
    if args.max_num_seqs is not None:
        overlay["MAX_NUM_SEQS"] = str(args.max_num_seqs)
    if args.max_model_len is not None:
        overlay["MAX_MODEL_LEN"] = str(args.max_model_len)
    return overlay


def run_describe(model_id: str, model_dir: Path, args, concurrency: int) -> int:
    """단일 모델 describe_eval 를 subprocess 로 호출. 반환: returncode."""
    cmd = [sys.executable, str(DESCRIBE_EVAL),
           "--from", str(model_dir), "--url", args.url, "--model", model_id,
           "--kind", args.kind, "--max-tokens", str(args.max_tokens),
           "--concurrency", str(concurrency)]
    if args.category:
        cmd += ["--category", args.category]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.gemini_model:
        cmd += ["--gemini-model", args.gemini_model]
    if not args.translate:
        cmd += ["--no-translate"]
    print(f"\n=== description 생성: {model_id} (동시 {concurrency}) ===", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="from_dir", required=True,
                    help="labeled 평가 결과 폴더 (각 <Model>/manifest.csv 보유). "
                         "예: results/sweep_labeled_20260529_095220")
    ap.add_argument("--models", default=None,
                    help="콤마 구분 모델 리스트 (생략 시 폴더에 manifest 가 있는 모든 모델). 단축 매칭 지원")
    ap.add_argument("--kind", default="fp,fn",
                    help="대상 케이스 종류 (describe_eval 로 전달). 기본: fp,fn")
    ap.add_argument("--category", default=None, help="카테고리 필터 콤마 구분")
    ap.add_argument("--limit", type=int, default=0, help="kind/카테고리당 case 수 (0=전수)")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="클라이언트 동시 요청 수 (생략 시 적용된 MAX_NUM_SEQS 에 자동 일치)")
    ap.add_argument("--max-tokens", type=int, default=512, help="영어 description 토큰 수 (기본 512)")
    ap.add_argument("--gemini-model", default=None,
                    help="번역에 쓸 Gemini 모델 (생략 시 .env GEMINI_TRANSLATE_MODEL)")
    ap.add_argument("--no-translate", dest="translate", action="store_false",
                    help="번역(Gemini) 건너뛰고 영어 description 만 저장")
    ap.set_defaults(translate=True)
    # --- describe 전용 서버 오버레이 (.env.describe 보다 우선) ---
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="서버 MAX_NUM_SEQS override (.env.describe 보다 우선)")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="서버 MAX_MODEL_LEN override (.env.describe 보다 우선)")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--load-timeout", type=int, default=900,
                    help="모델 로딩 헬스체크 대기(초). 기본 900 (15분)")
    ap.add_argument("--no-require-cached", dest="require_cached", action="store_false",
                    help="HF 캐시에 weight 없어도 시도 (기본은 자동 스킵)")
    ap.set_defaults(require_cached=True)
    args = ap.parse_args()

    sweep_dir = Path(args.from_dir)
    if not sweep_dir.is_dir():
        sys.exit(f"[에러] 폴더가 아닙니다: {sweep_dir}")

    # 대상 모델 결정: --models 가 있으면 그것, 없으면 MODELS 중 폴더에 manifest 가 있는 것.
    requested = parse_models(args.models) if args.models else list(MODELS)

    available, skipped_no_manifest, skipped_no_cache = [], [], []
    for m in requested:
        model_dir = sweep_dir / model_dirname(m)
        if not (model_dir / "manifest.csv").is_file():
            skipped_no_manifest.append(m)
            continue
        if args.require_cached and not cache_has_weights(m):
            skipped_no_cache.append(m)
            continue
        available.append(m)

    if skipped_no_manifest:
        print(f"[알림] manifest 없어 스킵: {[model_dirname(m) for m in skipped_no_manifest]}")
    if skipped_no_cache:
        print(f"[알림] 캐시 미존재 모델 자동 스킵: {skipped_no_cache}")
    if not available:
        sys.exit("[에러] description 생성 가능한 모델이 없습니다 (manifest+캐시 둘 다 필요).")

    # 서버 오버레이 + 클라이언트 동시성 결정
    overlay = build_overlay(args)
    base_env = load_env(ENV_PATH)
    eff_seqs = overlay.get("MAX_NUM_SEQS") or base_env.get("MAX_NUM_SEQS") or "16"
    concurrency = args.concurrency or int(eff_seqs)

    print(f"입력 폴더  : {sweep_dir}")
    print(f"대상 모델  : {available}")
    print(f"공통 옵션  : kind={args.kind}, category={args.category or '전체'}, "
          f"limit={args.limit or '전수'}, max_tokens={args.max_tokens}, "
          f"translate={'on' if args.translate else 'off'}")
    print(f"서버 오버레이: {overlay or '(없음 — .env 기본값)'}")
    print(f"클라이언트 동시성: {concurrency}\n")

    results = {}
    try:
        compose(["down"])  # 깨끗한 시작
        for i, model_id in enumerate(available, 1):
            print(f"\n----- [{i}/{len(available)}] {model_id} -----")
            if compose(["up", "-d"], model_id=model_id, extra_env=overlay) != 0:
                results[model_id] = "up_failed"
                continue
            if not wait_healthy(args.url, args.load_timeout):
                print(f"[실패] {model_id}: 헬스체크 타임아웃 → 스킵")
                results[model_id] = "load_timeout"
                compose(["down"])
                continue
            rc = run_describe(model_id, sweep_dir / model_dirname(model_id), args, concurrency)
            results[model_id] = "ok" if rc == 0 else f"failed(rc={rc})"
            # 마지막 모델은 띄워둔 채로 종료. 중간 모델은 다음을 위해 down.
            if model_id != available[-1]:
                compose(["down"])
    except KeyboardInterrupt:
        print("\n[중단] sweep 중단됨 — 컨테이너 정리 후 종료")
        compose(["down"])
    finally:
        print(f"\n=== describe sweep 종료 ===")
        for m in available:
            print(f"  {model_dirname(m):<20} {results.get(m, '미실행')}")
        print(f"  결과: {sweep_dir}/<모델명>/describe/")
        print(f"  분석: python src/analysis/describe_server.py --path {sweep_dir}")


if __name__ == "__main__":
    main()
