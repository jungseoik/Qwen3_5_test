#!/usr/bin/env python3
"""다중 모델 오탐감소율 sweep.

models.py 의 모델 리스트를 순서대로 돌면서 각각:
  1) docker compose down                       (이전 모델 종료)
  2) VLLM_MODEL 만 override 하여 docker compose up -d (그 모델로 띄움)
  3) /health 폴링                              (모델 로딩 대기)
  4) subprocess 로 fp_reduction.py 실행        (이 모델의 오탐감소율 측정)
  5) 결과를 sweep 폴더 하위로 격리 저장
끝나면 모델 × 카테고리 비교 표(summary.md/csv) 를 작성한다.

서버 모델 교체는 .env 를 건드리지 않고 docker compose 의 shell env override
(`VLLM_MODEL=<id> docker compose up -d`) 로 처리한다.
compose 변수 우선순위: shell env > .env 이므로 .env 의 VLLM_MODEL 은 그대로 보존된다.

sweep 종료 정책:
  - 정상 종료: 마지막 모델을 그대로 띄워둠 (다음 작업에 바로 사용 가능)
  - Ctrl+C: 컨테이너 정리(`docker compose down`) 후 종료

의존성: 표준 라이브러리만 사용.

사용 예
    python src/evaluation/sweep.py                          # 전 모델 × limit=50 (기본)
    python src/evaluation/sweep.py --limit 0                # 전수
    python src/evaluation/sweep.py --models 0.8B,9B         # 부분집합 (단축 매칭)
"""
import argparse
import csv
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from models import MODELS  # 같은 src/evaluation/ 폴더

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
HF_CACHE_HUB = Path.home() / ".cache" / "huggingface" / "hub"
FP_REDUCTION = REPO_ROOT / "src" / "evaluation" / "fp_reduction.py"
LABELED_EVAL = REPO_ROOT / "src" / "evaluation" / "labeled_eval.py"


def model_dirname(model_id: str) -> str:
    """`Qwen/Qwen3.5-0.8B` → `Qwen3.5-0.8B` (디렉토리·파일명용)."""
    return model_id.split("/")[-1]


def cache_has_weights(model_id: str) -> bool:
    """HF 캐시에 model_id 의 safetensors weight 가 있는지 확인."""
    d = HF_CACHE_HUB / ("models--" + model_id.replace("/", "--"))
    if not d.is_dir():
        return False
    return any(d.glob("snapshots/*/*.safetensors"))


def compose(cmd: list, model_id: str = None) -> int:
    """docker compose 명령 실행 (선택적 VLLM_MODEL override)."""
    env = os.environ.copy()
    if model_id:
        env["VLLM_MODEL"] = model_id
    full = ["docker", "compose"] + cmd
    note = f"   (VLLM_MODEL={model_id})" if model_id else ""
    print(f"$ {' '.join(full)}{note}", flush=True)
    return subprocess.run(full, cwd=REPO_ROOT, env=env).returncode


def wait_healthy(url: str, timeout: int) -> bool:
    """/health 가 200 응답할 때까지 폴링. 성공 True / 타임아웃 False."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
                if r.status == 200:
                    sys.stderr.write("\n")
                    return True
        except Exception:
            pass
        sys.stderr.write(f"\r모델 로딩 대기... {int(time.time()-t0)}s ")
        sys.stderr.flush()
        time.sleep(5)
    sys.stderr.write("\n")
    return False


def parse_models(s: str) -> list:
    """--models 인자 해석. 콤마 구분. 단축 매칭(예: '0.8B' → 'Qwen/Qwen3.5-0.8B') 지원."""
    if not s:
        return list(MODELS)
    requested = [x.strip() for x in s.split(",") if x.strip()]
    chosen = []
    for req in requested:
        if req in MODELS:
            chosen.append(req)
            continue
        match = [m for m in MODELS if req in m]
        if len(match) == 1:
            chosen.append(match[0])
        elif not match:
            sys.exit(f"[에러] '{req}' 와 매칭되는 모델이 models.py MODELS 에 없습니다.")
        else:
            sys.exit(f"[에러] '{req}' 가 모호합니다 ({len(match)}개 매칭): {match}")
    return chosen


def run_eval(model_id: str, sweep_dir: Path, args) -> dict:
    """단일 모델 평가를 subprocess 로 호출 (fp_reduction.py 또는 labeled_eval.py).
    --eval 에 따라 진입점이 바뀐다. returns: {"status": ..., "rows": csv 행}
    """
    model_dir = sweep_dir / model_dirname(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    script = LABELED_EVAL if args.eval == "labeled" else FP_REDUCTION
    # 모델 id 를 명시 전달 — 안 그러면 .env 의 VLLM_MODEL 을 들고 요청 → 서버와 불일치.
    cmd = [sys.executable, str(script),
           "--out", str(model_dir), "--url", args.url, "--model", model_id]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.category:
        cmd += ["--category", args.category]
    if args.concurrency:
        cmd += ["--concurrency", str(args.concurrency)]
    if args.collect:
        cmd += ["--collect", args.collect]
    if args.eval == "labeled" and args.path:
        cmd += ["--path", args.path]

    print(f"\n=== 평가 시작: {model_id} ({args.eval}) ===", flush=True)
    rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    if rc != 0:
        return {"status": "failed", "rc": rc}
    csv_path = model_dir / "eval.csv"
    if not csv_path.exists():
        return {"status": "no_output"}
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    return {"status": "ok", "rows": rows}


def _matrix_row(model: str, status_ok: bool, status: str, categories: list,
                by_cat: dict, col: str) -> str:
    """summary md 의 한 행 (모델 × 카테고리) 생성. col 은 csv 컬럼명."""
    if not status_ok:
        return (f"| {model} | " + " | ".join(["-"] * len(categories))
                + f" | - | `{status}` |\n")
    cells = []
    for c in categories:
        v = by_cat.get(c, {}).get(col, "")
        cells.append(f"{v}%" if v else "-")
    total = by_cat.get("합계", {}).get(col, "-")
    return f"| {model} | " + " | ".join(cells) + f" | **{total}%** | ok |\n"


def write_summary_fp(sweep_dir: Path, ordered_models: list, results: dict, categories: list):
    """기존 fp 모드: 오탐감소율 1개 표."""
    md_path = sweep_dir / "summary.md"
    csv_path = sweep_dir / "summary.csv"

    csv_cols = ["model", "category", "total", "valid", "no", "yes",
                "unparsed", "error", "fp_reduction_pct"]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_cols)
        for model in ordered_models:
            res = results.get(model, {})
            if res.get("status") != "ok":
                continue
            for row in res["rows"]:
                w.writerow([model, row["category"], row["total"], row["valid"],
                            row["no_filtered"], row["yes_passed"],
                            row["unparsed"], row["error"], row["fp_reduction_pct"]])

    with md_path.open("w") as f:
        f.write("# 모델 sweep 평가 요약 (오탐감소율)\n\n")
        f.write(f"- 일시: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"- sweep 폴더: `{sweep_dir.name}`\n")
        ok = [m for m in ordered_models if results.get(m, {}).get("status") == "ok"]
        f.write(f"- 모델: 평가 {len(ok)}개 / 전체 {len(ordered_models)}개\n\n")
        f.write("## 오탐감소율 (%)\n\n")
        header = "| 모델 | " + " | ".join(categories) + " | **전체** | 상태 |\n"
        sep = "|---|" + "|".join(["---"] * (len(categories) + 2)) + "|\n"
        f.write(header + sep)
        for model in ordered_models:
            res = results.get(model, {})
            ok = res.get("status") == "ok"
            by_cat = {r["category"]: r for r in res.get("rows", [])} if ok else {}
            f.write(_matrix_row(model, ok, res.get("status", "unknown"),
                                categories, by_cat, "fp_reduction_pct"))
        f.write("\n각 모델 상세는 같은 폴더의 `{모델명}/eval.md` / `manifest.csv` "
                "그리고 못 거른 케이스 썸네일 (`{모델명}/yes/{카테고리}/`) 을 참고하세요.\n")

    return md_path, csv_path


def write_summary_labeled(sweep_dir: Path, ordered_models: list, results: dict, categories: list):
    """라벨 모드: 오탐감소율 + 정탐률 두 표 + 카운트."""
    md_path = sweep_dir / "summary.md"
    csv_path = sweep_dir / "summary.csv"

    csv_cols = ["model", "category", "total", "positives", "negatives",
                "tp", "fn", "tn", "fp", "unparsed", "error",
                "fp_reduction_pct", "recall_pct"]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_cols)
        for model in ordered_models:
            res = results.get(model, {})
            if res.get("status") != "ok":
                continue
            for row in res["rows"]:
                w.writerow([model, row["category"], row["total"],
                            row["positives"], row["negatives"],
                            row["tp"], row["fn"], row["tn"], row["fp"],
                            row["unparsed"], row["error"],
                            row["fp_reduction_pct"], row["recall_pct"]])

    def write_table(f, title: str, col: str):
        f.write(f"## {title}\n\n")
        f.write("| 모델 | " + " | ".join(categories) + " | **전체** | 상태 |\n")
        f.write("|---|" + "|".join(["---"] * (len(categories) + 2)) + "|\n")
        for model in ordered_models:
            res = results.get(model, {})
            ok = res.get("status") == "ok"
            by_cat = {r["category"]: r for r in res.get("rows", [])} if ok else {}
            f.write(_matrix_row(model, ok, res.get("status", "unknown"),
                                categories, by_cat, col))
        f.write("\n")

    with md_path.open("w") as f:
        f.write("# 모델 sweep 평가 요약 (라벨 모드)\n\n")
        f.write(f"- 일시: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"- sweep 폴더: `{sweep_dir.name}`\n")
        ok = [m for m in ordered_models if results.get(m, {}).get("status") == "ok"]
        f.write(f"- 모델: 평가 {len(ok)}개 / 전체 {len(ordered_models)}개\n")
        f.write("- **오탐감소율** = `TN/(TN+FP)` (false 라벨 중 모델이 no 라 거른 비율)\n")
        f.write("- **정탐률**     = `TP/(TP+FN)` (true  라벨 중 모델이 yes 라 유지한 비율)\n\n")
        write_table(f, "오탐감소율 (%) — 모델이 오탐을 얼마나 잘 거르는가", "fp_reduction_pct")
        write_table(f, "정탐률 (%) — 모델이 진짜 사건을 얼마나 잘 유지하는가", "recall_pct")
        f.write("각 모델 상세는 같은 폴더의 `{모델명}/eval.md` / `manifest.csv`, "
                "그리고 오분류 썸네일 `{모델명}/fp/{카테고리}/` (놓친 오탐), "
                "`{모델명}/fn/{카테고리}/` (놓친 정탐) 을 참고하세요.\n")

    return md_path, csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=None,
                    help="콤마 구분 모델 리스트 (생략 시 models.py MODELS 전체). 단축 매칭 지원")
    ap.add_argument("--limit", type=int, default=0,
                    help="모델당 카테고리별 샘플 수. 기본 0=전수")
    ap.add_argument("--category", default=None, help="평가 카테고리 콤마 구분")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="동시 요청 수 (생략 시 .env MAX_NUM_SEQS)")
    ap.add_argument("--collect", default=None,
                    help='실패 케이스 수집 모드. fp 모드: yes/all/none / 라벨 모드: fp,fn 등. '
                         '생략 시 평가 모듈 디폴트 사용')
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--load-timeout", type=int, default=900,
                    help="모델 로딩 헬스체크 대기(초). 기본 900 (15분)")
    ap.add_argument("--no-require-cached", dest="require_cached", action="store_false",
                    help="HF 캐시에 weight 없어도 시도 (기본은 자동 스킵)")
    ap.add_argument("--eval", choices=["fp", "labeled"], default="fp",
                    help="평가 종류. fp=기존 오탐감소율 (기본), labeled=라벨 기반 오탐감소율+정탐률")
    ap.add_argument("--path", default=None,
                    help="라벨 모드 전용. labeled_eval 의 --path 로 전달 "
                         "(생략 시 최신 results/labeling/export/export_* 자동)")
    ap.set_defaults(require_cached=True)
    args = ap.parse_args()

    requested = parse_models(args.models)

    # 캐시 weight 사전 체크
    available, skipped = [], []
    for m in requested:
        if args.require_cached and not cache_has_weights(m):
            skipped.append(m)
        else:
            available.append(m)
    if skipped:
        print(f"[알림] 캐시 미존재 모델 자동 스킵: {skipped}")
    if not available:
        sys.exit("[에러] 평가 가능한 모델이 없습니다.")

    sweep_prefix = "sweep_labeled" if args.eval == "labeled" else "sweep"
    sweep_dir = RESULTS_DIR / f"{sweep_prefix}_{datetime.now():%Y%m%d_%H%M%S}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"sweep 폴더 : {sweep_dir}")
    print(f"평가 종류  : {args.eval}")
    print(f"평가 대상  : {available}")
    print(f"공통 옵션  : limit={args.limit or '전수'}, "
          f"concurrency={args.concurrency or '.env'}, collect={args.collect or '(모듈 기본)'}\n")

    # 모든 평가 대상에 결과 슬롯 준비 (스킵 모델도 표에 표시).
    results = {m: {"status": "skipped"} for m in skipped}
    categories_seen = []

    try:
        compose(["down"])  # 깨끗한 시작
        for i, model_id in enumerate(available, 1):
            print(f"\n----- [{i}/{len(available)}] {model_id} -----")
            if compose(["up", "-d"], model_id=model_id) != 0:
                results[model_id] = {"status": "up_failed"}
                continue
            if not wait_healthy(args.url, args.load_timeout):
                print(f"[실패] {model_id}: 헬스체크 타임아웃 → 스킵")
                results[model_id] = {"status": "load_timeout"}
                compose(["down"])
                continue
            res = run_eval(model_id, sweep_dir, args)
            results[model_id] = res
            if res.get("status") == "ok" and not categories_seen:
                categories_seen = [r["category"] for r in res["rows"]
                                   if r["category"] != "합계"]
            # 마지막 모델은 띄워둔 채로 종료. 중간 모델은 다음을 위해 down.
            if model_id != available[-1]:
                compose(["down"])
    except KeyboardInterrupt:
        print("\n[중단] sweep 중단됨 — 컨테이너 정리 후 종료")
        compose(["down"])
    finally:
        ordered = available + skipped
        writer = write_summary_labeled if args.eval == "labeled" else write_summary_fp
        md, csvf = writer(sweep_dir, ordered, results, categories_seen)
        print(f"\n=== sweep 종료 ===")
        print(f"  요약 md  : {md}")
        print(f"  요약 csv : {csvf}")
        print(f"  세부 결과: {sweep_dir}/<모델명>/")


if __name__ == "__main__":
    main()
