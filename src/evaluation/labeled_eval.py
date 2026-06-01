#!/usr/bin/env python3
"""라벨 기반 오탐감소율 + 정탐유지율 평가.

입력은 라벨링 도구가 만든 export 폴더로, 카테고리별 true(정탐)/false(오탐) 라벨이
폴더로 정리되어 있다 (`results/labeling/export/export_<TS>/{cat}/{true,false}/*.jpg`).

기존 fp_reduction.py 는 모든 이미지가 오탐이라는 가정 아래 "VLM 이 no 라고 거른 비율"
하나만 측정. 본 모듈은 라벨이 있으므로 두 지표를 동시에 산출한다:

    오탐감소율 = TN / (TN + FP)        false 라벨 중 모델이 no 라 거른 비율
    정탐유지율 = TP / positives         1차 양성(positives=TP+FN) 중 2차 VLM 이 유지한 비율
    손실률     = FN / positives         정탐을 놓친 비율 (= 100 − 정탐유지율)

정탐은 새로 만들 수 없고(1차에서 이미 검출됨) 2차에서 유지/손실만 가능하므로
"recall" 보다 "유지율" 프레임이 의미를 더 명확하게 보여준다 (수치는 동일).

평가 출력은 기존 평가와 동일하게 results/eval_labeled_<TS>/ 폴더 하나에 모이며,
오분류 케이스(fp/, fn/)는 자동 수집된다.

전제
    - docker compose 로 vLLM 서버가 떠 있어야 한다.
    - 입력 폴더 구조: {path}/{category}/{true,false}/*.jpg
의존성: 표준 라이브러리만 사용.

사용 예
    python src/evaluation/labeled_eval.py                          # 최신 export 자동
    python src/evaluation/labeled_eval.py --path /다른/경로
    python src/evaluation/labeled_eval.py --category fire,smoke
    python src/evaluation/labeled_eval.py --limit 50
"""
import argparse
import base64
import csv
import json
import random
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from fp_reduction import (
    ENV_PATH, RESULTS_DIR, SERVER_ENV_KEYS, Progress,
    build_payload, detect_model, load_env, parse_answer,
)
from prompts import PROMPTS, get_prompt

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELED_BASE = REPO_ROOT / "results" / "labeling" / "export"
DEFAULT_CATEGORIES = ["falldown", "fire", "smoke"]  # violence 제외 (true 데이터 없음)


def find_latest_export(base: Path) -> Path | None:
    if not base.is_dir():
        return None
    exports = sorted(p for p in base.glob("export_*") if p.is_dir())
    return exports[-1] if exports else None


def collect_images(root: Path, categories: list, limit: int, seed: int, max_neg: int = 0):
    """{(category, label): [Path,...]} — root/{cat}/{true,false}/*.jpg 수집.

    max_neg>0 (dev 서브셋 모드): positive(true) 는 전부 유지하고 negative(false) 만
    max_neg 개로 캡(seed 고정 랜덤). 정탐 표본이 희소하므로 retention 측정을 위해 전부 살린다.
    max_neg=0 (기본): 기존 동작 — limit 시 true/false 모두 limit 개로 캡.
    """
    rng = random.Random(seed)
    out = {}
    for cat in categories:
        for label in ("true", "false"):
            d = root / cat / label
            imgs = sorted(d.glob("*.jpg")) if d.is_dir() else []
            if max_neg:
                if label == "false" and len(imgs) > max_neg:
                    imgs = rng.sample(imgs, max_neg)
                # true 는 캡하지 않음
            elif limit and len(imgs) > limit:
                imgs = rng.sample(imgs, limit)
            out[(cat, label)] = imgs
    return out


def one_request(url: str, model: str, category: str, label: str, img: Path,
                max_tokens: int, prompt: str):
    payload = build_payload(model, img, prompt, max_tokens)
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            content = json.load(r)["choices"][0]["message"]["content"] or ""
        return {"category": category, "label": label, "image": img,
                "raw": content, "parsed": parse_answer(content)}
    except Exception as e:
        return {"category": category, "label": label, "image": img,
                "raw": f"<error: {e}>", "parsed": "error"}


def parse_collect_modes(s: str) -> set:
    """--collect 인자 — 콤마 구분 (tp, tn, fp, fn). 특수값 all/none.
    tp/tn 은 "잘 맞춘" 케이스. 4분면 전부 모으려면 --collect all (TN 은 양이 많으니 주의).
    """
    valid = {"tp", "tn", "fp", "fn"}
    if not s or s == "none":
        return set()
    if s == "all":
        return valid
    return {x.strip() for x in s.split(",") if x.strip() in valid}


def compute_confusion(records: list, categories: list) -> dict:
    """{category: {"tp":n,"fn":n,"tn":n,"fp":n,"unparsed":n,"error":n}}."""
    agg = {c: {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "unparsed": 0, "error": 0}
           for c in categories}
    for r in records:
        c = r["category"]
        d = agg[c]
        p = r["parsed"]
        if p in ("unparsed", "error"):
            d[p] += 1
        elif r["label"] == "true":
            d["tp" if p == "yes" else "fn"] += 1
        else:  # label == "false"
            d["fp" if p == "yes" else "tn"] += 1
    return agg


def rate(num: int, den: int) -> float:
    return (num / den * 100) if den else float("nan")


def fmt_rate(r: float) -> str:
    return "N/A" if r != r else f"{r:.1f}%"


def render(agg: dict, elapsed: float, total: int):
    """표 출력 + 행 데이터(리스트) 반환.
    columns: category,total,positives,negatives,tp,fn,tn,fp,unparsed,error,
             fp_reduction_pct,retention_pct,loss_pct
    """
    print(f"완료: {total}장 / {elapsed:.1f}s ({total/elapsed:.1f} img/s)\n")
    print("=== 카테고리별 결과 ===")
    print(f"{'카테고리':>10} {'총수':>5} {'true':>4} {'false':>5} "
          f"{'TP':>4} {'FN':>4} {'TN':>5} {'FP':>4} "
          f"{'미파싱':>5} {'에러':>4} {'오탐감소율':>10} {'정탐유지율':>10} {'손실률':>7}")
    print("-" * 102)

    rows = []
    grand = {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "unparsed": 0, "error": 0}
    for cat, d in agg.items():
        for k in grand:
            grand[k] += d[k]
        pos = d["tp"] + d["fn"]
        neg = d["tn"] + d["fp"]
        tot = pos + neg + d["unparsed"] + d["error"]
        spec = rate(d["tn"], d["tn"] + d["fp"])  # 오탐감소율
        ret = rate(d["tp"], pos)                 # 정탐유지율
        loss = rate(d["fn"], pos)                # 손실률
        print(f"{cat:>10} {tot:>5} {pos:>4} {neg:>5} "
              f"{d['tp']:>4} {d['fn']:>4} {d['tn']:>5} {d['fp']:>4} "
              f"{d['unparsed']:>5} {d['error']:>4} "
              f"{fmt_rate(spec):>10} {fmt_rate(ret):>10} {fmt_rate(loss):>7}")
        rows.append([cat, tot, pos, neg, d["tp"], d["fn"], d["tn"], d["fp"],
                     d["unparsed"], d["error"],
                     None if spec != spec else round(spec, 1),
                     None if ret != ret else round(ret, 1),
                     None if loss != loss else round(loss, 1)])

    gpos = grand["tp"] + grand["fn"]
    gneg = grand["tn"] + grand["fp"]
    gtot = gpos + gneg + grand["unparsed"] + grand["error"]
    gspec = rate(grand["tn"], grand["tn"] + grand["fp"])
    gret = rate(grand["tp"], gpos)
    gloss = rate(grand["fn"], gpos)
    print("-" * 102)
    print(f"{'합계':>10} {gtot:>5} {gpos:>4} {gneg:>5} "
          f"{grand['tp']:>4} {grand['fn']:>4} {grand['tn']:>5} {grand['fp']:>4} "
          f"{grand['unparsed']:>5} {grand['error']:>4} "
          f"{fmt_rate(gspec):>10} {fmt_rate(gret):>10} {fmt_rate(gloss):>7}")
    rows.append(["합계", gtot, gpos, gneg, grand["tp"], grand["fn"],
                 grand["tn"], grand["fp"], grand["unparsed"], grand["error"],
                 None if gspec != gspec else round(gspec, 1),
                 None if gret != gret else round(gret, 1),
                 None if gloss != gloss else round(gloss, 1)])

    print(f"\n오탐감소율 = TN / (TN+FP)    (false 라벨 중 모델이 no 라 거른 비율)")
    print(f"정탐유지율 = TP / positives  (1차 양성 중 2차 VLM 이 유지한 비율)")
    print(f"손실률     = FN / positives  (= 100 − 정탐유지율, 진짜 사건 놓친 비율)")
    return rows


def save(out_dir: Path, rows: list, model: str, args, env: dict, concurrency: int,
         categories: list, export_path: Path, prompt_by_cat: dict = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["category", "total", "positives", "negatives",
            "tp", "fn", "tn", "fp", "unparsed", "error",
            "fp_reduction_pct", "retention_pct", "loss_pct"]

    with (out_dir / "eval.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)

    with (out_dir / "eval.md").open("w") as f:
        f.write("# 라벨 기반 오탐감소율 + 정탐유지율 평가 결과\n\n")
        f.write(f"- 일시: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"- 모델: {model}\n")
        f.write(f"- 라벨 export: `{export_path}`\n")
        f.write(f"- 샘플: {'전수' if not args.limit else f'카테고리/라벨당 {args.limit}장'}\n")
        f.write(f"- 동시 요청: {concurrency}\n")
        f.write("- 지표 정의:\n")
        f.write("  - **오탐감소율** = `TN / (TN + FP)` (false 라벨 중 모델이 no 라 거른 비율)\n")
        f.write("  - **정탐유지율** = `TP / positives` (1차 양성 중 2차 VLM 이 유지한 비율)\n")
        f.write("  - **손실률**     = `FN / positives` (= 100 − 정탐유지율)\n\n")
        if env:
            f.write("## 서버 설정 (.env 기준)\n\n")
            f.write("| key | value |\n|---|---|\n")
            for k in SERVER_ENV_KEYS:
                f.write(f"| {k} | `{env.get(k, '(미설정)')}` |\n")
            f.write("\n")
        f.write("## 결과\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join("" if x is None else str(x) for x in r) + " |\n")

        f.write("\n## 사용 프롬프트\n\n")
        for cat in categories:
            prompt = (prompt_by_cat or {}).get(cat) or PROMPTS.get(cat, "(기본 프롬프트)")
            f.write(f"### {cat}\n\n```\n{prompt}\n```\n\n")

    print(f"\n저장: {out_dir}/  (eval.md, eval.csv)")


def _kind_of(label: str, parsed: str) -> str | None:
    """(label, parsed) → 혼동행렬 분면. 그 외는 None (unparsed/error)."""
    if label == "true" and parsed == "yes":
        return "tp"   # 진짜 → 잘 유지
    if label == "true" and parsed == "no":
        return "fn"   # 진짜 → 놓침
    if label == "false" and parsed == "no":
        return "tn"   # 오탐 → 잘 거름
    if label == "false" and parsed == "yes":
        return "fp"   # 오탐 → 놓침
    return None


def collect_cases(records: list, out_dir: Path, modes: set, symlink: bool) -> dict:
    """선택된 modes 의 케이스를 {kind}/{cat}/ 로 모은다 + manifest.csv 기록.
    kind ∈ {tp, fn, tn, fp}. tp/tn 은 잘 맞춘 케이스, fp/fn 은 오분류.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    counts = {k: 0 for k in modes}
    for r in records:
        kind = _kind_of(r["label"], r["parsed"])
        if kind is None or kind not in modes:
            continue
        counts[kind] += 1
        src = Path(r["image"])
        dest_dir = out_dir / kind / r["category"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if symlink:
            dest.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dest)
        manifest_rows.append({
            "kind": kind, "category": r["category"], "label": r["label"],
            "parsed": r["parsed"], "source": str(src), "copied": str(dest),
            "raw_response": (r["raw"] or "").strip().replace("\n", " ")[:500],
        })
    if manifest_rows:
        with (out_dir / "manifest.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["kind", "category", "label",
                                              "parsed", "source", "copied", "raw_response"])
            w.writeheader()
            w.writerows(manifest_rows)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=None,
                    help=f"라벨 export 폴더 (생략 시 {LABELED_BASE} 아래 최신 export_* 자동)")
    ap.add_argument("--category", default=",".join(DEFAULT_CATEGORIES),
                    help=f"평가 카테고리 콤마 구분 (기본 {','.join(DEFAULT_CATEGORIES)})")
    ap.add_argument("--limit", type=int, default=0,
                    help="카테고리/라벨당 샘플 수 (0=전수)")
    ap.add_argument("--max-neg", type=int, default=0,
                    help="dev 서브셋: negative 만 N개로 캡, positive 는 전부 유지 (0=미사용). "
                         "프롬프트 최적화 반복용. limit 보다 우선")
    ap.add_argument("--prompt-file", default=None,
                    help="이 파일 내용을 프롬프트로 사용(--category 의 모든 카테고리에 적용, "
                         "보통 단일 카테고리 최적화 후보 평가용). 생략 시 prompts.get_prompt 사용")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=None,
                    help="동시 요청 수 (생략 시 .env MAX_NUM_SEQS)")
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default=None,
                    help="모델 id (생략 시 .env VLLM_MODEL → /v1/models 자동감지)")
    ap.add_argument("--out", default=None,
                    help="결과 저장 폴더 (생략 시 results/eval_labeled_<타임스탬프>/)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--collect", default="fp,fn",
                    help='케이스 수집 모드 (콤마 구분). 항목: tp/tn/fp/fn. '
                         '특수값 all (4분면 전부, TN 양 많음 주의) / none. 기본: fp,fn')
    ap.add_argument("--symlink", action="store_true",
                    help="수집 시 복사 대신 심볼릭 링크")
    args = ap.parse_args()

    # 입력 경로 결정
    if args.path:
        export_path = Path(args.path)
    else:
        export_path = find_latest_export(LABELED_BASE)
        if not export_path:
            raise SystemExit(
                f"[에러] {LABELED_BASE} 아래 export_* 폴더가 없습니다.\n"
                f"  먼저 라벨링 도구로 라벨을 만들거나 --path 로 경로를 지정하세요.")
    if not export_path.is_dir():
        raise SystemExit(f"[에러] 경로가 폴더가 아닙니다: {export_path}")

    # 서버 / 모델 / 동시성
    env = load_env(ENV_PATH)
    concurrency = args.concurrency or int(env.get("MAX_NUM_SEQS", 16))
    try:
        model = args.model or env.get("VLLM_MODEL") or detect_model(args.url)
    except Exception as e:
        raise SystemExit(f"[에러] 서버 연결 실패 ({args.url}). docker compose up -d 로 기동하세요.\n  {e}")

    # 카테고리 별 이미지 수집
    categories = [c.strip() for c in args.category.split(",") if c.strip()]
    images = collect_images(export_path, categories, args.limit, args.seed, args.max_neg)
    tasks = [(cat, label, img)
             for (cat, label), imgs in images.items() for img in imgs]
    if not tasks:
        raise SystemExit(
            f"[에러] 평가할 이미지가 없습니다. 경로/카테고리 확인:\n  {export_path}")

    # 카테고리별 프롬프트 결정: --prompt-file 우선, 없으면 모델별 최적/baseline
    prompt_override = None
    if args.prompt_file:
        prompt_override = Path(args.prompt_file).read_text().strip()
    prompt_by_cat = {c: (prompt_override if prompt_override is not None else get_prompt(c, model))
                     for c in categories}
    prompt_src = (f"파일 {args.prompt_file}" if prompt_override is not None
                  else "prompts.get_prompt(모델별 최적/baseline)")

    # 헤더
    print(f"서버         : {args.url}")
    print(f"모델         : {model}")
    print(f"라벨 export  : {export_path}")
    print(f"카테고리     : {','.join(categories)}")
    sub = (f"dev(neg≤{args.max_neg}, pos 전부)" if args.max_neg
           else ('전수' if not args.limit else f'카테고리/라벨당 {args.limit}장'))
    print(f"샘플         : {sub}")
    print(f"프롬프트     : {prompt_src}")
    print(f"동시 요청    : {concurrency}")
    counts_str = ", ".join(
        f"{c}(t:{len(images.get((c,'true'),[]))}/f:{len(images.get((c,'false'),[]))})"
        for c in categories)
    print(f"총 이미지    : {len(tasks)}장   [{counts_str}]")
    if env:
        print(f"\n서버 설정 (.env 기준):")
        for k in SERVER_ENV_KEYS:
            print(f"  {k:<24} = {env.get(k, '(미설정)')}")
    print("\n평가 중...", flush=True)

    # 실행
    records = []
    progress = Progress(len(tasks))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(one_request, args.url, model, c, lbl, img,
                             args.max_tokens, prompt_by_cat[c])
                   for c, lbl, img in tasks]
        for done_idx, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            progress.update(done_idx)
    elapsed = time.perf_counter() - t0

    agg = compute_confusion(records, categories)
    rows = render(agg, elapsed, len(tasks))

    if not args.no_save:
        out_dir = Path(args.out) if args.out else (
            RESULTS_DIR / f"eval_labeled_{datetime.now():%Y%m%d_%H%M%S}")
        save(out_dir, rows, model, args, env, concurrency, categories, export_path, prompt_by_cat)
        modes = parse_collect_modes(args.collect)
        if modes:
            c = collect_cases(records, out_dir, modes, symlink=args.symlink)
            kinds = ", ".join(f"{k}:{v}" for k, v in c.items() if k in modes)
            print(f"수집: {kinds} → {out_dir}/")


if __name__ == "__main__":
    main()
