#!/usr/bin/env python3
"""오분류(fp/fn) 케이스에 대한 모델 description 생성 (단일 모델, 진단용).

라벨 평가(labeled_eval.py)는 yes/no 한 토큰만 받아 "왜 그렇게 판정했는가" 를 알 수 없다.
본 모듈은 평가가 이미 만든 오분류 목록(manifest.csv)을 입력으로 받아, 각 케이스에 대해
모델에게 장면을 한국어로 풀어 설명(description)하게 한다. 이로써 정탐유지율이 떨어지는 원인이
모델의 이미지 인식 실패인지, base yes/no 프롬프트의 과민/둔감인지 사람이 진단할 수 있다.

입력은 평가 결과의 모델 폴더(`results/sweep_labeled_*/<Model>/` 또는 `results/eval_labeled_*/`)이며
그 안의 manifest.csv 를 읽는다 (컬럼: kind,category,label,parsed,source,...).
출력은 그 폴더 안에 describe/ 서브폴더로 제자리(co-locate) 저장 — 기존 manifest/eval 은 건드리지 않는다.

    describe/describe.jsonl   케이스당 1줄 {id,kind,category,label,base_verdict,source,description}
    describe/describe.csv     동일 내용 표
    describe/describe.md       모델·프롬프트·건수 요약

전제
    - docker compose 로 vLLM 서버가 떠 있어야 한다 (description 을 만들 그 모델).
    - 입력 폴더에 manifest.csv 가 있어야 한다 (평가 시 --collect fp,fn 로 생성됨).
의존성: 표준 라이브러리만 사용.

사용 예
    python src/evaluation/describe_eval.py --from results/sweep_labeled_20260529_095220/Qwen3.5-4B
    python src/evaluation/describe_eval.py --from <폴더> --kind fn         # 놓친 정탐만
    python src/evaluation/describe_eval.py --from <폴더> --category smoke
    python src/evaluation/describe_eval.py --from <폴더> --limit 20        # kind/카테고리당 20장
"""
import argparse
import csv
import json
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from fp_reduction import (
    ENV_PATH, SERVER_ENV_KEYS, Progress,
    build_payload, detect_model, load_env,
)
from prompts import get_prompt, get_describe_prompt
from translate import translate_to_korean, get_key_and_model, TranslateError

VALID_KINDS = ("tp", "tn", "fp", "fn")


def read_manifest(model_dir: Path) -> list:
    """<model_dir>/manifest.csv 를 dict 리스트로 읽는다."""
    mpath = model_dir / "manifest.csv"
    if not mpath.is_file():
        raise SystemExit(
            f"[에러] manifest.csv 가 없습니다: {mpath}\n"
            f"  평가를 --collect fp,fn (기본) 로 먼저 돌려 오분류 목록을 만드세요.")
    with mpath.open() as f:
        return list(csv.DictReader(f))


def parse_kinds(s: str) -> set:
    """--kind 인자 — 콤마 구분 (fp, fn, tp, tn). 특수값 all."""
    if not s or s == "all":
        return set(VALID_KINDS)
    return {x.strip() for x in s.split(",") if x.strip() in VALID_KINDS}


def select_cases(rows: list, kinds: set, categories: list, limit: int) -> list:
    """manifest 행에서 (kind, category) 필터 + limit 적용. id 부여."""
    counter = defaultdict(int)
    out = []
    for r in rows:
        kind, cat = r.get("kind", ""), r.get("category", "")
        if kind not in kinds:
            continue
        if categories and cat not in categories:
            continue
        if limit and counter[(kind, cat)] >= limit:
            continue
        idx = counter[(kind, cat)]
        counter[(kind, cat)] += 1
        out.append({
            "id": f"{kind}_{cat}_{idx:04d}",
            "kind": kind,
            "category": cat,
            "label": r.get("label", ""),
            "base_verdict": r.get("parsed", ""),  # 평가 당시 yes/no
            "source": r.get("source", ""),
        })
    return out


def _chat(url: str, payload: bytes) -> str:
    """단일 vLLM chat 요청 → content 문자열 (실패 시 예외)."""
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return (json.load(r)["choices"][0]["message"]["content"] or "").strip()


def one_request(url: str, model: str, case: dict, max_tokens: int,
                translate: bool, gemini_key: str, gemini_model: str) -> dict:
    """단일 케이스: (1) vLLM 으로 영어 description 생성 → (2) Gemini 로 한국어 번역.

    영어 묘사는 vLLM(모델 평가 대상)이, 한국어 번역은 Gemini API 가 담당한다.
    case 에 description(영어 원문) + description_ko(한국어 번역) 를 추가해 반환.
    이미지/요청 실패는 description 에 `<error: ...>` 로 기록하고 번역은 건너뛴다.
    """
    img = Path(case["source"])
    out = dict(case)
    out["description"] = ""
    out["description_ko"] = ""

    if not img.is_file():
        out["description"] = f"<error: 원본 이미지 없음 {img}>"
        return out

    # (1) 영어 묘사 (vLLM)
    try:
        desc = _chat(url, build_payload(model, img, get_describe_prompt(case["category"]), max_tokens))
    except Exception as e:
        out["description"] = f"<error: {e}>"
        return out
    out["description"] = desc

    # (2) 한국어 번역 (Gemini) — 영어 원문이 정상일 때만
    if translate and desc and not desc.startswith("<error"):
        try:
            out["description_ko"] = translate_to_korean(desc, gemini_key, gemini_model)
        except TranslateError as e:
            out["description_ko"] = f"<error: {e}>"
    return out


def save(out_dir: Path, results: list, model: str, model_dir: Path,
         kinds: set, categories: list, max_tokens: int, env: dict, translate: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    # jsonl
    with (out_dir / "describe.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # csv
    cols = ["id", "kind", "category", "label", "base_verdict", "source",
            "description", "description_ko"]
    textcols = {"description", "description_ko"}
    with (out_dir / "describe.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({k: (r.get(k, "") or "").replace("\n", " ") if k in textcols
                        else r.get(k, "") for k in cols})

    # 건수 집계 (kind × category)
    counts = defaultdict(lambda: defaultdict(int))
    for r in results:
        counts[r["kind"]][r["category"]] += 1
    cats_seen = sorted({r["category"] for r in results})

    with (out_dir / "describe.md").open("w") as f:
        f.write("# 오분류 케이스 description (진단)\n\n")
        f.write(f"- 일시: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"- 모델: {model}\n")
        f.write(f"- 입력 평가 폴더: `{model_dir}`\n")
        f.write(f"- 대상 kind: {', '.join(sorted(kinds))}\n")
        f.write(f"- 카테고리: {', '.join(categories) if categories else '전체'}\n")
        f.write(f"- max_tokens: {max_tokens}\n")
        f.write(f"- 생성: 영어 묘사 → {'한국어 번역(translate)' if translate else '번역 안 함'}\n")
        f.write(f"- 총 case: {len(results)}건\n\n")
        if env:
            f.write("## 서버 설정 (.env 기준)\n\n| key | value |\n|---|---|\n")
            for k in SERVER_ENV_KEYS:
                f.write(f"| {k} | `{env.get(k, '(미설정)')}` |\n")
            f.write("\n")
        f.write("## kind × 카테고리 건수\n\n")
        f.write("| kind | " + " | ".join(cats_seen) + " | 합계 |\n")
        f.write("|---|" + "|".join(["---"] * (len(cats_seen) + 1)) + "|\n")
        for kind in sorted(counts):
            row = [str(counts[kind].get(c, 0)) for c in cats_seen]
            tot = sum(counts[kind].values())
            f.write(f"| {kind} | " + " | ".join(row) + f" | {tot} |\n")
        f.write("\n## 사용 프롬프트\n\n")
        for cat in cats_seen:
            f.write(f"### {cat} — base (yes/no)\n\n```\n{get_prompt(cat)}\n```\n\n")
            f.write(f"### {cat} — describe (영어 생성)\n\n```\n{get_describe_prompt(cat)}\n```\n\n")

    print(f"\n저장: {out_dir}/  (describe.jsonl, describe.csv, describe.md)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="from_dir", required=True,
                    help="평가 결과의 모델 폴더 (manifest.csv 보유). "
                         "예: results/sweep_labeled_*/Qwen3.5-4B")
    ap.add_argument("--kind", default="fp,fn",
                    help="대상 케이스 종류 콤마 구분 (fp/fn/tp/tn). 특수값 all. 기본: fp,fn")
    ap.add_argument("--category", default=None,
                    help="카테고리 필터 콤마 구분 (생략 시 manifest 전체)")
    ap.add_argument("--limit", type=int, default=0,
                    help="kind/카테고리당 case 수 (0=전수)")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="동시 요청 수 (생략 시 .env MAX_NUM_SEQS)")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="영어 description 응답 토큰 수 (기본 512)")
    ap.add_argument("--no-translate", dest="translate", action="store_false",
                    help="한국어 번역(Gemini) 단계를 건너뛰고 영어 description 만 저장")
    ap.add_argument("--gemini-model", default=None,
                    help="번역에 쓸 Gemini 모델 (생략 시 .env GEMINI_TRANSLATE_MODEL → 기본값)")
    ap.set_defaults(translate=True)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default=None,
                    help="모델 id (생략 시 .env VLLM_MODEL → /v1/models 자동감지)")
    ap.add_argument("--out", default=None,
                    help="저장 폴더 (생략 시 <from>/describe/)")
    args = ap.parse_args()

    model_dir = Path(args.from_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"[에러] 폴더가 아닙니다: {model_dir}")

    rows = read_manifest(model_dir)
    kinds = parse_kinds(args.kind)
    if not kinds:
        raise SystemExit(f"[에러] 유효한 --kind 가 없습니다 (fp/fn/tp/tn): {args.kind!r}")
    categories = [c.strip() for c in args.category.split(",")] if args.category else None
    cases = select_cases(rows, kinds, categories, args.limit)
    if not cases:
        raise SystemExit(
            f"[에러] 대상 case 가 없습니다. manifest 에 kind={kinds} "
            f"{'category=' + str(categories) if categories else ''} 케이스가 있는지 확인하세요.")

    env = load_env(ENV_PATH)
    concurrency = args.concurrency or int(env.get("MAX_NUM_SEQS", 16))
    try:
        model = args.model or env.get("VLLM_MODEL") or detect_model(args.url)
    except Exception as e:
        raise SystemExit(f"[에러] 서버 연결 실패 ({args.url}). docker compose up -d 로 기동하세요.\n  {e}")

    # 번역(Gemini) 키/모델 확인. 키 없으면 번역 비활성화하고 경고.
    gemini_key, gemini_model = get_key_and_model(None, args.gemini_model)
    translate = args.translate
    if translate and not gemini_key:
        print("[경고] GEMINI_API_KEY 가 없어 번역을 건너뜁니다 (.env 확인). 영어 description 만 저장합니다.")
        translate = False

    out_dir = Path(args.out) if args.out else (model_dir / "describe")

    # 헤더
    by_kind = defaultdict(int)
    for c in cases:
        by_kind[c["kind"]] += 1
    print(f"서버         : {args.url}")
    print(f"모델         : {model}")
    print(f"입력 폴더    : {model_dir}")
    print(f"대상 kind    : {', '.join(sorted(kinds))}  [{', '.join(f'{k}:{v}' for k,v in sorted(by_kind.items()))}]")
    print(f"카테고리     : {', '.join(categories) if categories else '전체'}")
    print(f"총 case      : {len(cases)}건")
    print(f"동시 요청    : {concurrency}")
    print(f"max_tokens   : {args.max_tokens}")
    print(f"번역         : {'영어→한국어 (Gemini ' + gemini_model + ')' if translate else '안 함 (영어만)'}")
    print(f"저장 폴더    : {out_dir}")
    print("\ndescription 생성 중...", flush=True)

    results = []
    progress = Progress(len(cases))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(one_request, args.url, model, c, args.max_tokens,
                             translate, gemini_key, gemini_model) for c in cases]
        for done_idx, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            progress.update(done_idx)
    elapsed = time.perf_counter() - t0
    print(f"완료: {len(cases)}건 / {elapsed:.1f}s ({len(cases)/elapsed:.1f} case/s)")

    # id 순으로 정렬 저장 (생성 순서 무관하게 안정적 보기)
    results.sort(key=lambda r: r["id"])
    save(out_dir, results, model, model_dir, kinds, categories or [],
         args.max_tokens, env, translate)


if __name__ == "__main__":
    main()
