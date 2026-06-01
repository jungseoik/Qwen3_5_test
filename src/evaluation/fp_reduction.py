#!/usr/bin/env python3
"""오탐감소율 평가.

카테고리별 오탐 썸네일을 vLLM 서버에 카테고리별 프롬프트로 보내 yes/no 검증을
받고, "no"(진짜 이벤트 아님) 비율 = 오탐감소율을 측정한다. 동시 요청으로 처리한다.

전제
    - docker compose 로 vLLM 서버가 떠 있어야 한다 (기본 http://localhost:8000).
    - 데이터가 {path}/by_category/{category}/thumbnail/{date}/*.jpg 구조여야 한다.
    - 프롬프트는 src/evaluation/prompts.py 에서 가져온다 (그 파일을 고쳐 튜닝).
의존성: 표준 라이브러리만 사용.

사용 예
    python src/evaluation/fp_reduction.py                          # 전수
    python src/evaluation/fp_reduction.py --limit 50               # 카테고리당 50장
    python src/evaluation/fp_reduction.py --category fire,smoke --concurrency 16
    python src/evaluation/fp_reduction.py --out results/eval       # md/csv 저장
"""
import argparse
import base64
import csv
import json
import random
import re
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from prompts import PROMPTS, get_prompt  # 같은 src/evaluation/ 폴더

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "nas/data/khonkaen"
RESULTS_DIR = REPO_ROOT / "results"
ENV_PATH = REPO_ROOT / ".env"
_YESNO = re.compile(r"[a-zA-Z]+")

# 서버 동작에 영향을 주는 .env 변수들 — 헤더/결과파일에 함께 기록한다.
SERVER_ENV_KEYS = (
    "VLLM_MODEL", "MAX_NUM_SEQS", "MAX_NUM_BATCHED_TOKENS",
    "GPU_MEMORY_UTILIZATION", "KV_CACHE_MEMORY_BYTES", "MAX_MODEL_LEN",
    "EXTRA_VLLM_ARGS",
)


def load_env(path: Path) -> dict:
    """간이 .env 파서 — KEY=VALUE 라인만 추출 (따옴표/Export 미지원)."""
    env = {}
    if not path.is_file():
        return env
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def detect_model(url: str) -> str:
    with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as r:
        return json.load(r)["data"][0]["id"]


def collect_images(root: Path, categories, limit: int, seed: int):
    """{category: [Path,...]} — by_category/{cat}/thumbnail/ 아래 jpg 수집 (limit 시 랜덤 샘플)."""
    by_cat = root / "by_category"
    if not by_cat.is_dir():
        raise SystemExit(f"[에러] 기대한 구조가 아닙니다: '{by_cat}' 없음.")
    rng = random.Random(seed)
    out = {}
    cats = categories or sorted(p.name for p in by_cat.iterdir() if p.is_dir())
    for cat in cats:
        thumb_dir = by_cat / cat / "thumbnail"
        imgs = sorted(thumb_dir.rglob("*.jpg")) if thumb_dir.is_dir() else []
        if limit and len(imgs) > limit:
            imgs = rng.sample(imgs, limit)
        out[cat] = imgs
    return out


def build_payload(model: str, img_path: Path, prompt: str, max_tokens: int) -> bytes:
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    return json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        # Qwen3 계열 4B+ 는 기본적으로 reasoning(thinking) 모드 ON 이라 답 앞에
        # <think>...</think> 가 먼저 나옴. max_tokens=1 이면 첫 thinking 토큰만 잡혀 unparsed.
        # 이 옵션으로 thinking 비활성화 → 바로 yes/no 출력. (지원 안 하는 작은 모델은 무시)
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()


def parse_answer(content: str) -> str:
    """응답에서 yes/no 추출. 그 외는 'unparsed'."""
    m = _YESNO.search(content or "")
    if not m:
        return "unparsed"
    w = m.group().lower()
    return w if w in ("yes", "no") else "unparsed"


class Progress:
    """stderr에 \\r로 갱신되는 진행 바 (tqdm 미사용, stdlib 전용).

    0.1초 이내 호출은 throttle. 마지막 호출 시 줄바꿈으로 마무리.
    """

    def __init__(self, total: int, bar_len: int = 30):
        self.total = total
        self.bar_len = bar_len
        self.t0 = time.perf_counter()
        self.last = 0.0

    def update(self, done: int):
        now = time.perf_counter()
        if done < self.total and now - self.last < 0.1:
            return
        self.last = now
        elapsed = now - self.t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else 0.0
        fill = int(self.bar_len * done / self.total) if self.total else self.bar_len
        bar = "#" * fill + "-" * (self.bar_len - fill)
        pct = done / self.total * 100 if self.total else 100.0
        sys.stderr.write(
            f"\r[{bar}] {done}/{self.total} {pct:5.1f}%  "
            f"{rate:5.1f} img/s  ETA {eta:5.1f}s "
        )
        sys.stderr.flush()
        if done == self.total:
            sys.stderr.write("\n")


def one_request(url: str, model: str, category: str, img: Path, max_tokens: int):
    """단일 요청. {category, image, raw, parsed} 레코드 반환.
    parsed 는 yes/no/unparsed/error 중 하나.
    """
    payload = build_payload(model, img, get_prompt(category), max_tokens)
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            content = json.load(r)["choices"][0]["message"]["content"] or ""
        return {"category": category, "image": img, "raw": content,
                "parsed": parse_answer(content)}
    except Exception as e:
        return {"category": category, "image": img, "raw": f"<error: {e}>",
                "parsed": "error"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=str(DEFAULT_PATH), help=f"데이터셋 루트 (기본: {DEFAULT_PATH})")
    ap.add_argument("--category", default=None, help="평가할 카테고리 (콤마 구분). 생략 시 전체")
    ap.add_argument("--limit", type=int, default=0, help="카테고리당 샘플 수 (0=전수)")
    ap.add_argument("--seed", type=int, default=0, help="샘플링 시드 (재현용)")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="동시 요청 수 (생략 시 .env MAX_NUM_SEQS 사용)")
    ap.add_argument("--max-tokens", type=int, default=1, help="응답 토큰 수 (yes/no면 1)")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default=None,
                    help="모델 id (생략 시 .env VLLM_MODEL → 그것도 없으면 /v1/models 자동감지)")
    ap.add_argument("--out", default=None,
                    help="결과 저장 폴더 (eval.md/eval.csv/manifest.csv/yes 가 그 안에 모임). "
                         "생략 시 results/eval_{타임스탬프}/")
    ap.add_argument("--no-save", action="store_true", help="결과 파일을 저장하지 않음")
    ap.add_argument("--collect", default="yes",
                    help='실패 케이스 썸네일 수집 모드. 콤마 구분 ("yes,unparsed,error"), '
                         '특수값 "all" / "none". 기본: yes (못 거른 오탐만)')
    ap.add_argument("--symlink", action="store_true",
                    help="썸네일 수집 시 복사 대신 심볼릭 링크 사용 (디스크 절약)")
    args = ap.parse_args()

    # 서버 설정의 단일 진실(.env) 로드. 평가 실행 기준 + 결과 파일 기록.
    env = load_env(ENV_PATH)
    concurrency = args.concurrency or int(env.get("MAX_NUM_SEQS", 16))

    try:
        model = args.model or env.get("VLLM_MODEL") or detect_model(args.url)
    except Exception as e:
        raise SystemExit(f"[에러] 서버 연결 실패 ({args.url}). docker compose up -d 로 기동하세요.\n  {e}")

    cats = [c.strip() for c in args.category.split(",")] if args.category else None
    images = collect_images(Path(args.path), cats, args.limit, args.seed)
    tasks = [(cat, img) for cat, imgs in images.items() for img in imgs]
    if not tasks:
        raise SystemExit("[에러] 평가할 이미지가 없습니다.")

    conc_src = "--concurrency" if args.concurrency else (
        "env MAX_NUM_SEQS" if "MAX_NUM_SEQS" in env else "기본값 16")

    print(f"server      : {args.url}")
    print(f"model       : {model}")
    print(f"data        : {args.path}")
    print(f"limit       : {'전수' if not args.limit else f'카테고리당 {args.limit}장'}")
    print(f"concurrency : {concurrency} ({conc_src})")
    print(f"총 이미지   : {len(tasks)}장 ({', '.join(f'{c}:{len(v)}' for c, v in images.items())})")

    if env:
        print(f"\n서버 설정 (.env 기준):")
        for k in SERVER_ENV_KEYS:
            print(f"  {k:<24} = {env.get(k, '(미설정)')}")
    else:
        print(f"\n[알림] {ENV_PATH} 를 찾을 수 없어 서버 설정을 기록하지 못합니다.")

    print("\n평가 중...", flush=True)

    records = []
    progress = Progress(len(tasks))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(one_request, args.url, model, c, img, args.max_tokens)
                   for c, img in tasks]
        for done_idx, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            progress.update(done_idx)
    elapsed = time.perf_counter() - t0

    # 레코드 → 카테고리별 카운트.
    agg = {c: {"no": 0, "yes": 0, "unparsed": 0, "error": 0} for c in images}
    for r in records:
        agg[r["category"]][r["parsed"]] += 1

    rows = render(agg, elapsed, len(tasks))

    if not args.no_save:
        out_dir = Path(args.out) if args.out else (
            RESULTS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M%S}")
        out_dir.mkdir(parents=True, exist_ok=True)
        save(out_dir, rows, model, args, env, concurrency, images)
        modes = parse_collect_modes(args.collect)
        if modes:
            n_collected = collect_cases(records, out_dir, modes, symlink=args.symlink)
            print(f"수집: {n_collected}장 ({','.join(sorted(modes))}) → {out_dir}/")


def _rate(no: int, yes: int) -> float:
    """오탐감소율 = no / (no + yes). unparsed/error는 분모에서 제외."""
    valid = no + yes
    return (no / valid * 100) if valid else float("nan")


def render(agg: dict, elapsed: float, total: int):
    """표 출력 후 행 데이터(리스트) 반환.
    오탐감소율은 유효 응답(no+yes)만 분모로 사용. unparsed/error 는 측정에서 제외.
    """
    print(f"완료: {total}장 / {elapsed:.1f}s ({total/elapsed:.1f} img/s)\n")
    print("=== 오탐감소율 (정답=no, no일수록 좋음. unparsed/error 제외) ===")
    hdr = (f"{'카테고리':>10} {'총수':>6} {'유효':>6} {'no(필터)':>9} "
           f"{'yes(통과)':>10} {'미파싱':>7} {'에러':>5} {'오탐감소율':>9}")
    print(hdr)
    print("-" * 70)
    rows = []
    tot = {"no": 0, "yes": 0, "unparsed": 0, "error": 0}
    for cat, d in agg.items():
        n = d["no"] + d["yes"] + d["unparsed"] + d["error"]
        valid = d["no"] + d["yes"]
        for k in tot:
            tot[k] += d[k]
        rate = _rate(d["no"], d["yes"])
        rate_s = f"{rate:>8.1f}%" if rate == rate else "      N/A"  # NaN 체크
        print(f"{cat:>10} {n:>6} {valid:>6} {d['no']:>9} {d['yes']:>10} "
              f"{d['unparsed']:>7} {d['error']:>5} {rate_s}")
        rows.append([cat, n, valid, d["no"], d["yes"], d["unparsed"], d["error"],
                     None if rate != rate else round(rate, 1)])
    gn = sum(tot.values())
    gvalid = tot["no"] + tot["yes"]
    grate = _rate(tot["no"], tot["yes"])
    grate_s = f"{grate:>8.1f}%" if grate == grate else "      N/A"
    print("-" * 70)
    print(f"{'합계':>10} {gn:>6} {gvalid:>6} {tot['no']:>9} {tot['yes']:>10} "
          f"{tot['unparsed']:>7} {tot['error']:>5} {grate_s}")
    rows.append(["합계", gn, gvalid, tot["no"], tot["yes"], tot["unparsed"], tot["error"],
                 None if grate != grate else round(grate, 1)])
    return rows


def save(out_dir: Path, rows: list, model: str, args, env: dict, concurrency: int,
         images: dict):
    """한 평가의 모든 산출물을 out_dir 안에 저장한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["category", "total", "valid", "no_filtered", "yes_passed",
            "unparsed", "error", "fp_reduction_pct"]

    csv_path = out_dir / "eval.csv"
    md_path = out_dir / "eval.md"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)

    with open(md_path, "w") as f:
        f.write(f"# 오탐감소율 평가 결과\n\n")
        f.write(f"- 일시: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"- 모델: {model}\n")
        f.write(f"- 데이터: {args.path}\n")
        f.write(f"- 샘플: {'전수' if not args.limit else f'카테고리당 {args.limit}장'}\n")
        f.write(f"- 동시 요청: {concurrency}\n")
        f.write(f"- 오탐감소율 = no / (no + yes) × 100  *(unparsed/error 제외)*\n\n")
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

        # 추적/재현용으로 평가에 쓴 프롬프트 전문을 기록.
        f.write("\n## 사용 프롬프트\n\n")
        for cat in images:
            prompt = PROMPTS.get(cat, "(기본 프롬프트)")
            f.write(f"### {cat}\n\n```\n{prompt}\n```\n\n")

    print(f"\n저장: {out_dir}/  ({md_path.name}, {csv_path.name})")


def parse_collect_modes(s: str) -> set:
    """--collect 인자 해석. 'none'/'all'/콤마 구분 부분집합 지원."""
    valid = {"yes", "unparsed", "error"}
    if not s or s == "none":
        return set()
    if s == "all":
        return valid
    return {x.strip() for x in s.split(",") if x.strip() in valid}


def collect_cases(records: list, out_dir: Path, modes: set, symlink: bool) -> int:
    """records 중 parsed 가 modes 에 들면 out_dir/{parsed}/{category}/ 로 모음 + manifest.csv 기록."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for r in records:
        if r["parsed"] not in modes:
            continue
        src = Path(r["image"])
        dest_dir = out_dir / r["parsed"] / r["category"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if symlink:
            dest.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dest)
        manifest_rows.append({
            "category": r["category"], "parsed": r["parsed"],
            "source": str(src), "copied": str(dest),
            "raw_response": (r["raw"] or "").strip().replace("\n", " ")[:500],
        })
    if manifest_rows:
        with (out_dir / "manifest.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["category", "parsed", "source", "copied", "raw_response"])
            w.writeheader()
            w.writerows(manifest_rows)
    return len(manifest_rows)


if __name__ == "__main__":
    main()
