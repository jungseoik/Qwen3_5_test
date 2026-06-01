#!/usr/bin/env python3
"""[일회성] 기존 describe 결과의 한국어 번역을 Gemini 로 다시 만든다.

초기 describe 결과는 번역을 vLLM 자기번역으로 했더니 작은 모델에서 한국어가 어색·깨짐.
이 스크립트는 이미 있는 영어 description(`description`)은 그대로 두고, `description_ko` 만
Gemini 로 재생성해 describe.jsonl / describe.csv 를 덮어쓴다. (영어 묘사는 재생성 안 함 → 빠름)

일회성 보정용이라 확장성은 고려하지 않는다. 원본 jsonl 은 describe.jsonl.bak 으로 1회 백업한다.

사용 예
    python src/evaluation/retranslate_gemini.py                              # 최신 sweep 전 모델
    python src/evaluation/retranslate_gemini.py --from results/sweep_labeled_20260529_095220
    python src/evaluation/retranslate_gemini.py --models 0.8B,2B --concurrency 8
"""
import argparse
import csv
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fp_reduction import Progress
from translate import translate_batch_safe, get_key_and_model

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
CSV_COLS = ["id", "kind", "category", "label", "base_verdict", "source",
            "description", "description_ko"]
TEXTCOLS = {"description", "description_ko"}


def find_latest_sweep():
    cands = sorted(p for p in RESULTS_DIR.glob("sweep_labeled_*") if p.is_dir())
    return cands[-1] if cands else None


def model_dirs(sweep_dir: Path, models_filter):
    """describe/describe.jsonl 이 있는 모델 폴더 목록. models_filter 는 부분 문자열 매칭."""
    out = []
    for sub in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        if not (sub / "describe" / "describe.jsonl").is_file():
            continue
        if models_filter and not any(f in sub.name for f in models_filter):
            continue
        out.append(sub)
    return out


def load_jsonl(path: Path):
    recs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def write_jsonl(path: Path, recs):
    with path.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, recs):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in recs:
            w.writerow({k: (r.get(k, "") or "").replace("\n", " ") if k in TEXTCOLS
                        else r.get(k, "") for k in CSV_COLS})


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def retranslate_model(model_dir: Path, key: str, gmodel: str,
                      batch_size: int, batch_concurrency: int) -> dict:
    """한 모델의 description_ko 를 배치 번역으로 재생성하고 jsonl/csv 덮어씀.

    targets 를 batch_size 묶음으로 나눠 batch_concurrency 개씩 동시에 Gemini 에 보낸다.
    묶음이 어긋나거나 잘리면 translate_batch_safe 가 분할 재시도한다.
    """
    jsonl = model_dir / "describe" / "describe.jsonl"
    recs = load_jsonl(jsonl)

    # 번역 대상: 영어 description 이 정상인 레코드만
    targets = [r for r in recs if (r.get("description", "") or "").strip()
               and not r["description"].startswith("<error")]
    batches = list(_chunks(targets, batch_size))
    print(f"\n[{model_dir.name}] {len(recs)}건 중 번역 대상 {len(targets)}건 "
          f"→ {len(batches)}묶음(묶음당 {batch_size}, 동시 {batch_concurrency})", flush=True)

    ok = err = 0
    progress = Progress(len(batches)) if batches else None

    def work(batch):
        texts = [r["description"] for r in batch]
        return batch, translate_batch_safe(texts, key, gmodel)

    if batches:
        with ThreadPoolExecutor(max_workers=batch_concurrency) as ex:
            futs = [ex.submit(work, b) for b in batches]
            for i, fut in enumerate(as_completed(futs), 1):
                batch, kos = fut.result()
                for rec, ko in zip(batch, kos):
                    rec["description_ko"] = ko
                    if (ko or "").startswith("<error"):
                        err += 1
                    else:
                        ok += 1
                progress.update(i)

    # 1회 백업 후 덮어쓰기
    bak = jsonl.with_suffix(".jsonl.bak")
    if not bak.exists():
        shutil.copy2(jsonl, bak)
    write_jsonl(jsonl, recs)
    write_csv(model_dir / "describe" / "describe.csv", recs)
    print(f"[{model_dir.name}] 완료: 성공 {ok}, 실패 {err}  → {jsonl}")
    return {"ok": ok, "err": err}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="from_dir", default=None,
                    help="sweep_labeled 폴더 (생략 시 최신 자동)")
    ap.add_argument("--models", default=None, help="부분 매칭 콤마 필터 (예: 0.8B,2B)")
    ap.add_argument("--batch-size", type=int, default=50,
                    help="한 API 호출에 묶을 번역 개수 (기본 50). 어긋나면 자동 분할 재시도")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="동시에 보낼 묶음 수 (기본 4)")
    ap.add_argument("--gemini-model", default=None, help="Gemini 모델 (생략 시 .env)")
    args = ap.parse_args()

    sweep_dir = Path(args.from_dir) if args.from_dir else find_latest_sweep()
    if not sweep_dir or not sweep_dir.is_dir():
        raise SystemExit(f"[에러] sweep 폴더를 찾을 수 없습니다: {sweep_dir}")

    key, gmodel = get_key_and_model(None, args.gemini_model)
    if not key:
        raise SystemExit("[에러] GEMINI_API_KEY 가 없습니다 (.env 확인).")

    models_filter = [m.strip() for m in args.models.split(",")] if args.models else None
    dirs = model_dirs(sweep_dir, models_filter)
    if not dirs:
        raise SystemExit(f"[에러] describe/describe.jsonl 이 있는 모델이 없습니다: {sweep_dir}")

    print(f"sweep 폴더 : {sweep_dir}")
    print(f"대상 모델  : {[d.name for d in dirs]}")
    print(f"Gemini     : {gmodel}  (묶음 {args.batch_size} × 동시 {args.concurrency})")

    total = {"ok": 0, "err": 0}
    for d in dirs:
        c = retranslate_model(d, key, gmodel, args.batch_size, args.concurrency)
        total["ok"] += c["ok"]; total["err"] += c["err"]

    print(f"\n=== 재번역 종료 === 성공 {total['ok']}, 실패 {total['err']}")
    print("UI 에 반영하려면 describe_server 를 재시작하세요 (메모리 캐시).")


if __name__ == "__main__":
    main()
