#!/usr/bin/env python3
"""카테고리·모델별 yes/no 프롬프트 최적화 오케스트레이터.

describe 진단 결과 모델은 사건을 잘 인식하는데 base yes/no 프롬프트가 그것을 no 로 떨어뜨려
FN(놓친 정탐)이 많다. 이 도구는 프롬프트 후보를 평가·점수화·비교·기록하는 CLI 를 제공한다.
"판단"(오분류·설명을 읽고 다음 프롬프트 문구를 쓰는 것)만 사람/에이전트가 하고, 나머지는 전부
동일한 CLI 라 Claude 가 하든 인간이 하든 똑같이 재현된다 (docs/prompt_opt.md 런북 참고).

평가 엔진은 labeled_eval.py 를 subprocess 로 재사용한다. 떠 있는 vLLM 서버를 그대로 쓰므로
반복 시 도커 재기동이 없다 (한 모델 띄운 채 후보만 교체).

목표 지표:
    fp_reduction = TN/(TN+FP)   (오탐 잘 거름)
    retention    = TP/(TP+FN)   (정탐 유지 = 정탐유지율)
    objective    = retention            (fp_reduction >= floor)
                 = retention - (floor-fp_reduction)*W  (하한 미달 패널티)
    best = objective 최대 (사실상 하한 통과 후보 중 retention 최대)

서브커맨드:
    eval    : 후보(또는 --baseline) 1개를 dev(기본) 또는 --full 로 평가 → vNN.json + vNN.txt
    report  : (model,category) 의 버전×지표 표 + 현재 best
    record  : 승자 버전을 --full 검증 후 optimized_prompts.json + best + report.md 기록

사용 예 (단일 27B 띄운 상태)
    python src/evaluation/prompt_opt.py eval --model Qwen/Qwen3.5-27B --category falldown --baseline
    python src/evaluation/prompt_opt.py eval --model Qwen/Qwen3.5-27B --category falldown --prompt-file cand.txt
    python src/evaluation/prompt_opt.py report --model Qwen/Qwen3.5-27B --category falldown
    python src/evaluation/prompt_opt.py record --model Qwen/Qwen3.5-27B --category falldown --version v03 --notes-file why.md
"""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from prompts import PROMPTS, get_prompt

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
OPT_DIR = RESULTS_DIR / "prompt_opt"
LABELED_EVAL = REPO_ROOT / "src" / "evaluation" / "labeled_eval.py"
OPTIMIZED_JSON = Path(__file__).with_name("optimized_prompts.json")

DEFAULT_FLOOR = 0.90
PENALTY_W = 10.0
SAMPLE_K = 8          # 리포트에 붙일 FN/FP 샘플 수
DEFAULT_MAX_NEG = 200


def model_dirname(model: str) -> str:
    return model.split("/")[-1]


def cat_dir(model: str, category: str) -> Path:
    return OPT_DIR / model_dirname(model) / category


def find_latest_sweep() -> Path | None:
    cands = sorted(p for p in RESULTS_DIR.glob("sweep_labeled_*") if p.is_dir())
    return cands[-1] if cands else None


def load_describe_index(sweep_dir: Path, model: str) -> dict:
    """{이미지 basename: {desc_en, desc_ko}} — 해당 모델 describe.jsonl 에서."""
    out = {}
    if not sweep_dir:
        return out
    jsonl = sweep_dir / model_dirname(model) / "describe" / "describe.jsonl"
    if not jsonl.is_file():
        return out
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[Path(r.get("source", "")).name] = {
            "desc_en": r.get("description", ""), "desc_ko": r.get("description_ko", "")}
    return out


def next_version(cdir: Path) -> str:
    cdir.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in cdir.glob("v*.json"):
        s = p.stem.lstrip("v")
        if s.isdigit():
            nums.append(int(s))
    return f"v{(max(nums) + 1) if nums else 1:02d}"


def run_labeled_eval(model: str, category: str, prompt_file: Path, run_dir: Path,
                     full: bool, max_neg: int, url: str, seed: int) -> tuple:
    """labeled_eval.py 를 subprocess 로 실행 → (category row dict, manifest rows)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(LABELED_EVAL),
           "--category", category, "--model", model, "--url", url,
           "--prompt-file", str(prompt_file), "--out", str(run_dir),
           "--collect", "fp,fn", "--symlink", "--seed", str(seed), "--max-tokens", "1"]
    if not full:
        cmd += ["--max-neg", str(max_neg)]
    rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    if rc != 0:
        raise SystemExit(f"[에러] labeled_eval 실패 (rc={rc})")
    # eval.csv 에서 해당 카테고리 행
    row = None
    with (run_dir / "eval.csv").open() as f:
        for r in csv.DictReader(f):
            if r["category"] == category:
                row = r
                break
    if row is None:
        raise SystemExit(f"[에러] eval.csv 에 {category} 행이 없습니다: {run_dir}")
    manifest = []
    mpath = run_dir / "manifest.csv"
    if mpath.is_file():
        with mpath.open() as f:
            manifest = list(csv.DictReader(f))
    return row, manifest


def score(row: dict, floor: float) -> dict:
    tp, fn = int(row["tp"]), int(row["fn"])
    tn, fp = int(row["tn"]), int(row["fp"])
    pos, neg = tp + fn, tn + fp
    fp_red = tn / neg if neg else float("nan")
    ret = tp / pos if pos else float("nan")
    passes = (fp_red >= floor) if fp_red == fp_red else False
    if ret != ret:
        obj = float("nan")
    elif passes:
        obj = ret
    else:
        obj = ret - (floor - fp_red) * PENALTY_W
    return {"counts": {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
                       "unparsed": int(row["unparsed"]), "error": int(row["error"])},
            "positives": pos, "negatives": neg,
            "fp_reduction": None if fp_red != fp_red else round(fp_red, 4),
            "retention": None if ret != ret else round(ret, 4),
            "objective": None if obj != obj else round(obj, 4),
            "passes_floor": passes, "floor": floor}


def build_samples(manifest: list, desc_idx: dict, k: int) -> dict:
    out = {"fn": [], "fp": []}
    for r in manifest:
        kind = r.get("kind")
        if kind not in ("fn", "fp") or len(out[kind]) >= k:
            continue
        src = r.get("source", "")
        d = desc_idx.get(Path(src).name, {})
        out[kind].append({
            "image": src, "base_verdict": r.get("parsed", ""),
            "raw": r.get("raw_response", ""),
            "desc_ko": d.get("desc_ko", ""), "desc_en": d.get("desc_en", "")})
    return out


def fmt_pct(x):
    return "N/A" if x is None else f"{x*100:.1f}%"


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
def cmd_eval(args):
    cdir = cat_dir(args.model, args.category)
    cdir.mkdir(parents=True, exist_ok=True)

    # 프롬프트 결정
    if args.baseline:
        prompt_text = PROMPTS.get(args.category, "")
        tag = "baseline"
    elif args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text().strip()
        tag = args.tag or ""
    else:
        raise SystemExit("[에러] --baseline 또는 --prompt-file 중 하나가 필요합니다.")

    version = "v00" if args.baseline else next_version(cdir)
    txt_path = cdir / f"{version}.txt"
    txt_path.write_text(prompt_text)

    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else find_latest_sweep()
    desc_idx = load_describe_index(sweep_dir, args.model)

    run_dir = cdir / f"_run_{version}{'_full' if args.full else ''}"
    row, manifest = run_labeled_eval(args.model, args.category, txt_path, run_dir,
                                     args.full, args.max_neg, args.url, args.seed)
    sc = score(row, args.floor)
    rec = {
        "model": args.model, "category": args.category, "version": version, "tag": tag,
        "set": "full" if args.full else "dev", "max_neg": (0 if args.full else args.max_neg),
        "prompt_file": str(txt_path), "prompt": prompt_text,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **sc,
        "samples": build_samples(manifest, desc_idx, SAMPLE_K),
    }
    out_json = cdir / f"{version}{'_full' if args.full else ''}.json"
    out_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2))

    print(f"\n[{version}] {args.model} / {args.category} / {'full' if args.full else 'dev'}")
    print(f"  fp_reduction={fmt_pct(sc['fp_reduction'])}  retention={fmt_pct(sc['retention'])}  "
          f"objective={sc['objective']}  floor통과={sc['passes_floor']}")
    print(f"  counts tp={sc['counts']['tp']} fn={sc['counts']['fn']} "
          f"tn={sc['counts']['tn']} fp={sc['counts']['fp']}")
    print(f"  저장: {out_json}")
    nfn, nfp = len(rec['samples']['fn']), len(rec['samples']['fp'])
    print(f"  오분류 샘플: FN {nfn}건, FP {nfp}건 (describe 설명 포함) — JSON 의 samples 참고")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def load_versions(cdir: Path, full: bool) -> list:
    recs = []
    for p in sorted(cdir.glob("v*.json")):
        is_full = p.stem.endswith("_full")
        if is_full != full:
            continue
        try:
            recs.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            pass
    return recs


def pick_best(recs: list):
    cands = [r for r in recs if r.get("objective") is not None]
    if not cands:
        return None
    return max(cands, key=lambda r: (r["objective"], r.get("fp_reduction") or 0))


def cmd_report(args):
    cdir = cat_dir(args.model, args.category)
    recs = load_versions(cdir, full=False)
    if not recs:
        raise SystemExit(f"[에러] dev 평가 결과가 없습니다: {cdir}")
    print(f"\n=== {args.model} / {args.category} (dev) ===")
    print(f"{'ver':>6} {'tag':<10} {'fp_red':>8} {'reten':>8} {'obj':>7} {'floor':>6}  counts")
    for r in recs:
        c = r["counts"]
        print(f"{r['version']:>6} {r.get('tag',''):<10} {fmt_pct(r['fp_reduction']):>8} "
              f"{fmt_pct(r['retention']):>8} {str(r['objective']):>7} "
              f"{'OK' if r['passes_floor'] else 'X':>6}  "
              f"tp{c['tp']} fn{c['fn']} tn{c['tn']} fp{c['fp']}")
    best = pick_best(recs)
    if best:
        print(f"\n현재 best: {best['version']} (objective={best['objective']}, "
              f"retention={fmt_pct(best['retention'])}, fp_reduction={fmt_pct(best['fp_reduction'])})")


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------
def ensure_full(args, version: str) -> dict:
    """해당 버전의 full 검증 결과를 보장(없으면 평가)하고 반환."""
    cdir = cat_dir(args.model, args.category)
    full_json = cdir / f"{version}_full.json"
    if full_json.is_file():
        return json.loads(full_json.read_text())
    # full 평가 실행 (해당 버전 txt 사용)
    txt = cdir / f"{version}.txt"
    if not txt.is_file():
        raise SystemExit(f"[에러] {txt} 없음")
    ns = argparse.Namespace(model=args.model, category=args.category, baseline=False,
                            prompt_file=str(txt), tag="", full=True, max_neg=args.max_neg,
                            url=args.url, seed=args.seed, floor=args.floor,
                            sweep_dir=args.sweep_dir)
    # cmd_eval 은 next_version 을 새로 만들어버리므로, 여기선 직접 평가 경로를 재현
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else find_latest_sweep()
    desc_idx = load_describe_index(sweep_dir, args.model)
    run_dir = cdir / f"_run_{version}_full"
    row, manifest = run_labeled_eval(args.model, args.category, txt, run_dir,
                                     True, args.max_neg, args.url, args.seed)
    sc = score(row, args.floor)
    rec = {"model": args.model, "category": args.category, "version": version,
           "set": "full", "prompt": txt.read_text().strip(),
           "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **sc,
           "samples": build_samples(manifest, desc_idx, SAMPLE_K)}
    full_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    return rec


def cmd_record(args):
    cdir = cat_dir(args.model, args.category)
    # 최종(승자) full 검증
    final = ensure_full(args, args.version)
    # baseline full 검증 (v00)
    base_txt = cdir / "v00.txt"
    if not base_txt.is_file():
        base_txt.write_text(PROMPTS.get(args.category, ""))
    baseline = ensure_full(args, "v00")

    # optimized_prompts.json 갱신
    reg = {}
    if OPTIMIZED_JSON.is_file():
        try:
            reg = json.loads(OPTIMIZED_JSON.read_text())
        except json.JSONDecodeError:
            reg = {}
    reg.setdefault(args.model, {})[args.category] = final["prompt"]
    OPTIMIZED_JSON.write_text(json.dumps(reg, ensure_ascii=False, indent=2))

    # best 파일
    (cdir / "best.txt").write_text(final["prompt"])
    (cdir / "best.json").write_text(json.dumps(final, ensure_ascii=False, indent=2))

    # report.md
    notes = ""
    if args.notes_file and Path(args.notes_file).is_file():
        notes = Path(args.notes_file).read_text().strip()
    dev_recs = load_versions(cdir, full=False)

    def metr(r):
        return (f"fp_reduction={fmt_pct(r['fp_reduction'])}, retention={fmt_pct(r['retention'])} "
                f"(tp{r['counts']['tp']} fn{r['counts']['fn']} tn{r['counts']['tn']} fp{r['counts']['fp']})")

    d_ret = (final['retention'] or 0) - (baseline['retention'] or 0)
    d_fp = (final['fp_reduction'] or 0) - (baseline['fp_reduction'] or 0)
    rp = cdir / "report.md"
    with rp.open("w") as f:
        f.write(f"# 프롬프트 최적화 리포트 — {args.model} / {args.category}\n\n")
        f.write(f"- 일시: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"- 검증: 전수(full). 선택 버전 {args.version}.\n\n")
        f.write("## 성능 (전수 검증)\n\n")
        f.write("| | fp_reduction | retention | tp | fn | tn | fp |\n|---|---|---|---|---|---|---|\n")
        for nm, r in (("baseline(v00)", baseline), (f"최종({args.version})", final)):
            c = r["counts"]
            f.write(f"| {nm} | {fmt_pct(r['fp_reduction'])} | {fmt_pct(r['retention'])} | "
                    f"{c['tp']} | {c['fn']} | {c['tn']} | {c['fp']} |\n")
        f.write(f"| **delta** | **{d_fp*100:+.1f}p** | **{d_ret*100:+.1f}p** | | | | |\n\n")
        f.write("## 원인 분석\n\n")
        f.write((notes or "(여기에 베이스라인이 틀린 원인 — 오분류+describe 설명 근거 — 를 작성)") + "\n\n")
        f.write("## 베이스라인 프롬프트 (v00)\n\n```\n" + baseline["prompt"] + "\n```\n\n")
        f.write(f"## 최종 선택 프롬프트 ({args.version})\n\n```\n" + final["prompt"] + "\n```\n\n")
        f.write("## 반복 이력 (dev)\n\n")
        f.write("| ver | tag | fp_reduction | retention | objective |\n|---|---|---|---|---|\n")
        for r in dev_recs:
            f.write(f"| {r['version']} | {r.get('tag','')} | {fmt_pct(r['fp_reduction'])} | "
                    f"{fmt_pct(r['retention'])} | {r['objective']} |\n")
        if final["positives"] <= 3:
            f.write(f"\n> 캐비엇: positive {final['positives']}건 — retention 통계 신뢰도 낮음.\n")

    print(f"기록 완료:")
    print(f"  optimized_prompts.json ← [{args.model}][{args.category}]")
    print(f"  {cdir}/best.txt, best.json, report.md")
    print(f"  delta: fp_reduction {d_fp*100:+.1f}p, retention {d_ret*100:+.1f}p")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", required=True)
        p.add_argument("--category", required=True)
        p.add_argument("--url", default="http://localhost:8000")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--floor", type=float, default=DEFAULT_FLOOR,
                       help=f"오탐감소율 하한 (기본 {DEFAULT_FLOOR})")
        p.add_argument("--max-neg", type=int, default=DEFAULT_MAX_NEG,
                       help=f"dev negative 캡 (기본 {DEFAULT_MAX_NEG})")
        p.add_argument("--sweep-dir", default=None,
                       help="describe.jsonl 가 있는 sweep 폴더 (생략 시 최신 sweep_labeled_*)")

    pe = sub.add_parser("eval", help="후보/베이스라인 1개 평가")
    common(pe)
    pe.add_argument("--prompt-file", default=None, help="후보 프롬프트 파일")
    pe.add_argument("--baseline", action="store_true", help="현행 PROMPTS 를 v00 으로 평가")
    pe.add_argument("--tag", default=None, help="버전 메모(선택)")
    pe.add_argument("--full", action="store_true", help="dev 대신 전수 평가")
    pe.set_defaults(func=cmd_eval)

    pr = sub.add_parser("report", help="버전×지표 표")
    common(pr)
    pr.set_defaults(func=cmd_report)

    prc = sub.add_parser("record", help="승자 버전 전수검증+기록")
    common(prc)
    prc.add_argument("--version", required=True, help="승자 버전 (예: v03)")
    prc.add_argument("--notes-file", default=None, help="원인 분석 markdown (리포트에 삽입)")
    prc.set_defaults(func=cmd_record)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
