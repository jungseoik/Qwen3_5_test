#!/usr/bin/env python3
"""오탐 로그 데이터셋 통계 출력.

지정한 경로가 아래 계층 구조를 따른다고 가정하고, 카테고리별 / 날짜별
오탐 로그(썸네일 jpg, 비디오 mp4) 개수를 집계해 표로 출력한다.

기대하는 디렉토리 구조:
    {root}/
        by_category/
            {category}/                 (예: falldown, fire, smoke, violence)
                thumbnail/
                    {YYYYMMDD}/*.jpg
                video/
                    {YYYYMMDD}/*.mp4

의존성: 표준 라이브러리만 사용.

사용 예:
    python src/dataset/stats.py
    python src/dataset/stats.py --path nas/data/khonkaen
    python src/dataset/stats.py --path /data/site_a --summary-only
"""
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "nas/data/khonkaen"
KINDS = ("thumbnail", "video")  # (jpg, mp4)


def collect(root: Path):
    """{category: {date: {"thumbnail": n, "video": n}}} 형태로 집계."""
    by_cat_dir = root / "by_category"
    if not by_cat_dir.is_dir():
        raise SystemExit(f"[에러] 기대한 구조가 아닙니다: '{by_cat_dir}' 가 없습니다.\n"
                         f"  '{root}/by_category/{{category}}/{{thumbnail,video}}/{{date}}/' 구조를 기대합니다.")

    stats = {}
    for cat_dir in sorted(p for p in by_cat_dir.iterdir() if p.is_dir()):
        per_date = {}
        for kind in KINDS:
            kind_dir = cat_dir / kind
            if not kind_dir.is_dir():
                continue
            for date_dir in kind_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                n = sum(1 for f in date_dir.iterdir() if f.is_file())
                per_date.setdefault(date_dir.name, {"thumbnail": 0, "video": 0})[kind] = n
        stats[cat_dir.name] = per_date
    return stats


def print_category_detail(category: str, per_date: dict):
    print(f"\n[{category}]")
    print(f"{'날짜':>10} {'thumbnail':>10} {'video':>8} {'mismatch':>9}")
    print("-" * 41)
    t_sum = v_sum = 0
    for date in sorted(per_date):
        t, v = per_date[date]["thumbnail"], per_date[date]["video"]
        t_sum += t
        v_sum += v
        flag = "" if t == v else "  <-- 불일치"
        print(f"{date:>10} {t:>10} {v:>8} {flag:>9}")
    print("-" * 41)
    print(f"{'소계':>10} {t_sum:>10} {v_sum:>8}   ({len(per_date)}일)")


def print_date_summary(stats: dict):
    """날짜 중심: 날짜 x 카테고리 매트릭스 (썸네일 기준 개수)."""
    categories = list(stats)
    dates = sorted({d for per_date in stats.values() for d in per_date})

    print("\n=== 날짜별 요약 (썸네일 기준) ===")
    header = f"{'날짜':>10}" + "".join(f"{c:>10}" for c in categories) + f"{'합계':>8}"
    print(header)
    print("-" * (10 + 10 * len(categories) + 8))

    col_totals = {c: 0 for c in categories}
    for date in dates:
        row_total = 0
        cells = ""
        for c in categories:
            n = stats[c].get(date, {}).get("thumbnail", 0)
            col_totals[c] += n
            row_total += n
            cells += f"{n:>10}"
        print(f"{date:>10}{cells}{row_total:>8}")

    print("-" * (10 + 10 * len(categories) + 8))
    grand = sum(col_totals.values())
    cells = "".join(f"{col_totals[c]:>10}" for c in categories)
    print(f"{'합계':>10}{cells}{grand:>8}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=str(DEFAULT_PATH),
                    help=f"데이터셋 루트 경로 (기본: {DEFAULT_PATH})")
    ap.add_argument("--summary-only", action="store_true",
                    help="카테고리별 날짜 상세 표를 생략하고 전체 요약만 출력")
    args = ap.parse_args()

    root = Path(args.path)
    stats = collect(root)
    if not stats:
        raise SystemExit(f"[에러] by_category 아래에 카테고리 폴더가 없습니다: {root}/by_category")

    all_dates = {d for per_date in stats.values() for d in per_date}
    print(f"데이터 경로 : {root}")
    print(f"카테고리   : {len(stats)}개 ({', '.join(stats)})")
    if all_dates:
        print(f"날짜 범위  : {min(all_dates)} ~ {max(all_dates)}")

    if not args.summary_only:
        for category, per_date in stats.items():
            print_category_detail(category, per_date)

    # 전체 요약
    print("\n=== 전체 요약 ===")
    print(f"{'카테고리':>10} {'thumbnail':>10} {'video':>8} {'날짜수':>6}")
    print("-" * 38)
    gt = gv = 0
    for category, per_date in stats.items():
        t = sum(d["thumbnail"] for d in per_date.values())
        v = sum(d["video"] for d in per_date.values())
        gt += t
        gv += v
        print(f"{category:>10} {t:>10} {v:>8} {len(per_date):>6}")
    print("-" * 38)
    print(f"{'합계':>10} {gt:>10} {gv:>8}")
    print(f"\n총 오탐 로그(썸네일 기준): {gt}개,  비디오: {gv}개")

    # 날짜 중심 요약 (카테고리 요약과 함께 항상 출력)
    print_date_summary(stats)


if __name__ == "__main__":
    main()
