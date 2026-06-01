#!/usr/bin/env python3
"""카테고리 True/False 이미지 라벨링 도구 (웹 기반).

1차 탐지기가 카테고리별로 분류해 둔 썸네일이 진짜 그 카테고리가 맞는지
(True) 아닌지(False) 사람이 빠르게 검수하고, 결과를 카테고리별 true/false
폴더로 재정렬해 복사한다. 외부 의존성 없이 표준 라이브러리 http.server 로
로컬 웹서버를 띄우고 브라우저에서 키보드로 라벨링한다.

전제
    - 데이터가 {input}/by_category/{category}/thumbnail/{date}/*.jpg 구조여야 한다.
    - 경로 기본값은 config.py 에서 가져온다 (그 파일을 고쳐 입출력 위치를 바꿈).
의존성: 표준 라이브러리만 사용.

사용 예
    python src/labeling/label_server.py                       # config.py 기본 경로
    python src/labeling/label_server.py --port 8800
    python src/labeling/label_server.py --category fire,smoke  # 일부 카테고리만
    python src/labeling/label_server.py --output /mnt/work     # 출력 위치 변경
    그 후 브라우저에서 http://localhost:8800 접속.
"""
import argparse
import csv
import json
import shutil
import sys
import threading
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import config

STATIC_DIR = Path(__file__).resolve().parent / "static"


def collect_images(root: Path, categories):
    """카테고리 순서대로 평탄화한 이미지 작업 리스트를 만든다.

    by_category/{cat}/thumbnail/ 아래 jpg 를 rglob 으로 모은다
    (fp_reduction.collect_images 와 같은 스캔 규칙).

    반환: (tasks, counts)
        tasks  = [{"idx", "category", "rel"}]  rel 은 root 기준 상대경로(POSIX)
        counts = {category: 개수}
    """
    by_cat = root / "by_category"
    if not by_cat.is_dir():
        raise SystemExit(f"[에러] 기대한 구조가 아닙니다: '{by_cat}' 없음.")
    cats = categories or sorted(p.name for p in by_cat.iterdir() if p.is_dir())
    tasks, counts = [], {}
    for cat in cats:
        thumb_dir = by_cat / cat / "thumbnail"
        imgs = sorted(thumb_dir.rglob("*.jpg")) if thumb_dir.is_dir() else []
        counts[cat] = len(imgs)
        for img in imgs:
            tasks.append({
                "idx": len(tasks),
                "category": cat,
                "rel": img.relative_to(root).as_posix(),
            })
    return tasks, counts


class LabelStore:
    """labels.json 을 메모리에 들고 매 변경마다 디스크에 저장 (이어하기 원천).

    키는 입력 루트 기준 상대경로. 입력 절대경로가 바뀌어도 라벨이 매칭된다.
    값은 "true" / "false". 라벨 취소 시 키를 제거한다.
    """

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {}
        if path.is_file():
            try:
                self.data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                self.data = {}

    def set(self, rel: str, label):
        with self.lock:
            if label in ("true", "false"):
                self.data[rel] = label
            else:  # None/그 외 → 취소
                self.data.pop(rel, None)
            self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=0))
        tmp.replace(self.path)  # 원자적 교체로 손상 방지

    def counts(self):
        with self.lock:
            t = sum(1 for v in self.data.values() if v == "true")
            f = sum(1 for v in self.data.values() if v == "false")
        return {"true": t, "false": f, "total": t + f}


def export(input_root: Path, store: LabelStore, export_base: Path) -> dict:
    """labels.json 의 라벨을 타임스탬프 폴더 아래 {category}/{true|false}/ 로 원본 복사.

    매 export 마다 {export_base}/export_{YYYYMMDD_HHMMSS}/ 를 새로 만들어 기존
    결과를 덮어쓰지 않는다. fp_reduction.collect_cases 의 복사+manifest 패턴을
    재사용하되 목적지를 카테고리/라벨 2단으로 둔다.
    manifest.csv 컬럼: category,label,source,copied.
    반환: {"copied": n, "skipped": m, "out": str}
    """
    out_dir = export_base / f"export_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, copied, skipped = [], 0, 0
    with store.lock:
        items = list(store.data.items())
    for rel, label in items:
        src = input_root / rel
        if not src.is_file():
            skipped += 1
            continue
        # rel = by_category/{category}/thumbnail/{date}/{name}.jpg → category 추출
        parts = Path(rel).parts
        category = parts[1] if len(parts) > 1 else "unknown"
        dest_dir = out_dir / category / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.exists():
            dest.unlink()
        shutil.copy2(src, dest)
        copied += 1
        rows.append({"category": category, "label": label,
                     "source": str(src), "copied": str(dest)})
    with (out_dir / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "label", "source", "copied"])
        w.writeheader()
        w.writerows(rows)
    return {"copied": copied, "skipped": skipped, "out": str(out_dir)}


def make_handler(input_root: Path, tasks: list, counts: dict, store: LabelStore,
                 export_base: Path):
    """클로저로 상태를 잡은 요청 핸들러 클래스를 만든다."""
    rel_by_idx = [t["rel"] for t in tasks]
    index_html = (STATIC_DIR / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # 요청마다 콘솔 찍는 기본 동작 끔
            pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            route = parsed.path
            if route == "/" or route == "/index.html":
                return self._send(200, index_html, "text/html; charset=utf-8")
            if route == "/api/manifest":
                labels = store.data
                items = [{"idx": t["idx"], "category": t["category"],
                          "rel": t["rel"], "label": labels.get(t["rel"])}
                         for t in tasks]
                return self._send(200, {
                    "total": len(tasks),
                    "counts": counts,
                    "labeled": store.counts(),
                    "items": items,
                })
            if route == "/api/image":
                qs = urllib.parse.parse_qs(parsed.query)
                try:
                    idx = int(qs.get("idx", ["-1"])[0])
                    rel = rel_by_idx[idx]
                except (ValueError, IndexError):
                    return self._send(404, {"error": "bad idx"})
                img = input_root / rel
                if not img.is_file():
                    return self._send(404, {"error": "missing"})
                data = img.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            route = parsed.path
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "bad json"})

            if route == "/api/label":
                try:
                    idx = int(payload["idx"])
                    rel = rel_by_idx[idx]
                except (KeyError, ValueError, IndexError):
                    return self._send(400, {"error": "bad idx"})
                label = payload.get("label")  # "true"/"false"/None(취소)
                store.set(rel, label)
                return self._send(200, {"ok": True, "labeled": store.counts()})

            if route == "/api/export":
                result = export(input_root, store, export_base)
                print(f"[export] {result['copied']}장 복사 → {result['out']}/"
                      + (f"  (원본 누락 {result['skipped']}장 건너뜀)"
                         if result["skipped"] else ""))
                return self._send(200, result)

            return self._send(404, {"error": "not found"})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(config.INPUT_ROOT),
                    help=f"데이터셋 루트 (by_category/ 포함). 기본: {config.INPUT_ROOT}")
    ap.add_argument("--output", default=str(config.OUTPUT_ROOT),
                    help=f"출력 루트. 기본: {config.OUTPUT_ROOT}")
    ap.add_argument("--session", default=config.SESSION_NAME,
                    help=f"세션 폴더 이름. 기본: {config.SESSION_NAME}")
    ap.add_argument("--category", default=None,
                    help="라벨링할 카테고리 (콤마 구분). 생략 시 전체")
    ap.add_argument("--host", default="0.0.0.0", help="바인드 호스트")
    ap.add_argument("--port", type=int, default=8800, help="포트 (기본 8800)")
    args = ap.parse_args()

    input_root = Path(args.input)
    session_dir = Path(args.output) / args.session
    cats = [c.strip() for c in args.category.split(",")] if args.category else None

    tasks, counts = collect_images(input_root, cats)
    if not tasks:
        raise SystemExit("[에러] 라벨링할 이미지가 없습니다.")

    store = LabelStore(session_dir / "labels.json")
    export_base = session_dir / "export"
    handler = make_handler(input_root, tasks, counts, store, export_base)

    print(f"입력      : {input_root}")
    print(f"세션 폴더 : {session_dir}")
    print(f"총 이미지 : {len(tasks)}장 "
          f"({', '.join(f'{c}:{n}' for c, n in counts.items())})")
    done = store.counts()
    if done["total"]:
        print(f"이어하기  : 이미 {done['total']}장 라벨됨 "
              f"(T:{done['true']} F:{done['false']})")
    shown = args.host if args.host not in ("0.0.0.0",) else "localhost"
    print(f"\n서버 시작 → http://{shown}:{args.port}  (Ctrl+C 종료)")
    print("원격/헤드리스면: ssh -L {0}:localhost:{0} <서버>  후 위 주소 접속".format(args.port))
    sys.stdout.flush()  # 로그 리다이렉트(non-tty) 시에도 배너 즉시 표시

    server = ThreadingHTTPServer((args.host, args.port), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료합니다.")
        server.shutdown()


if __name__ == "__main__":
    main()
