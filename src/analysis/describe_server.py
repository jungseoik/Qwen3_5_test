#!/usr/bin/env python3
"""오분류 케이스 description 분석 도구 (웹 기반).

describe_eval.py / describe_sweep.py 가 만든 결과(`<Model>/describe/describe.jsonl`)를 읽어,
모델별 · FN/FP별로 오분류 케이스를 사람이 빠르게 넘겨보며 "모델이 이미지를 어떻게 이해했는가"
를 base(yes/no) 프롬프트·정답·모델의 한국어 description 과 나란히 비교 분석한다.

외부 의존성 없이 표준 라이브러리 http.server 로 로컬 웹서버를 띄우고 브라우저에서 키보드로 넘긴다.

전제
    - --path 는 labeled 평가 결과 폴더이며, 분석할 모델 폴더에 describe/describe.jsonl 이 있어야 한다.
      (먼저 describe_eval.py 또는 describe_sweep.py 를 돌려 생성)
의존성: 표준 라이브러리만 사용.

사용 예
    python src/analysis/describe_server.py                                   # 최신 sweep_labeled_* 자동
    python src/analysis/describe_server.py --path results/sweep_labeled_20260529_095220
    python src/analysis/describe_server.py --port 8801
    그 후 브라우저에서 http://localhost:8801 접속.
"""
import argparse
import json
import sys
import urllib.parse
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import config

# base/describe 프롬프트를 evaluation/prompts.py 에서 그대로 가져온다 (단일 진실).
sys.path.insert(0, str(config.REPO_ROOT / "src" / "evaluation"))
from prompts import get_prompt, get_describe_prompt  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
KIND_LABEL = {"fn": "FN (놓친 정탐)", "fp": "FP (못 거른 오탐)",
              "tp": "TP (정탐 유지)", "tn": "TN (오탐 제거)"}


def model_dirname(name: str) -> str:
    """표시용 모델명 (폴더명 그대로)."""
    return name


def load_results(root: Path) -> dict:
    """root 아래 각 모델 폴더의 describe/describe.jsonl 을 읽는다.

    반환: {model_name: {"cases": {id: case}, "order": [id,...]}}
    case = {id, kind, category, label, base_verdict, source, description}
    """
    out = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        jsonl = sub / "describe" / "describe.jsonl"
        if not jsonl.is_file():
            continue
        cases, order = {}, []
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = rec.get("id")
                if not cid:
                    continue
                cases[cid] = rec
                order.append(cid)
        if cases:
            out[sub.name] = {"cases": cases, "order": order}
    return out


def build_index(results: dict) -> dict:
    """프런트 진입용 메타 — 모델 목록·카테고리·kind 집계·프롬프트."""
    models = []
    all_cats = set()
    for name, blob in results.items():
        kc = defaultdict(lambda: defaultdict(int))  # kind -> cat -> n
        cats = set()
        for cid in blob["order"]:
            c = blob["cases"][cid]
            kc[c.get("kind", "")][c.get("category", "")] += 1
            cats.add(c.get("category", ""))
            all_cats.add(c.get("category", ""))
        counts = {k: dict(v) for k, v in kc.items()}
        models.append({
            "name": name,
            "categories": sorted(cats),
            "counts": counts,
            "total": sum(sum(v.values()) for v in counts.values()),
        })
    cats_sorted = sorted(all_cats)
    return {
        "models": models,
        "categories": cats_sorted,
        "kind_label": KIND_LABEL,
        "base_prompts": {c: get_prompt(c) for c in cats_sorted},
        "describe_prompts": {c: get_describe_prompt(c) for c in cats_sorted},
    }


def make_handler(root: Path, results: dict, index_meta: dict):
    index_html = (STATIC_DIR / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
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

        def _case(self, qs):
            model = qs.get("model", [""])[0]
            cid = qs.get("id", [""])[0]
            blob = results.get(model)
            if not blob or cid not in blob["cases"]:
                return None
            return blob["cases"][cid]

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            route, qs = parsed.path, urllib.parse.parse_qs(parsed.query)

            if route in ("/", "/index.html"):
                return self._send(200, index_html, "text/html; charset=utf-8")

            if route == "/api/index":
                return self._send(200, {**index_meta, "path": str(root)})

            if route == "/api/cases":
                model = qs.get("model", [""])[0]
                kind = qs.get("kind", [""])[0]
                cat = qs.get("category", [""])[0]
                blob = results.get(model)
                if not blob:
                    return self._send(404, {"error": "unknown model"})
                items = []
                for cid in blob["order"]:
                    c = blob["cases"][cid]
                    if kind and c.get("kind") != kind:
                        continue
                    if cat and cat != "__all__" and c.get("category") != cat:
                        continue
                    items.append({
                        "id": c["id"], "category": c.get("category", ""),
                        "kind": c.get("kind", ""), "label": c.get("label", ""),
                        "base_verdict": c.get("base_verdict", ""),
                    })
                return self._send(200, {"model": model, "items": items})

            if route == "/api/case":
                c = self._case(qs)
                if not c:
                    return self._send(404, {"error": "not found"})
                cat = c.get("category", "")
                return self._send(200, {
                    **c,
                    "base_prompt": get_prompt(cat),
                    "describe_prompt": get_describe_prompt(cat),
                })

            if route == "/img":
                c = self._case(qs)
                if not c:
                    return self._send(404, {"error": "not found"})
                # source 는 우리 manifest 가 만든 신뢰된 절대경로. 존재 확인 후 서빙.
                img = Path(c.get("source", ""))
                if not img.is_file():
                    return self._send(404, {"error": "missing image"})
                data = img.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return

            return self._send(404, {"error": "not found"})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=None,
                    help="labeled 평가 결과 폴더 (각 <Model>/describe/describe.jsonl 보유). "
                         "생략 시 results/ 아래 최신 sweep_labeled_* 자동")
    ap.add_argument("--host", default="0.0.0.0", help="바인드 호스트")
    ap.add_argument("--port", type=int, default=config.PORT, help=f"포트 (기본 {config.PORT})")
    args = ap.parse_args()

    if args.path:
        root = Path(args.path)
    else:
        root = config.DEFAULT_PATH or config.find_latest_sweep()
        if not root:
            raise SystemExit(
                f"[에러] {config.RESULTS_DIR} 아래 sweep_labeled_* 폴더가 없습니다.\n"
                f"  --path 로 평가 결과 폴더를 지정하세요.")
        root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"[에러] 폴더가 아닙니다: {root}")

    results = load_results(root)
    if not results:
        raise SystemExit(
            f"[에러] {root} 아래 어떤 모델에도 describe/describe.jsonl 이 없습니다.\n"
            f"  먼저 describe_eval.py 또는 describe_sweep.py 로 description 을 생성하세요.")

    index_meta = build_index(results)
    handler = make_handler(root, results, index_meta)

    print(f"분석 폴더 : {root}")
    print(f"모델      : {len(results)}개")
    for m in index_meta["models"]:
        kinds = ", ".join(f"{k}:{sum(v.values())}" for k, v in m["counts"].items())
        print(f"  - {m['name']:<20} {m['total']}건  [{kinds}]")
    shown = args.host if args.host not in ("0.0.0.0",) else "localhost"
    print(f"\n서버 시작 → http://{shown}:{args.port}  (Ctrl+C 종료)")
    print("원격/헤드리스면: ssh -L {0}:localhost:{0} <서버>  후 위 주소 접속".format(args.port))
    sys.stdout.flush()

    server = ThreadingHTTPServer((args.host, args.port), handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료합니다.")
        server.shutdown()


if __name__ == "__main__":
    main()
