#!/usr/bin/env python3
"""동시 요청 레이턴시 벤치마크.

띄워둔 vLLM 서버에 이미지 VQA 요청(출력 1토큰, yes/no)을 동시성 수준별로
보내 처리 시간을 측정하고 mean / p50 / p95 / p99 / throughput 을 표로 출력한다.

요청마다 assets/test/{해상도}/ 안의 색깔 이미지(red/yellow/...)를 랜덤으로 골라
"Is this color yellow?" 같은 yes/no 질문과 함께 보낸다. 실제 운영처럼 매번
다른 이미지가 들어오는 상황을 재현한다.

전제
    - docker compose 로 qwen-vllm 서버가 떠 있어야 한다 (기본 http://localhost:8000).
    - 테스트 이미지가 assets/test/{720p,1080p}/*.jpg 에 있어야 한다.
의존성
    - 표준 라이브러리만 사용 (별도 pip 설치 불필요).

사용 예
    python src/tests/bench_concurrency.py
    python src/tests/bench_concurrency.py --resolution 1080p --rounds 8
    python src/tests/bench_concurrency.py --levels 1,2,4,8,16,32,64,128
"""
import argparse
import base64
import json
import math
import random
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
IMAGE_DIRS = {
    "720p": REPO_ROOT / "assets/test/720p",
    "1080p": REPO_ROOT / "assets/test/1080p",
}


def load_env(path: Path) -> dict:
    """간이 .env 파서 — KEY=VALUE 라인만 추출."""
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


def default_levels(max_seqs: int) -> list:
    """1부터 MAX_NUM_SEQS 까지 2의 거듭제곱 + 상한값. 예) 128 → [1,2,4,8,16,32,64,128]."""
    out, v = [], 1
    while v < max_seqs:
        out.append(v)
        v *= 2
    out.append(max_seqs)
    return out


def detect_model(url: str) -> str:
    with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as r:
        return json.load(r)["data"][0]["id"]


def build_payload(model: str, img_b64: str, prompt: str) -> bytes:
    payload = {
        "model": model,
        "max_tokens": 1,          # yes/no 1토큰 고정 워크로드
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]}],
    }
    return json.dumps(payload).encode()


def one_request(url: str, payloads: list):
    """payloads 중 하나를 랜덤으로 골라 1회 요청. (latency_sec, error_or_None) 반환."""
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=random.choice(payloads),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            r.read()
        return time.perf_counter() - t0, None
    except Exception as e:  # 연결/HTTP/타임아웃 등 전부 에러로 집계
        return time.perf_counter() - t0, str(e)


def percentile(sorted_vals, p: float) -> float:
    """선형 보간 백분위수. sorted_vals 는 오름차순 정렬된 리스트."""
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def run_level(url: str, payloads: list, concurrency: int, n: int):
    """동시성 concurrency 로 총 n 개 요청을 보내고 (latencies, errors, wall_sec) 반환."""
    latencies, errors = [], 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for lat, err in ex.map(lambda _: one_request(url, payloads), range(n)):
            if err:
                errors += 1
            else:
                latencies.append(lat)
    return latencies, errors, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default=None,
                    help="모델 id (생략 시 .env VLLM_MODEL → /v1/models 자동감지)")
    ap.add_argument("--resolution", choices=["720p", "1080p"], default="720p")
    ap.add_argument("--image-dir", default=None, help="이미지 폴더 직접 지정 (resolution 보다 우선)")
    ap.add_argument("--levels", default=None,
                    help="동시성 수준 콤마 구분 (생략 시 .env MAX_NUM_SEQS 까지 2의 거듭제곱)")
    ap.add_argument("--rounds", type=int, default=5, help="수준별 요청 수 = 동시성 x rounds")
    ap.add_argument("--prompt", default="Is this color yellow? Answer with only 'yes' or 'no'.",
                    help="모든 요청에 보낼 yes/no VQA 질문")
    args = ap.parse_args()

    # 서버 설정(.env) 단일 진실. 모델·동시성 레벨 디폴트를 여기서 끌어온다.
    env = load_env(ENV_PATH)
    max_seqs = int(env.get("MAX_NUM_SEQS", 128))

    img_dir = Path(args.image_dir) if args.image_dir else IMAGE_DIRS[args.resolution]
    try:
        model = args.model or env.get("VLLM_MODEL") or detect_model(args.url)
    except Exception as e:
        raise SystemExit(f"[에러] 서버에 연결할 수 없습니다 ({args.url}). 먼저 docker compose up -d 로 기동하세요.\n  원인: {e}")
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"[에러] 테스트 이미지가 없습니다: {img_dir}/*.jpg")

    # 모든 이미지를 미리 payload로 인코딩 → 요청 시 random.choice로 하나씩 선택
    payloads = [build_payload(model, base64.b64encode(p.read_bytes()).decode(), args.prompt)
                for p in images]
    if args.levels:
        levels = [int(x) for x in args.levels.split(",") if x.strip()]
        levels_src = "--levels"
    else:
        levels = default_levels(max_seqs)
        levels_src = f"env MAX_NUM_SEQS={max_seqs}"

    print(f"server : {args.url}")
    print(f"model  : {model}")
    print(f"images : {img_dir}  ({len(images)}장: {', '.join(p.stem for p in images)})")
    print(f"prompt : {args.prompt}")
    print(f"levels : {','.join(map(str, levels))}  ({levels_src})")
    print(f"rounds : {args.rounds}  (요청 수 = 동시성 x rounds, 이미지는 매 요청 랜덤 선택)")
    print()

    one_request(args.url, payloads)  # warmup

    cols = f"{'동시성':>6} {'요청수':>6} {'mean':>9} {'p50':>9} {'p95':>9} {'p99':>9} {'max':>9} {'rps':>8} {'err':>4}"
    print(cols)
    print("-" * (len(cols) - 6))  # 한글 폭 보정
    for c in levels:
        n = c * args.rounds
        lats, errors, wall = run_level(args.url, payloads, c, n)
        s = sorted(lats)
        mean = statistics.mean(s) if s else float("nan")
        rps = len(s) / wall if wall > 0 else float("nan")
        mx = s[-1] if s else float("nan")
        print(f"{c:>6} {n:>6} "
              f"{mean*1000:>9.1f} {percentile(s,0.50)*1000:>9.1f} "
              f"{percentile(s,0.95)*1000:>9.1f} {percentile(s,0.99)*1000:>9.1f} "
              f"{mx*1000:>9.1f} {rps:>8.1f} {errors:>4}")
    print("\n단위: mean/p50/p95/p99/max = ms,  rps = 초당 처리 요청 수,  err = 실패 요청 수")


if __name__ == "__main__":
    main()
