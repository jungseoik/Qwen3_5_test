# 다중 모델 sweep 평가

`src/evaluation/sweep.py` 는 여러 모델을 차례로 띄워가며 같은 오탐 데이터셋에 대해
오탐감소율을 측정하고, 한 폴더 안에 결과를 정리해 모델별로 비교합니다.

## 동작 흐름

```
[준비] docker compose down
loop  models.py 의 각 model_id 에 대해:
    1) VLLM_MODEL=<model_id> docker compose up -d   (.env 안 건드림)
    2) /health 200 폴링 (기본 15분 타임아웃)
    3) subprocess 로 fp_reduction.py 실행
       → 모델별 폴더에 eval.md / eval.csv / yes/{cat}/*.jpg / manifest.csv 저장
    4) 마지막이 아니면 docker compose down
[종료] 마지막 모델은 그대로 띄워둔 채로 sweep 종료 + summary.md/csv 작성
       (Ctrl+C 시에는 컨테이너 정리)
```

## 모델 교체 방식

`.env` 를 수정하지 않습니다. compose 변수 우선순위(shell env > .env)를 이용해
sweep 이 `docker compose` 실행 시점에만 `VLLM_MODEL` 환경변수를 덮어씁니다.
sweep 이 끝난 뒤에도 `.env` 의 `VLLM_MODEL` 값은 그대로 유지됩니다.

`MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, `GPU_MEMORY_UTILIZATION` 등 다른 모든
설정은 `.env` 값을 그대로 공유합니다 (모델별 다른 env 는 두지 않음).

## 평가 모델 리스트 — `src/evaluation/models.py`

`prompts.py` 처럼 별도 모듈에 모델 리스트를 두고 직접 편집합니다.

```python
MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    # "Qwen/Qwen3.5-27B",     # 캐시 미다운로드시 주석 처리
    # "Qwen/Qwen3.5-35B-A3B",
]
```

HF 캐시(`~/.cache/huggingface/hub`)에 safetensors weight 가 없는 모델은 sweep 이
**자동으로 스킵** 합니다 (요약 표에 `skipped` 로 표시). 강제로 시도하려면
`--no-require-cached`.

## 실행

```bash
# 기본: models.py 의 전 모델 × 전수 평가
python src/evaluation/sweep.py

# 빠른 비교용 샘플 (카테고리당 50장)
python src/evaluation/sweep.py --limit 50

# 일부 모델만 (단축 매칭: '0.8B' → 'Qwen/Qwen3.5-0.8B')
python src/evaluation/sweep.py --models 0.8B,9B

# 특정 카테고리만 (violence 제외 등)
python src/evaluation/sweep.py --category falldown,fire,smoke
```

## 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--models` | `models.py` 전체 | 콤마 구분, 단축 매칭 지원 (예: `0.8B,9B`) |
| `--limit` | `0` (전수) | 모델당 카테고리별 샘플 수. 수치 지정 시 그만큼만 |
| `--category` | 전체 | 평가 카테고리 콤마 구분 (fp_reduction 으로 전달) |
| `--concurrency` | `.env` `MAX_NUM_SEQS` | 모든 모델에 동일 적용 |
| `--collect` | `yes` | 실패 케이스 수집 모드 (`yes`/`all`/`none`/...) |
| `--url` | `http://localhost:8000` | 서버 주소 |
| `--load-timeout` | `900` | 모델 로딩 헬스체크 대기 (초) |
| `--no-require-cached` | off | 캐시에 weight 없어도 강제 시도 |

## 결과 저장 구조

sweep 한 번이 한 폴더로 묶입니다.

```
results/
└── sweep_20260528_120000/
    ├── summary.md                    # 모델 × 카테고리 비교 표
    ├── summary.csv                   # 같은 데이터 (plot/엑셀용)
    ├── Qwen3.5-0.8B/                 # 모델별 폴더 = 단일 eval 폴더와 동일 구조
    │   ├── eval.md                   # 요약 + 서버 설정 + 사용 프롬프트
    │   ├── eval.csv
    │   ├── manifest.csv              # 못 거른 케이스 메타데이터
    │   └── yes/{category}/*.jpg      # 이 모델이 못 거른 오탐 썸네일
    ├── Qwen3.5-2B/
    ├── Qwen3.5-4B/
    └── Qwen3.5-9B/
```

## summary.md 형식

```
## 오탐감소율 (%)

| 모델               | falldown | fire   | smoke | violence | 전체      | 상태 |
|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-0.8B | 80.0%    | 100.0% | 60.0% | 100.0%   | **85.0%** | ok |
| Qwen/Qwen3.5-2B   | 88.0%    | 100.0% | 72.0% | 100.0%   | **90.0%** | ok |
| Qwen/Qwen3.5-9B   | 92.0%    | 100.0% | 84.0% | 100.0%   | **94.0%** | ok |
| Qwen/Qwen3.5-27B  | -        | -      | -     | -        | -         | `skipped` |
```

각 셀은 `fp_reduction_pct = no / (no+yes) × 100` 값입니다 (unparsed/error 제외).

## 실패 / 예외 처리

| 상황 | 동작 |
|---|---|
| 모델 weight 캐시 없음 | `--require-cached` 가 기본 ON → 자동 스킵, 표에 `skipped` |
| 모델 로딩 헬스체크 타임아웃 | 해당 모델 `load_timeout`, 컨테이너 down 후 다음 모델 |
| 평가 일부 요청 에러 | `unparsed/error` 컬럼에 카운트만 늘고 sweep 진행 |
| Ctrl+C | 현재 컨테이너 `docker compose down`, 부분 결과로 `summary.md` 작성 |
| sweep 정상 종료 | 마지막 모델 그대로 띄워둠 (이어서 추가 실험 바로 가능) |

## 다른 도구와의 관계

- `src/evaluation/fp_reduction.py` — 단일 모델 평가 (sweep 이 subprocess 로 호출)
- `src/evaluation/prompts.py` — 카테고리별 프롬프트 (sweep 의 모든 모델이 공유)
- `src/evaluation/models.py` — sweep 대상 모델 리스트
- `.env` — 서버 공통 설정 (sweep 은 `VLLM_MODEL` 만 일시 override)
