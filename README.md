# Qwen3.5 vLLM 서버

Docker로 Qwen3.5 모델을 OpenAI 호환 API 서버로 띄우는 튜토리얼입니다. 별도의 백엔드 코드 없이 vLLM 공식 이미지가 `/v1/chat/completions` 엔드포인트를 그대로 제공하며, 기동 후에는 OpenAI SDK 또는 `curl`로 바로 호출할 수 있습니다.

## 프로젝트 구조

```
Qwen3_5_test/
├── docker-compose.yml         # vLLM 서버 기동 compose
├── .env / .env.example        # 서버 설정 (모델, GPU, VRAM 등)
├── assets/test/{720p,1080p}/  # 부하 테스트용 색깔 이미지
├── nas/data/                  # 평가 데이터셋 (오탐 썸네일·비디오)
├── results/                   # 평가 결과 자동 저장 (gitignore, md/csv)
├── src/
│   ├── tests/                 # 동시 요청 레이턴시 벤치
│   │   └── bench_concurrency.py
│   ├── dataset/               # 데이터셋 통계
│   │   └── stats.py
│   ├── evaluation/            # 오탐감소율 평가
│   │   ├── prompts.py         # 카테고리별 VQA 프롬프트 (편집 대상)
│   │   ├── models.py          # sweep 평가 모델 리스트 (편집 대상)
│   │   ├── fp_reduction.py    # 단일 모델 FP 평가 (전부 오탐 가정)
│   │   ├── labeled_eval.py    # 단일 모델 라벨 평가 (오탐감소율 + 정탐률)
│   │   └── sweep.py           # 다중 모델 sweep (--eval fp|labeled)
│   └── labeling/             # 카테고리 True/False 라벨링 도구 (웹)
│       ├── config.py         # 입력/출력 경로 설정 (편집 대상)
│       ├── label_server.py   # http.server 백엔드 + CLI
│       └── static/index.html # 키보드 라벨링 UI
└── docs/                      # 각 도구 상세 문서
    ├── dataset_stats.md
    ├── eval_fp_reduction.md
    ├── labeled_eval.md            # 라벨 기반 평가 (오탐감소율 + 정탐률)
    ├── sweep.md
    ├── labeling_tool.md           # 라벨링 도구 사용법
    └── labeling_tool_design.md    # 라벨링 도구 구현 설계(재현용)
```

## 요구 사항

| 항목 | 확인 명령 | 정상 동작 |
|---|---|---|
| NVIDIA GPU 드라이버 | `nvidia-smi` | GPU 이름 / VRAM 출력 |
| Docker | `docker --version` | 버전 출력 |
| NVIDIA Container Toolkit | `docker run --rm --gpus all ubuntu nvidia-smi` | 컨테이너 내부에서 GPU 출력 |

마지막 명령에서 GPU가 보이지 않으면 NVIDIA Container Toolkit이 설치되지 않은 것입니다. 이 경우 컨테이너가 GPU를 사용할 수 없습니다.

### GPU 아키텍처별 이미지

GPU에 맞는 이미지 태그를 `.env`의 `VLLM_IMAGE`에 지정합니다.

| GPU | 이미지 태그 |
|---|---|
| Blackwell (B200, RTX 5090, RTX PRO 6000) | `vllm/vllm-openai:cu130-nightly` |
| Hopper (H100, H200) | `vllm/vllm-openai:latest` |
| Ampere (A100, A10, RTX 30/40 시리즈) | `vllm/vllm-openai:latest` |

이 장비는 RTX PRO 6000 Blackwell(96GB)이므로 기본값 `cu130-nightly`를 사용합니다.

## 빠른 시작

### 1. 설정 파일 생성

```bash
cp .env.example .env
```

`.env`는 모델, 포트, GPU, VRAM 등 서버 설정을 담습니다. 기본값을 유지하면 `Qwen/Qwen3.5-0.8B` 모델이 8000번 포트로 기동됩니다. 처음에는 수정 없이 진행해도 됩니다.

`.env`는 `.gitignore`에 포함되어 커밋되지 않으므로 토큰 등 민감 정보를 넣어도 됩니다.

### 2. 서버 기동

```bash
docker compose up -d
```

첫 기동 시 모델을 HuggingFace에서 내려받으므로 수 분이 소요될 수 있습니다. 이후에는 캐시되어 빠르게 기동됩니다.

기동 로그는 다음으로 확인합니다.

```bash
docker compose logs -f qwen-vllm
```

`Application startup complete` 또는 `Uvicorn running on http://0.0.0.0:8000` 로그가 출력되면 준비가 완료된 것입니다. `Ctrl+C`로 로그 보기를 종료해도 서버는 계속 실행됩니다.

### 3. 상태 확인

```bash
curl http://localhost:8000/health      # HTTP 200이면 정상
docker compose ps                       # STATUS가 Up (healthy)이면 정상
```

## API 호출

### 모델 목록

```bash
curl http://localhost:8000/v1/models
```

### 채팅 (curl)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-0.8B",
    "max_tokens": 64,
    "temperature": 0,
    "messages": [
      {"role": "user", "content": "Say hello in one short sentence."}
    ]
  }'
```

응답은 `choices[0].message.content`에 담깁니다.

### 채팅 (Python, OpenAI SDK)

```bash
pip install openai
```

```python
from openai import OpenAI

# vLLM은 인증하지 않으므로 api_key는 임의 값을 사용합니다.
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-0.8B",
    max_tokens=64,
    temperature=0,
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)
print(resp.choices[0].message.content)
```

## 서버 관리

| 동작 | 명령 |
|---|---|
| 상태 확인 | `docker compose ps` |
| 로그 확인 | `docker compose logs -f qwen-vllm` |
| 중지 (VRAM 반납) | `docker compose stop` |
| 재개 | `docker compose start` |
| 종료 및 컨테이너 삭제 | `docker compose down` |
| 설정 변경 후 재기동 | `docker compose down && docker compose up -d` |

`.env` 값을 변경한 경우 `down && up`으로 컨테이너를 재생성해야 반영됩니다. `restart`만으로는 환경변수가 갱신되지 않습니다.

## 주요 설정값

자주 사용하는 값입니다. 각 변수의 상세 설명은 `.env.example` 주석을 참고하십시오.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `VLLM_IMAGE` | `vllm/vllm-openai:cu130-nightly` | GPU 아키텍처별 이미지 |
| `VLLM_MODEL` | `Qwen/Qwen3.5-0.8B` | 사용할 HuggingFace 모델 |
| `VLLM_PORT` | `8000` | 호스트 접근 포트 |
| `GPU_MEMORY_UTILIZATION` | `0.85` | vLLM이 점유할 VRAM 상한 (0~1) |
| `MAX_MODEL_LEN` | `4096` | 한 요청의 토큰 상한 |
| `MAX_NUM_SEQS` | `128` | 동시 처리 요청 상한 |
| `MAX_NUM_BATCHED_TOKENS` | `32768` | 한 스텝에 처리할 토큰 상한 |
| `NVIDIA_VISIBLE_DEVICES` | `0` | 사용할 GPU index |

## 모델 교체

`.env`의 `VLLM_MODEL`을 변경한 뒤 `docker compose down && docker compose up -d`를 실행하면 모델이 교체됩니다. 아래는 캐시에 준비된 Qwen3.5 계열 모델의 repo ID와 fp16 기준 추정 VRAM입니다.

| `VLLM_MODEL` (repo ID) | weight VRAM | 비고 |
|---|---|---|
| `Qwen/Qwen3.5-0.8B` | ~1.6GB | 기본값 |
| `Qwen/Qwen3.5-2B` | ~4GB | |
| `Qwen/Qwen3.5-4B` | ~8GB | |
| `cyankiwi/Qwen3.5-4B-AWQ-4bit` | ~3GB | AWQ 4bit 양자화 |
| `Qwen/Qwen3.5-9B` | ~18GB | |
| `QuantTrio/Qwen3.5-9B-AWQ` | ~7GB | AWQ 4bit 양자화 |
| `Qwen/Qwen3.5-27B` | ~54GB | |
| `Qwen/Qwen3.5-35B-A3B` | ~70GB | MoE |

- 모델 크기가 커서 VRAM이 부족하면 `GPU_MEMORY_UTILIZATION`을 상향합니다.
- 단일 GPU에 적재되지 않으면 `GPU_COUNT`와 `TENSOR_PARALLEL_SIZE`를 GPU 수만큼 늘립니다.
- AWQ 양자화 모델은 vLLM이 자동 인식하므로 repo ID만 변경하면 됩니다.

## 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| 컨테이너 즉시 종료, CUDA 오류 | `VLLM_IMAGE` 태그가 GPU와 불일치 → 아키텍처에 맞는 태그로 교체 후 `down && up` |
| `could not select device driver ... gpu` | NVIDIA Container Toolkit 미설치 → 요구 사항 항목 확인 |
| `port is already allocated` | 8000번 포트 사용 중 → `.env`의 `VLLM_PORT` 변경 |
| 모델 다운로드 실패 (401/403) | gated/private 모델 → `.env`의 `HF_TOKEN` 설정 후 재기동 |
| 기동이 오래 걸림 | 모델 다운로드 진행 중 → `docker compose logs -f`로 확인 (start_period 10분 유예) |
| VRAM 부족 (OOM) | `GPU_MEMORY_UTILIZATION` 하향 또는 다른 GPU 프로세스 종료 |

## 부하 테스트 (동시 요청 레이턴시)

`src/tests/bench_concurrency.py`는 서버에 이미지 VQA 요청(출력 1토큰, yes/no)을 동시성 수준별로 보내 처리 시간을 측정하고 표로 출력합니다. 표준 라이브러리만 사용하므로 별도 설치가 필요 없습니다.

요청마다 색깔 이미지를 랜덤으로 골라 `"Is this color yellow? Answer with only 'yes' or 'no'."` 질문과 함께 보냅니다. 매번 다른 이미지가 들어오는 실제 운영 상황을 재현합니다.

### 테스트 이미지

`assets/test/{해상도}/`에 해상도별 단색 이미지 7장(빨주노초파남보)이 있습니다.

| 폴더 | 해상도 | 파일 |
|---|---|---|
| `assets/test/720p/` | 1280×720 | `red.jpg`, `orange.jpg`, `yellow.jpg`, `green.jpg`, `blue.jpg`, `indigo.jpg`, `violet.jpg` |
| `assets/test/1080p/` | 1920×1080 | (동일) |

### 실행

서버가 기동된 상태에서 실행합니다.

```bash
python src/tests/bench_concurrency.py
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--resolution` | `720p` | 테스트 이미지 해상도 (`720p` / `1080p`) |
| `--levels` | `.env` `MAX_NUM_SEQS`까지 2의 거듭제곱 | 측정할 동시성 수준 (예: `MAX_NUM_SEQS=128` → `1,2,4,8,16,32,64,128`) |
| `--rounds` | `5` | 수준별 요청 수 = 동시성 × rounds |
| `--prompt` | `Is this color yellow? ...` | 모든 요청에 보낼 yes/no 질문 |
| `--url` | `http://localhost:8000` | 서버 주소 |
| `--model` | `.env` `VLLM_MODEL` → `/v1/models` | 모델 id (이 우선순위로 폴백) |

bench와 eval 모두 별도 인자가 없으면 **레포 루트의 `.env`** 를 읽어 떠 있는 서버의 모델·동시성 한도를 자동으로 가져옵니다.

```bash
python src/tests/bench_concurrency.py --resolution 1080p --rounds 8
```

### 출력 예시

```
   동시성    요청수      mean       p50       p95       p99       max      rps  err
-----------------------------------------------------------------------
     1      4      54.7      46.7      81.8      85.8      86.8     18.2    0
    16     64     326.0     340.5     394.7     408.6     428.5     43.3    0
   128    512    2330.3    2588.8    2666.1    2742.0    2821.2     48.7    0

단위: mean/p50/p95/p99/max = ms,  rps = 초당 처리 요청 수,  err = 실패 요청 수
```

### 표 보는 법

각 행은 "동시에 N개씩 요청을 보냈을 때"의 결과입니다.

| 열 | 의미 | 해석 |
|---|---|---|
| `동시성` | 동시에 보낸 요청 수 | 높을수록 부하가 큼 |
| `요청수` | 그 행에서 보낸 총 요청 수 (동시성 × rounds) | 측정 표본 수 |
| `mean` | 요청 1건의 평균 처리 시간 (ms) | 전반적인 속도 |
| `p50` | 중앙값 — 요청의 50%가 이 시간 안에 끝남 (ms) | 보통의 사용자 체감 |
| `p95` | 요청의 95%가 이 시간 안에 끝남 (ms) | 느린 쪽 5%의 경계 |
| `p99` | 요청의 99%가 이 시간 안에 끝남 (ms) | 최악에 가까운 지연. 운영 SLA 기준으로 자주 씀 |
| `max` | 가장 느렸던 요청 (ms) | 최악값 |
| `rps` | 초당 처리한 요청 수 | 서버의 실제 처리량(throughput) |
| `err` | 실패한 요청 수 | 0이어야 정상 |

**핵심 읽는 법**: 동시성을 올리면 `rps`(처리량)는 어느 지점까지 오르다가 더 안 오르고 멈춥니다(포화). 그 지점을 넘어서 동시성을 더 올리면 `rps`는 그대로인데 `p95`/`p99`(지연)만 계속 커집니다 — 요청이 큐에서 기다리기 때문입니다. 따라서 **`rps`가 포화되기 직전의 동시성**이 그 설정의 적정 운영 지점이며, 이를 보고 `MAX_NUM_SEQS` / `MAX_NUM_BATCHED_TOKENS` 값을 조정합니다.

## 데이터셋 통계

오탐 로그 데이터셋(카테고리별 / 날짜별 썸네일·비디오)의 개수를 집계해 표로 출력하는 도구입니다.

```bash
python src/dataset/stats.py
```

자세한 사용법, 기대하는 디렉토리 구조, 옵션, 출력 예시는 [docs/dataset_stats.md](docs/dataset_stats.md)를 참고하십시오.

## 오탐감소율 평가

카테고리별 오탐 썸네일을 vLLM 서버에 보내 yes/no 2차 검증을 받고, 오탐감소율을 측정하는 평가 도구입니다. 동시 요청으로 처리하며 실행 중 진행 바가 stderr에 표시됩니다. 동시성·모델·서버 설정은 `.env`에서 자동으로 끌어옵니다.

오탐감소율은 `no / (no + yes) × 100` 으로, `unparsed`/`error` 같은 비정상 응답은 분모에서 제외해 측정 노이즈가 점수를 깎지 않게 합니다.

실행 결과는 **레포 루트 `results/` 아래 타임스탬프 폴더 하나로 묶여 자동 저장**됩니다.

```
results/eval_20260528_115901/
├── eval.md            # 요약 표 + 서버 설정 + 사용 프롬프트 전문
├── eval.csv           # 요약 표
├── manifest.csv       # 못 거른 오탐 케이스 메타데이터
└── yes/{category}/*.jpg   # VLM이 잘못 통과시킨 썸네일 원본
```

```bash
python src/evaluation/fp_reduction.py --limit 50
```

카테고리별 프롬프트는 `src/evaluation/prompts.py`에서 편집합니다. 자세한 개념, 옵션, 출력 항목, 실패 케이스 수집 동작은 [docs/eval_fp_reduction.md](docs/eval_fp_reduction.md)를 참고하십시오.

## 라벨 기반 평가 (오탐감소율 + 정탐률)

사람이 검수한 라벨(`true`/`false`)이 붙은 이미지셋에 대해 **두 지표를 동시에** 측정합니다. 단순 FP 거름만이 아니라 "오탐을 줄이면서 정탐을 얼마나 유지하는가"를 양면으로 본다는 게 핵심입니다.

```
오탐감소율 = TN / (TN + FP)     # false 라벨 중 모델이 no 라 거른 비율
정탐률     = TP / (TP + FN)     # true  라벨 중 모델이 yes 라 유지한 비율
```

입력은 라벨링 도구가 만든 export 폴더 (`results/labeling/export/export_<TS>/{cat}/{true,false}/*.jpg`)를 그대로 사용합니다. `--path` 생략 시 가장 최근 export 자동 선택.

```bash
python src/evaluation/labeled_eval.py                                # 최신 export × 전수
python src/evaluation/labeled_eval.py --category fire,smoke --limit 50
```

결과는 `results/eval_labeled_<TS>/`에 모이며, 오분류 케이스가 **fp/**(놓친 오탐)와 **fn/**(놓친 정탐) 두 폴더로 자동 분리됩니다.

```
results/eval_labeled_20260529_092921/
├── eval.md / eval.csv                  # 카테고리 × {TP/FN/TN/FP, 오탐감소율, 정탐률}
├── manifest.csv                        # 오분류 케이스 메타
├── fp/{category}/*.jpg                 # label=false 인데 모델이 yes (필터 실패)
└── fn/{category}/*.jpg                 # label=true 인데 모델이 no (정탐 놓침)
```

자세한 개념·옵션은 [docs/labeled_eval.md](docs/labeled_eval.md) 참고. sweep 과 함께 쓰려면 아래 섹션 `--eval labeled` 참고.

## 다중 모델 sweep 평가

여러 모델을 차례로 띄워가며 같은 오탐 데이터셋에 대해 평가를 반복하고, 모델별 성능을 한 표로 비교하는 도구입니다. sweep은 `.env`를 수정하지 않고 `docker compose` 호출 시점에 `VLLM_MODEL`만 일시 override하는 방식으로 모델을 교체합니다. 단일 평가 도구(`fp_reduction.py`)는 그대로 두고 위에서 wrapping합니다.

평가 대상 모델은 `src/evaluation/models.py`에서 편집합니다.

```bash
python src/evaluation/sweep.py                              # 전 모델 × 전수 (FP 모드)
python src/evaluation/sweep.py --limit 50                   # 빠른 샘플 (카테고리당 50장)
python src/evaluation/sweep.py --models 0.8B,9B             # 부분집합 (단축 매칭)
python src/evaluation/sweep.py --category falldown,fire,smoke   # 특정 카테고리만

# 라벨 모드 — 오탐감소율과 정탐률을 모델별로 동시 비교
python src/evaluation/sweep.py --eval labeled
```

결과는 `results/sweep_YYYYMMDD_HHMMSS/` 한 폴더에 묶여 저장됩니다.

```
sweep_20260528_120000/
├── summary.md                       # 모델 × 카테고리 비교 표
├── summary.csv
├── Qwen3.5-0.8B/                    # 모델별 폴더 = 단일 평가 폴더와 동일 구조
│   ├── eval.md
│   ├── eval.csv
│   ├── manifest.csv
│   └── yes/{category}/*.jpg
└── ...
```

sweep 종료 후에는 마지막 모델이 그대로 띄워져 있어 이어서 다른 작업을 바로 할 수 있습니다. 모델 교체 방식, 옵션, 예외 처리는 [docs/sweep.md](docs/sweep.md)를 참고하십시오.

## 카테고리 True/False 이미지 라벨링

카테고리별로 분류된 썸네일이 진짜 그 카테고리가 맞는지 사람이 키보드로 빠르게 검수하고, 결과를 `{카테고리}/true|false/` 구조로 복사·재정렬하는 웹 기반 도구입니다. 표준 라이브러리 `http.server`로 로컬 웹서버를 띄우므로 별도 설치가 필요 없습니다.

```bash
python src/labeling/label_server.py        # 기동 후 브라우저에서 http://localhost:8800
```

`F`/`J`(또는 `↑`/`↓`)로 True/False를 찍으면 자동으로 다음 이미지로 넘어가고, 라벨은 즉시 저장되어 언제든 이어서 작업할 수 있습니다. `E` 키로 결과를 `results/labeling/export/{category}/{true,false}/`에 복사합니다. 입력/출력 경로는 `src/labeling/config.py`에서 바꿉니다.

헤드리스 GPU 서버라면 `ssh -L 8800:localhost:8800 <서버>`로 포트포워딩 후 로컬 브라우저에서 접속합니다. 키보드 조작표, 이어하기, Export, 옵션은 [docs/labeling_tool.md](docs/labeling_tool.md), 구현 세부는 [docs/labeling_tool_design.md](docs/labeling_tool_design.md)를 참고하십시오.

## 다음 단계

- 이미지(멀티모달) 입력 기반 VQA 호출
- 프롬프트 빌더 모듈 구성
- 동시성 / VRAM / max_tokens 부하 측정 및 운영 튜닝
