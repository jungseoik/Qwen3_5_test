# 오분류 description 진단 도구

라벨 평가에서 모델은 `yes`/`no` 한 토큰만 답하므로 **왜 그렇게 판정했는지** 알 수 없습니다. 특히 진짜 사건을 `no` 로 걸러버리는 **FN(놓친 정탐)** 이 많으면 정탐유지율이 떨어집니다. 이 도구는 오분류 케이스에 대해 모델이 장면을 **한국어로 풀어 설명(description)** 하게 만들고, 그 설명을 사람이 이미지·프롬프트·정답과 나란히 빠르게 넘겨보며 원인(인식 실패 vs 프롬프트 과민/둔감)을 진단하게 합니다.

외부 의존성 없이 표준 라이브러리만 사용합니다.

## 개념

```
평가(sweep_labeled)  →  manifest.csv (fp/fn 케이스)
        │  describe 생성
        ▼
{Model}/describe/describe.jsonl  (한국어 description)
        │  분석
        ▼
Web UI  (모델별 · FN/FP별로 넘겨보기)
```

- **base 프롬프트**: 평가에 쓰인 yes/no 판정 프롬프트.
- **describe 프롬프트**: 진단용 **영어** 묘사 프롬프트 (falldown=사람, fire=장면/불꽃, smoke=연기 색·구름 구별). 오직 "보이는 것 묘사"에 집중하며 판정 단어를 넣지 않습니다.
- 작은 모델은 한국어를 직접 생성하면 글자가 깨질 수 있어, **영어 묘사는 vLLM 이, 한국어 번역은 Google Gemini API**가 합니다. UI 는 한국어를 기본 표시하고 영어 원문은 `e` 키/토글로 봅니다.
- description 은 max_tokens 1 짜리 yes/no 평가와 달리 수백 토큰을 생성하므로, 서버 KV/동시성을 `.env.describe` 오버레이로 그 실행에만 맞춥니다(아래 참고).

## 전제

- 먼저 라벨 평가를 돌려 각 `results/sweep_labeled_*/{Model}/manifest.csv` 가 있어야 합니다 (평가 기본 옵션 `--collect fp,fn` 으로 생성됨). 평가 방법은 [labeled_eval.md](labeled_eval.md) / [sweep.md](sweep.md) 참고.
- description 을 만들 그 모델로 vLLM 서버가 떠 있어야 합니다 (`docker compose up -d`).
- 한국어 번역용 **Gemini API 키**가 필요합니다. [aistudio.google.com/apikey](https://aistudio.google.com/apikey) 에서 발급해 `.env` 의 `GEMINI_API_KEY` 에 넣습니다(`.env` 는 gitignore). 번역 모델은 `GEMINI_TRANSLATE_MODEL`(기본 `gemini-2.5-flash`). 키 작동은 `python src/evaluation/translate.py --selftest` 로 확인합니다. 키가 없으면 번역을 건너뛰고 영어 description 만 저장합니다(`--no-translate` 와 동일).

## 전체 순서 (3단계)

```bash
# 0) (선행) 라벨 평가 — 이미 했다면 생략
python src/evaluation/sweep.py --eval labeled

# 1) description 생성
#    단일 모델 (현재 떠 있는 서버 기준)
python src/evaluation/describe_eval.py --from results/sweep_labeled_20260529_095220/Qwen3.5-4B
#    다중 모델 (모델을 차례로 재기동하며 전부)
python src/evaluation/describe_sweep.py --from results/sweep_labeled_20260529_095220

# 2) 분석 Web UI
python src/analysis/describe_server.py --path results/sweep_labeled_20260529_095220
#    → 브라우저에서 http://localhost:8801
```

## 1단계: description 생성

### 단일 모델 — `describe_eval.py`

현재 떠 있는 vLLM 서버로, 지정한 모델 폴더의 `manifest.csv` 케이스에 description 을 만들어 그 폴더 안 `describe/` 에 저장합니다.

```bash
# 기본: fp+fn 전수
python src/evaluation/describe_eval.py --from results/sweep_labeled_20260529_095220/Qwen3.5-4B

# 놓친 정탐(FN)만
python src/evaluation/describe_eval.py --from <폴더> --kind fn

# 특정 카테고리 / kind·카테고리당 20장 / 토큰 수 조정
python src/evaluation/describe_eval.py --from <폴더> --category smoke --limit 20 --max-tokens 512
```

| 인자 | 의미 | 기본 |
|---|---|---|
| `--from` | 평가 결과의 모델 폴더(`manifest.csv` 보유) **(필수)** | — |
| `--kind` | 대상 종류 `fp,fn,tp,tn` (콤마, `all` 가능) | `fp,fn` |
| `--category` | 카테고리 필터 (콤마) | 전체 |
| `--limit` | kind/카테고리당 case 수 (0=전수) | `0` |
| `--max-tokens` | 영어 description 토큰 수 | `512` |
| `--gemini-model` | 번역에 쓸 Gemini 모델 | `.env GEMINI_TRANSLATE_MODEL` |
| `--no-translate` | 번역 생략, 영어만 저장 | (번역 on) |
| `--concurrency` | 동시 요청 수 | `.env MAX_NUM_SEQS` |
| `--model` | 모델 id | `.env`→자동감지 |
| `--out` | 저장 폴더 | `{from}/describe/` |

> 단일 실행은 **현재 떠 있는 서버 설정**을 그대로 씁니다. 생성형 워크로드에 맞춘 서버 튜닝(`.env.describe`)은 모델을 재기동하는 `describe_sweep` 가 적용합니다. 단일 모델도 튜닝된 서버로 돌리려면 `describe_sweep --models <하나>` 를 쓰세요.

### 다중 모델 — `describe_sweep.py`

`sweep_labeled_*` 폴더의 각 모델을 차례로 재기동하며 영어 description 을 만들고, **전 모델 생성이 끝난 뒤 한 번에 Gemini 배치 번역**을 돌립니다(생성=vLLM/GPU, 번역=Gemini 를 분리해 빠름). 결과는 각 모델 폴더 `describe/` 에 쌓입니다.

```bash
python src/evaluation/describe_sweep.py --from results/sweep_labeled_20260529_095220

# 일부 모델 / FN만 / 번역 끄고 영어만
python src/evaluation/describe_sweep.py --from <폴더> --models 4B,9B --kind fn
python src/evaluation/describe_sweep.py --from <폴더> --no-translate
```

동작은 2단계입니다:
1. **영어 생성** — 모델마다 재기동(`.env.describe` 오버레이 적용) → `describe_eval --no-translate` 로 영어 description 만.
2. **일괄 번역** — 전 모델 생성 후, 각 모델 결과를 Gemini 배치(`--batch-size`)로 한 번에 번역.

`manifest.csv` 가 없는 모델, HF 캐시에 weight 가 없는 모델은 자동 스킵합니다. 종료 시 마지막 모델은 띄워둡니다(`sweep.py` 와 동일 정책). Gemini 키가 없으면 번역 단계는 자동 생략됩니다.

`describe_sweep` 전용 인자:

| 인자 | 의미 | 기본 |
|---|---|---|
| `--max-num-seqs` | 서버 `MAX_NUM_SEQS` override (`.env.describe` 보다 우선) | `.env.describe`→`.env` |
| `--max-model-len` | 서버 `MAX_MODEL_LEN` override | 〃 |
| `--concurrency` | 클라이언트(vLLM) 동시 요청 수 | 적용된 `MAX_NUM_SEQS` 에 자동 일치 |
| `--batch-size` | 번역 배치 크기 (한 Gemini 호출당 건수) | `50` |
| `--translate-concurrency` | 동시 번역 묶음 수 | `4` |
| `--max-tokens` / `--gemini-model` / `--no-translate` | 영어 토큰수 / 번역 모델 / 번역 생략 | 512 / .env / 번역 on |

### 서버 튜닝 오버레이 (`.env.describe`)

평가는 `max_tokens=1` 이라 동시 128 도 무방했지만, description 은 수백 토큰을 디코드하므로 동시 시퀀스를 줄여 KV 캐시를 확보하고 컨텍스트 여유를 둡니다. `describe_sweep` 는 모델 기동 시 `.env.describe` 를 **`.env` 위에 덮어** 적용합니다(`.env` 파일은 불변). 기본값:

```
MAX_NUM_SEQS=32          # 128 → 32
MAX_MODEL_LEN=8192       # 4096 → 8192
MAX_NUM_BATCHED_TOKENS=16384
```

여기 없는 키는 `.env` 를 상속합니다. 파일을 비우면 `.env` 기본값으로 동작합니다. 그때그때 바꾸려면 `--max-num-seqs`/`--max-model-len` 으로 덮어쓰세요.

### 산출물

```
results/sweep_labeled_{TS}/{Model}/describe/
├── describe.jsonl   # 케이스당 1줄 {id,kind,category,label,base_verdict,source,description,description_ko}
├── describe.csv
└── describe.md      # 모델·프롬프트·kind×카테고리 건수 요약
```

`description` 은 영어 원문, `description_ko` 는 한국어 번역입니다. 기존 `manifest.csv`/`eval.*`/`fp`/`fn` 은 건드리지 않습니다(추가만). 재실행하면 `describe/` 만 갱신됩니다.

### 번역 모듈 (`translate.py`)

번역은 `src/evaluation/translate.py` 가 Gemini API 로 처리합니다(SDK 없이 `urllib` REST). 단건/배치 함수와 재시도(429·5xx 지수 백오프)·배치 정렬 검증·분할 폴백을 제공합니다.

```bash
python src/evaluation/translate.py --selftest             # API 작동 확인
python src/evaluation/translate.py "A person is lying down."  # 임의 문장 번역
```

### 기존 결과 재번역 (`retranslate_gemini.py`, 일회성)

이미 만들어진 describe 결과의 **한국어 번역만** 다시 만들고 싶을 때 씁니다. 영어 `description` 은 재사용하고 `description_ko` 만 Gemini 로 재생성해 jsonl/csv 를 덮어씁니다(원본은 `describe.jsonl.bak` 백업).

```bash
python src/evaluation/retranslate_gemini.py                       # 최신 sweep 전 모델
python src/evaluation/retranslate_gemini.py --models 4B,9B
python src/evaluation/retranslate_gemini.py --batch-size 100 --concurrency 4
```

번역을 **한 번의 API 호출에 여러 건 묶어**(배치) 보냅니다. 기본값은 **묶음 50개 × 동시 4묶음**(= 최대 200건 동시 진행). `--batch-size` 로 묶음 크기를 키울 수 있고(예: 100), 응답이 잘리거나 개수가 어긋나면 그 묶음만 자동으로 반씩 분할 재시도하므로 안전합니다. 재번역 후에는 **Web UI 서버를 재시작**해야 새 번역이 반영됩니다(메모리 캐시).

## 2단계: 분석 Web UI — `describe_server.py`

```bash
# 최신 sweep_labeled_* 자동 선택
python src/analysis/describe_server.py

# 폴더/포트 지정
python src/analysis/describe_server.py --path results/sweep_labeled_20260529_095220 --port 8801
```

기동하면 콘솔에 모델·건수가 출력됩니다. 브라우저에서 `http://localhost:8801` 로 접속합니다. (원격/헤드리스면 `ssh -L 8801:localhost:8801 <서버>` 후 접속)

### 화면 구성

- **상단**: 모델 선택 · 종류(FN/FP, 한 번에 하나) · 카테고리 필터 · 현재 위치(i/N). 칩마다 건수가 표시됩니다.
- **왼쪽**: 이미지(크게) + 배지(정답 라벨 / 모델 판정 / kind / 카테고리) + 틀린 맥락 한 줄 + **base(yes/no) 프롬프트 전문**.
- **오른쪽**: describe 질문(영어) + **모델 설명(한국어 기본)**. `[원문 EN 보기]` 토글 또는 `e` 키로 영어 원문 전환. 한국어 번역이 없으면 영어로 자동 폴백.
- **하단**: 썸네일 스트립(클릭 선택).

### 키보드

| 키 | 동작 |
|---|---|
| `←` / `→` | 이전 / 다음 케이스 |
| `Home` / `End` | 처음 / 끝 |
| `1`~`9` / `0` | 카테고리 선택 / 전체 |
| `f` / `p` | FN / FP 종류 전환 |
| `e` | 설명 한국어 ↔ 영어 원문 전환 |
| `[` / `]` | 이전 / 다음 모델 |

## 경로/포트 설정

`src/analysis/config.py` 에서 기본값을 바꿉니다.

| 상수 | 의미 | 기본 |
|---|---|---|
| `DEFAULT_PATH` | 분석 폴더 (None 이면 최신 `sweep_labeled_*` 자동) | `None` |
| `PORT` | Web UI 포트 (라벨링 8800 과 분리) | `8801` |

## 자주 보는 경우

- **FN 인데 description 에 사건이 잘 묘사됨** → 모델은 봤는데 base 프롬프트가 과하게 보수적(둔감). 프롬프트 완화 검토.
- **FN 인데 description 에서 사건을 못 봄/딴 걸 묘사** → 모델의 인식 능력 한계. 모델 교체/해상도 검토.
- **FP 인데 description 이 사건을 단정** → 무엇을 오인했는지 확인해 base 프롬프트의 제외 규칙 보강.
