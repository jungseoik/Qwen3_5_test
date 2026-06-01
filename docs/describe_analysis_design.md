# description 진단 파이프라인 + 분석 Web UI 구현 설계 (재현용)

이 문서만 보고 동일한 도구를 처음부터 재구현할 수 있도록 작성한 설계서다. 사용법은 [describe_analysis.md](describe_analysis.md) 를 본다.

## 목표

라벨 평가(`labeled_eval.py`)에서 모델이 yes/no 한 토큰만 답하므로 **"왜 그렇게 판정했는가"** 를 알 수 없다. 특히 진짜 사건을 `no` 로 걸러버리는 **FN(놓친 정탐)** 이 많아 정탐유지율이 떨어진다. 이를 진단하기 위해:

1. 평가가 이미 모아둔 오분류 목록(`manifest.csv`)을 입력으로, 각 오분류 케이스에 vLLM 모델이 장면을 **영어로 풀어 설명(description)** 하게 한 뒤 **Google Gemini API 로 한국어 번역** 한다 → 인식 실패 vs 프롬프트 과민/둔감 구분.
2. 그 결과를 모델별 · FN/FP별로 사람이 키보드로 빠르게 넘겨보며 base 프롬프트·정답·한국어 설명을 나란히 비교하는 Web UI 를 제공한다.

외부 의존성 없이 **표준 라이브러리만** 사용한다 (프로젝트 전체 철학과 동일 — Gemini 도 SDK 없이 `urllib` REST 로 호출). 작은 모델은 한국어를 직접 생성하면 글자가 깨지는 경우가 있어, **영어 묘사는 vLLM 이, 한국어 번역은 Gemini API 가** 담당한다 (UI 는 한국어를 기본 표시, 영어 원문은 토글로 확인). description 은 max_tokens 1 짜리 yes/no 평가와 달리 수백 토큰을 디코드하는 생성형 워크로드라, 서버 KV/동시성 설정을 `.env.describe` 오버레이로 그 실행에만 맞춘다.

## 입력 / 출력 데이터 구조

입력 (기존 평가 결과, 읽기 전용):
```
results/sweep_labeled_{TS}/{Model}/manifest.csv   # 컬럼: kind,category,label,parsed,source,copied,raw_response
```
- `kind` ∈ `fp/fn/tp/tn`, `label` ∈ `true/false`(정답), `parsed` ∈ `yes/no`(평가 당시 모델 판정), `source` = 원본 이미지 절대경로.
- 이 manifest 는 평가를 `--collect fp,fn`(기본)으로 돌리면 생성된다.

출력 (기존 폴더 안에 제자리 co-locate, 기존 파일 불변):
```
results/sweep_labeled_{TS}/{Model}/describe/
├── describe.jsonl   # 케이스당 1줄 (아래 스키마)
├── describe.csv     # 동일 내용 표 (description 내 개행은 공백 치환)
└── describe.md      # 모델·프롬프트·kind×카테고리 건수 요약
```

`describe.jsonl` 한 줄 스키마:
```json
{"id":"fn_smoke_0001","kind":"fn","category":"smoke","label":"true","base_verdict":"no",
 "source":"/abs/.../xxx.jpg",
 "description":"Gray smoke rises from the rooftop...","description_ko":"옥상에서 회색 연기가..."}
```
- `id` = `{kind}_{category}_{NNNN}` ((kind,category)별 0부터 4자리). 안정적이라 Web UI 가 이 키로 케이스를 조회한다.
- `base_verdict` = manifest 의 `parsed`(평가 당시 yes/no) 그대로.
- `description` = 영어 원문, `description_ko` = 한국어 번역. `--no-translate` 면 `description_ko` 는 빈 문자열.
- 이미지/요청 실패 시 `description` 에 `<error: ...>` 기록(번역 생략).

## 아키텍처 (파이프라인 2 + Web UI 3 + 프롬프트 + 문서)

```
.env / .env.example     # (수정) GEMINI_API_KEY / GEMINI_TRANSLATE_MODEL 추가 (.env 는 gitignore)
.env.describe           # (신규) describe 전용 서버 오버레이 (.env 위에 덮음)
src/evaluation/
├── prompts.py          # (수정) DESCRIBE_PROMPTS + get_describe_prompt() 추가 (번역은 translate.py 담당)
├── sweep.py            # (수정) compose() 에 optional extra_env 인자 (하위호환)
├── translate.py        # (신규) Gemini REST 번역 (단건 translate_to_korean + 배치 translate_batch_safe)
├── describe_eval.py    # (신규) 단일 모델: manifest 소비 → vLLM 영어 description → Gemini 한국어 번역 → describe/
├── describe_sweep.py   # (신규) 다중 모델: sweep.py 헬퍼 재사용, .env.describe 오버레이 주입 래핑
└── retranslate_gemini.py # (신규·일회성) 기존 describe 결과의 description_ko 만 Gemini 배치로 재생성
src/analysis/
├── config.py           # 경로/포트 상수 + find_latest_sweep(). 사용자 편집 지점
├── describe_server.py  # stdlib http.server 백엔드 + argparse CLI (메인)
└── static/index.html   # 단일 HTML(CSS/JS 인라인). 갤러리·상세·키보드 UI
```

의존 방향은 기존 패턴과 동일하다: `describe_eval` → `fp_reduction`/`prompts` import, `describe_sweep` → `sweep` import (모두 같은 `src/evaluation/` 폴더라 스크립트 실행 시 sys.path[0] 으로 해결).

### prompts.py (수정)

기존 `PROMPTS`/`get_prompt` 와 동일 구조로 `DESCRIBE_PROMPTS` dict + `get_describe_prompt(category)` 추가. 카테고리별 **영어** 묘사 프롬프트:
- **falldown**: 사람 수·자세(standing/sitting/lying/sprawled 등)·균형·지지·낙상 근거를 묘사.
- **fire**: 실제 불꽃/화염 유무와 위치, 불로 오인할 요소(lights·reflections·fire trucks·smoke without flame), 전반 색감.
- **smoke**: 연기 유무·색(black/white/gray)·출처·방향, clouds/fog/haze/lens blur 와의 구별, 전체 뿌염 vs 국소 연기.

공통: **보이는 것을 묘사하는 데만 집중** — 판정 단어(yes/no)나 출력 언어 지시는 넣지 않고, 길이 제한도 두지 않는다. 미정의 카테고리는 `DEFAULT_DESCRIBE_PROMPT`. 번역 프롬프트는 prompts.py 가 아니라 translate.py 안에 있다.

### translate.py (신규)

Google Gemini API 로 영어→한국어 번역. SDK 없이 `urllib` REST 호출(`{BASE}/models/{model}:generateContent?key=`).

- **`get_key_and_model(key=None, model=None)`** — `.env` 의 `GEMINI_API_KEY` / `GEMINI_TRANSLATE_MODEL`(없으면 기본 `gemini-2.5-flash`) 읽기.
- **`translate_to_korean(text, api_key, model, retries=4)`** — 단건 번역. 429/5xx/네트워크 오류는 지수 백오프 재시도, 4xx 영구오류는 즉시 `TranslateError`.
- **`translate_batch(texts, api_key, model, max_output_tokens=65536)`** — 여러 건을 한 호출로. `generationConfig.responseMimeType=application/json` + `responseSchema=array<string>` 로 받아 **보낸 개수 == 받은 개수** 검증, 어긋나거나 잘리면 `BatchMismatch`.
- **`translate_batch_safe(texts, ...)`** — `translate_batch` + 폴백: 어긋나면 묶음을 반으로 분할 재귀, 1건이면 단건 번역. 한 항목 실패가 다른 항목을 죽이지 않음(실패 항목만 `<error: ...>`).
- CLI `--selftest` 로 API 작동 확인.

### describe_eval.py (신규)

`fp_reduction` 에서 `build_payload, detect_model, load_env, Progress, ENV_PATH, SERVER_ENV_KEYS` 를, `prompts` 에서 `get_prompt(기록용), get_describe_prompt` 를 import 한다.

핵심 함수:
1. **`read_manifest(model_dir) -> [dict]`** — `{model_dir}/manifest.csv` 를 `csv.DictReader` 로 읽음. 없으면 `SystemExit`.
2. **`parse_kinds(s) -> set`** — `--kind` 콤마 파싱 (`fp/fn/tp/tn`, 특수값 `all`).
3. **`select_cases(rows, kinds, categories, limit) -> [case]`** — (kind,category) 필터 + (kind,category)별 limit. 각 case 에 `id`(=`{kind}_{cat}_{NNNN}`) 부여, `base_verdict`=manifest `parsed`.
4. **`_chat(url, payload) -> str`** — 단일 vLLM chat 요청 → content. 실패 시 예외.
5. **`one_request(url, model, case, max_tokens, translate, gemini_key, gemini_model)`** — (1) `build_payload` 로 vLLM 영어 description 생성 → `description`. (2) `translate` 면 `translate_to_korean(desc, gemini_key, gemini_model)` 로 Gemini 한국어 번역 → `description_ko`. 영어가 `<error`/빈값이면 번역 생략. 원본 누락은 `description="<error: 원본 이미지 없음>"`.
6. **`save(out_dir, ..., translate)`** — describe.jsonl / describe.csv(텍스트 컬럼 개행 치환) / describe.md. md 에 base+describe 프롬프트 전문과 kind×카테고리 건수표, 생성/번역 여부 기록.
7. **`main()`** — argparse → manifest 읽기 → case 선택 → 서버/모델/동시성 결정(.env) → Gemini 키 확인(없으면 번역 비활성+경고) → `ThreadPoolExecutor` 동시 요청(Progress 진행바) → id 정렬 후 save. 기본 저장 위치 `{from}/describe/`.

기본값: `--kind fp,fn`, `--max-tokens 512`(영어), `--no-translate` 로 번역 끄기, `--gemini-model` 로 모델 지정, `--concurrency`=`.env MAX_NUM_SEQS`. 영어 묘사 요청엔 `chat_template_kwargs.enable_thinking=False`.

> 단건 번역(`translate_to_korean`)을 case 마다 호출한다. 다음에 전체 sweep 을 새로 돌릴 때 번역을 더 빠르게 하려면, 영어를 전부 모은 뒤 `translate_batch_safe` 로 묶어 번역하도록 바꾸면 된다(현재는 일회성 retranslate_gemini.py 만 배치 사용).

### retranslate_gemini.py (신규·일회성)

기존 describe 결과의 `description`(영어)은 그대로 두고 `description_ko` 만 Gemini 로 재생성해 describe.jsonl/csv 를 덮어쓴다(원본은 `describe.jsonl.bak` 1회 백업). 영어 묘사를 재생성하지 않아 빠르다.

- targets 를 `--batch-size`(기본 50) 묶음으로 나눠 `--concurrency`(기본 4) 개씩 동시에 `translate_batch_safe` 로 번역. 어긋나거나 잘리면 묶음을 반으로 분할 재시도.
- 일회성 보정용이라 확장성은 고려하지 않는다.

### describe_sweep.py (신규)

`sweep` 에서 `compose, wait_healthy, parse_models, model_dirname, cache_has_weights` 를, `models` 에서 `MODELS` 를, `fp_reduction` 에서 `load_env` 를 import.

- **서버 오버레이** (`build_overlay(args)`): `.env.describe`(있으면) 를 `load_env` 로 읽고 CLI `--max-num-seqs`/`--max-model-len` 을 위에 덮어 dict 반환. 우선순위 CLI > `.env.describe` > (.env/compose 기본값). 이 dict 만 `.env` 를 덮고 나머지는 상속.
- **대상 결정**: `--models` 지정 시 그 집합, 아니면 `MODELS` 전체. 각 모델에 대해 `{from}/{model_dirname}/manifest.csv` 존재 + HF 캐시 weight 존재인 것만 `available`. 나머지는 사유별 스킵 로그.
- **클라이언트 동시성**: `--concurrency` 없으면 적용된 `MAX_NUM_SEQS`(오버레이→.env) 에 자동 일치시켜 서버 admission 과 맞춤.
- **루프(모델마다)**: `compose(["down"])` → `compose(["up","-d"], model_id, extra_env=overlay)` → `wait_healthy(url, load_timeout)` → `subprocess` 로 `describe_eval.py --from {model_dir} --model {id} --kind ... --max-tokens ... --concurrency N [--no-translate]` 실행. 마지막 모델은 띄워둠, 중간 모델은 `down`. `KeyboardInterrupt` 시 `down`.
- **출력**: 입력 sweep 폴더의 각 `{Model}/describe/` 가 채워짐 (별도 결과 폴더 없음).

`compose()` 는 `sweep.py` 에서 optional `extra_env: dict=None` 인자를 받도록 확장한다(기본 None → 기존 sweep 동작 그대로, 하위호환). `extra_env` 는 `os.environ` 복사본에 덮여 `docker compose` 자식 프로세스에 shell-env 로 전달된다(compose 우선순위 shell > .env).

### .env.describe (신규 오버레이 파일)

`.env` 와 같은 `KEY=VALUE` 형식. describe 실행에만 적용할 서버 설정만 적는다. 기본 제공값:
```
MAX_NUM_SEQS=32           # 128 → 32 (시퀀스당 KV 확보, 대형 모델 preempt 스래싱 방지)
MAX_MODEL_LEN=8192        # 4096 → 8192 (입력 image토큰 + 출력 여유)
MAX_NUM_BATCHED_TOKENS=16384
```
여기 없는 키(GPU_MEMORY_UTILIZATION 등)는 `.env` 를 그대로 상속한다. 파일을 지우거나 비우면 `.env` 기본값으로 동작(하위호환). `.env` 파일 자체는 절대 수정하지 않는다.

### config.py (analysis)

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_PATH = None            # 생략 시 find_latest_sweep() 사용
PORT = 8801                    # 라벨링 8800 과 충돌 회피
def find_latest_sweep(base=RESULTS_DIR):  # results/sweep_labeled_* 중 최신
```

### describe_server.py

`config.REPO_ROOT/src/evaluation` 를 `sys.path` 에 넣어 `prompts.get_prompt/get_describe_prompt` 를 단일 진실로 import (프롬프트를 md 에서 재파싱하지 않음).

1. **`load_results(root) -> {model: {cases:{id:rec}, order:[id]}}`** — `root` 하위 각 모델 폴더의 `describe/describe.jsonl` 을 읽어 메모리 적재. jsonl 없으면 그 모델 제외.
2. **`build_index(results) -> meta`** — 모델 목록(각 `{name, categories, counts:{kind:{cat:n}}, total}`), 전체 카테고리, `kind_label`, 카테고리별 `base_prompts`/`describe_prompts`.
3. **`make_handler(root, results, index_meta)`** — 클로저로 상태를 잡은 `BaseHTTPRequestHandler`. `_case(qs)` 로 (model,id) → record 조회. `log_message` 오버라이드로 콘솔 로그 끔.
4. **`main()`** — argparse(`--path/--host/--port`) → path 없으면 `find_latest_sweep` → `load_results` (비면 SystemExit) → `build_index` → 배너 출력 → `ThreadingHTTPServer.serve_forever()`.

### API 계약

| 메서드/경로 | 요청 | 응답 |
|---|---|---|
| `GET /` | — | `index.html` (text/html) |
| `GET /api/index` | — | `{path, models:[{name,categories,counts,total}], categories:[..], kind_label, base_prompts:{cat:..}, describe_prompts:{cat:..}}` |
| `GET /api/cases?model=&kind=&category=` | `category` 비거나 `__all__` 이면 전체 | `{model, items:[{id,category,kind,label,base_verdict}]}` (order 보존) |
| `GET /api/case?model=&id=` | — | jsonl record(`description`+`description_ko` 포함) + `base_prompt` + `describe_prompt`. 없으면 404 |
| `GET /img?model=&id=` | — | 이미지 바이트(`image/jpeg`, `no-store`). record 의 `source` 절대경로 서빙, 없으면 404 |

이미지는 사용자 입력 경로가 아니라 **우리 manifest 가 만든 신뢰된 `source`** 로만 서빙하므로 경로탈출 위험이 없다 (id→record→source 매핑).

### static/index.html

- **상태**: `META`(index), `state={model,kind,category}`, `items`(현재 목록), `cur`(인덱스).
- **부팅**: `/api/index` → 모델 탭/종류 칩/카테고리 칩 렌더 → 첫 모델·기본 kind(FN 우선)·전체 카테고리로 `loadCases`.
- **종류(kind) 노출**: 모델별 `counts` 에 존재하는 kind 만 `fn,fp,tp,tn` 순서로 칩 표시. 칩에 건수 배지.
- **카테고리 칩**: `전체(__all__)` + 전체 카테고리. 칩마다 현재 (model,kind) 기준 건수 갱신.
- **`loadCases`**: `/api/cases` → `items`, `cur=0`, 썸네일 스트립 렌더 → `loadCase`. 비면 안내 표시.
- **`loadCase`**: `/img` 로 큰 이미지, `/api/case` 로 상세. 왼쪽 = 이미지+배지(정답/모델판정/kind/카테고리)+kind 해설+base 프롬프트, 오른쪽 = describe 프롬프트+모델 설명(`white-space:pre-wrap`). 썸네일 현재 항목 하이라이트+스크롤.
- **설명 언어**: `renderDesc()` 가 한국어(`description_ko`) 기본 표시, 한국어가 없거나 `<error` 면 영어(`description`) 폴백. `[원문 EN 보기]` 토글(클릭 또는 `e` 키)로 한↔영 전환.
- **레이아웃**: 좌(이미지·base) / 우(description) 분할, 하단 썸네일 스트립.

### 키보드 스킴

| 키 | 동작 |
|---|---|
| `←` / `→` | 이전 / 다음 케이스 |
| `Home` / `End` | 처음 / 끝 |
| `1`~`9` | 카테고리 선택(목록 순서) |
| `0` | 전체 카테고리 |
| `f` | FN 종류 |
| `p` | FP 종류 |
| `e` | 설명 한국어 ↔ 영어 원문 전환 |
| `[` / `]` | 이전 / 다음 모델 |

## 엣지케이스 처리 규칙

- **manifest 없음**: `describe_eval` 가 `SystemExit` (평가를 `--collect fp,fn` 로 먼저 돌리라 안내).
- **대상 case 0건**: `describe_eval` `SystemExit`. Web UI 는 해당 조건이면 "케이스 없음" 안내.
- **원본 이미지 누락**: `describe_eval` 은 `description="<error: 원본 이미지 없음>"` 로 기록(중단 X). 서버 `/img` 는 404.
- **describe.jsonl 없는 모델**: `load_results` 가 제외 → Web UI 모델 목록에 안 뜸.
- **kind 가 일부만 존재**: 모델별 실제 존재 kind 만 칩 노출. 기본 kind 는 FN 우선, 없으면 첫 존재 kind.
- **재실행**: `describe/` 만 덮어씀. 기존 `manifest.csv`/`eval.*`/`fp`/`fn` 은 불변.
- **동시성**: 생성은 `ThreadPoolExecutor`(요청), 서버는 `ThreadingHTTPServer`. 서버는 읽기 전용이라 lock 불필요.
- **포트 충돌**: 기본 8801 (라벨링 8800 과 분리).

## 검증 체크리스트

1. `describe_eval.py --help` / `describe_sweep.py --help` 가 에러 없이 출력 (argparse·import 정상).
1b. `python src/evaluation/translate.py --selftest` 로 Gemini API 작동(키·모델) 확인.
2. (서버 기동 후) 단일 실행 → `{Model}/describe/describe.jsonl` 생성, 줄 수 == manifest 의 대상 kind 건수.
3. `description`(영어)+`description_ko`(한국어)가 채워지고 카테고리별 관점(사람/장면/연기 색·구름)이 드러남.
4. `describe_sweep` → 각 모델 `describe/` 채워짐, compose up 로그에 오버레이 값(MAX_NUM_SEQS 등) 표시, 모델 교체·헬스체크 정상.
5. `describe_server --path <sweep_labeled>` → 배너에 모델·건수 출력, `GET /api/index` 200.
6. Web UI: 모델 전환(`[`/`]`), FN/FP 토글(한 종류씩), 카테고리 필터, `←/→` 넘김, 이미지 서빙, 좌 base 프롬프트/우 한국어 description 표시.
