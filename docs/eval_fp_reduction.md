# 오탐감소율 평가

`src/evaluation/fp_reduction.py`는 카테고리별 오탐 썸네일을 vLLM 서버에 카테고리별 프롬프트로 보내 yes/no 2차 검증을 받고, **오탐감소율**을 측정합니다. 동시 요청으로 빠르게 처리하며, 결과를 화면과 파일(md/csv)로 출력합니다. 표준 라이브러리만 사용합니다.

## 개념

입력 데이터(`khonkaen/by_category/{category}/thumbnail/`)는 모두 1차 탐지기의 **오탐(false positive)**입니다. 따라서 정답은 전부 `no`(진짜 이벤트 아님)이며, VLM이 `no`라고 답할수록 오탐이 잘 걸러진 것입니다.

```
유효 응답 = no + yes
오탐감소율(%) = no / 유효 응답 × 100
```

`unparsed`(응답에서 yes/no 추출 실패)와 `error`(요청 실패)는 측정 노이즈이므로 **분모에서 제외**합니다. 두 카운트는 표에 별도로 표시되어 비정상 응답이 많을 경우 즉시 감지할 수 있습니다.

## 전제

- vLLM 서버가 떠 있어야 합니다 (기본 `http://localhost:8000`).
- 데이터가 `{path}/by_category/{category}/thumbnail/{date}/*.jpg` 구조여야 합니다.

## 서버 설정 자동 반영

평가는 떠 있는 서버를 대상으로 하므로, 모델과 동시 요청 수의 기본값은 **레포 루트의 `.env`** 에서 가져옵니다.

| 항목 | 출처 |
|---|---|
| 모델 | `--model` → `.env`의 `VLLM_MODEL` → `/v1/models` 자동 감지 (이 순서로 폴백) |
| 동시 요청 수 | `--concurrency` → `.env`의 `MAX_NUM_SEQS` → 16 |

또한 `MAX_NUM_BATCHED_TOKENS`, `GPU_MEMORY_UTILIZATION`, `KV_CACHE_MEMORY_BYTES`, `MAX_MODEL_LEN`, `EXTRA_VLLM_ARGS` 값을 헤더에 출력하고 `--out`으로 저장된 md 파일에도 함께 기록합니다. 같은 데이터를 서로 다른 서버 설정으로 평가했을 때 결과를 비교/재현하기 위해서입니다.

## 프롬프트 (편집해서 튜닝)

카테고리별 프롬프트는 `src/evaluation/prompts.py`에 정의되어 있습니다. **이 파일만 수정**하고 평가를 재실행하면 프롬프트 변경 효과를 비교할 수 있습니다.

- `fire`, `smoke`, `falldown` — `pe_vqa_2stage/validation_server/prompts.py`를 시작점으로 가져옴
- `violence` — 동일한 톤으로 신규 작성
- 맵에 없는 카테고리는 `DEFAULT_PROMPT` 사용

각 프롬프트는 "진짜 이벤트면 yes, 아니면 no"를 한 토큰으로 답하도록 작성되어 있습니다.

## 진행 표시

실행 중에는 stderr에 `\r`로 갱신되는 진행 바가 표시됩니다 (외부 의존성 없음).

```
[#############-----------------] 1024/2660  38.5%   42.1 img/s  ETA  38.9s
```

진행 바는 stderr이라 stdout으로 가는 결과 표·저장 파일과 분리되어 있어 `> result.txt`로 표만 리다이렉트해도 진행 바는 터미널에서 그대로 보입니다.

## 실행

```bash
# 전수 평가
python src/evaluation/fp_reduction.py

# 카테고리당 50장 샘플 (빠른 반복용)
python src/evaluation/fp_reduction.py --limit 50

# 특정 카테고리만, 동시성 지정
python src/evaluation/fp_reduction.py --category fire,smoke --concurrency 16

# 결과 경로 직접 지정 (md/csv 두 파일 생성)
python src/evaluation/fp_reduction.py --out results/my_run

# 파일 저장 끄기
python src/evaluation/fp_reduction.py --no-save
```

## 결과 저장

기본적으로 실행 결과를 **레포 루트 `results/` 아래 타임스탬프 폴더 하나에** 모아 저장합니다.

```
results/eval_20260528_115901/
├── eval.md            # 요약 + 서버 설정 + 사용 프롬프트
├── eval.csv           # 같은 데이터의 CSV
├── manifest.csv       # 수집된 실패 케이스 메타데이터
└── yes/{category}/*.jpg   # VLM 이 못 거른 오탐 썸네일 (--collect 모드에 따라)
```

- `results/` 가 없으면 자동 생성됩니다 (`.gitignore` 에 등록되어 커밋되지 않음).
- `--out FOLDER` 로 폴더명을 직접 지정할 수 있습니다 (예: `--out results/baseline_0.8B`).
- `--no-save` 를 주면 파일을 만들지 않고 화면 출력만 합니다.

`.md` 파일에는 일시·모델·데이터·샘플 수·동시성 + `.env` 의 서버 설정 7개 + 결과 표 + **사용한 카테고리별 프롬프트 전문**이 포함되어, 프롬프트나 서버 설정을 바꿔가며 측정한 이력을 비교/재현할 수 있습니다.

## 실패 케이스 자동 수집

평가에서 VLM이 못 걸러낸 오탐(yes 응답)의 **썸네일 원본을 결과 폴더로 자동 수집**합니다. 프롬프트 약점 분석이나 다음 fine-tuning 데이터 후보로 활용할 수 있습니다.

기본 동작: `yes` 응답만 수집, 원본 복사. 끄려면 `--collect none`, 모두 수집하려면 `--collect all`.

저장 구조 (한 실행 = 한 폴더):

```
results/eval_20260528_115901/
├── eval.md
├── eval.csv
├── manifest.csv                   # 수집 케이스 메타데이터
├── yes/                           # VLM이 yes로 답해 통과한 것
│   ├── falldown/*.jpg
│   ├── fire/*.jpg
│   ├── smoke/*.jpg
│   └── violence/*.jpg
├── unparsed/{category}/*.jpg      # (--collect 에 포함 시)
└── error/{category}/*.jpg         # (--collect 에 포함 시)
```

`manifest.csv` 컬럼: `category, parsed, source, copied, raw_response`. 원본 위치와 복사된 위치, 그리고 모델의 실제 응답 텍스트(앞 500자)까지 한 줄에 묶여 있어 어떤 응답으로 인해 잘못 분류됐는지 바로 확인할 수 있습니다.

NAS 마운트가 끊겨도 결과 폴더만으로 자기 완결되도록 **기본은 실제 복사**입니다. 디스크를 아끼려면 `--symlink` 로 심볼릭 링크 모드를 켤 수 있습니다.

## 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--path` | `nas/data/khonkaen` | 데이터셋 루트 (`by_category/` 포함) |
| `--category` | 전체 | 평가할 카테고리 (콤마 구분) |
| `--limit` | `0` (전수) | 카테고리당 샘플 수. 0이면 전수 |
| `--seed` | `0` | 샘플링 시드 (재현용) |
| `--concurrency` | `.env` `MAX_NUM_SEQS` (없으면 16) | 동시 요청 수 |
| `--max-tokens` | `1` | 응답 토큰 수 (yes/no면 1) |
| `--url` | `http://localhost:8000` | 서버 주소 |
| `--model` | `.env` `VLLM_MODEL` → `/v1/models` | 모델 id (위 우선순위로 폴백) |
| `--out` | `results/eval_{타임스탬프}/` | 결과 저장 폴더 (eval.md/eval.csv/manifest.csv/yes 가 그 안에 모임) |
| `--no-save` | off | 결과 파일을 저장하지 않음 (화면 출력만) |
| `--collect` | `yes` | 실패 케이스 썸네일 수집 모드. 콤마 구분(`yes,unparsed,error`), `all`, `none` |
| `--symlink` | off | 수집 시 복사 대신 심볼릭 링크 사용 |

## 출력 항목

| 열 | 의미 |
|---|---|
| `총수` | 보낸 이미지 수 |
| `유효` | `no + yes` (오탐감소율 계산의 분모) |
| `no(필터)` | VLM이 no로 답함 = 오탐 제거 성공 |
| `yes(통과)` | VLM이 yes로 답함 = 여전히 오탐으로 통과 |
| `미파싱` | 응답에서 yes/no를 못 뽑음 (분모 제외) |
| `에러` | 요청 실패 (분모 제외) |
| `오탐감소율` | `no / 유효 × 100` |

### 출력 예시

```
=== 오탐감소율 (정답=no, no일수록 좋음) ===
      카테고리     총수    no(필터)    yes(통과)     미파싱    에러     오탐감소율
--------------------------------------------------------------
  falldown      5         4          1       0     0     80.0%
      fire      5         5          0       0     0    100.0%
     smoke      5         4          1       0     0     80.0%
  violence      5         5          0       0     0    100.0%
--------------------------------------------------------------
        합계     20        18          2       0     0     90.0%
```

결과는 기본적으로 `results/` 폴더에 md/csv 한 쌍으로 자동 저장되며, 헤더에 모델·일시·샘플 수·서버 설정이 기록되어 프롬프트/모델 버전별 비교에 쓸 수 있습니다.
