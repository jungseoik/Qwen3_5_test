# 라벨 기반 평가 (오탐감소율 + 정탐률)

`src/evaluation/labeled_eval.py`는 사람이 검수한 라벨(true/false)이 붙은 이미지셋에 대해, 모델이 오탐을 얼마나 잘 거르는지(**오탐감소율**)와 진짜 사건을 얼마나 잘 유지하는지(**정탐률**)를 동시에 측정합니다. 기존 `fp_reduction.py`(모두 오탐 가정)와 출력 폴더 구조·옵션 패턴이 동일하며, 자동 저장·진행 바·서버 설정 기록·sweep 연동까지 그대로 호환됩니다.

## 개념

입력 이미지는 **1차 탐지기가 "이벤트"로 예측한 결과**이며, 사람이 검수해서 라벨을 붙입니다.

- `true/`  — 1차 예측이 맞음 (진짜 사건)
- `false/` — 1차 예측이 틀림 (오탐)

2차 VLM이 다시 yes/no로 판정하면 4가지 결과가 나옵니다.

| 라벨 | 모델 응답 | 의미 |
|------|----------|------|
| `false` | `no`  | **TN** — 오탐 잘 거름 ✓ |
| `false` | `yes` | **FP** — 오탐 놓침 (필터 실패) |
| `true`  | `yes` | **TP** — 정탐 잘 유지함 ✓ |
| `true`  | `no`  | **FN** — 진짜 사건 놓침 ⚠️ |

두 핵심 지표:

```
오탐감소율 = TN / (TN + FP)     # false 중 no 응답 비율 (specificity)
정탐률     = TP / (TP + FN)     # true 중 yes 응답 비율 (recall)
```

목표는 **둘 다 높이는 것** — 오탐만 줄이려다 진짜 사건을 놓치면 안 됨.

## 입력 폴더 구조

라벨링 도구(`src/labeling/`)가 출력하는 export 폴더를 그대로 사용합니다.

```
results/labeling/export/export_<YYYYMMDD_HHMMSS>/
├── manifest.csv                 # 라벨링 메타데이터
├── falldown/
│   ├── true/*.jpg               # 진짜 사건
│   └── false/*.jpg              # 오탐
├── fire/{true,false}/*.jpg
├── smoke/{true,false}/*.jpg
└── violence/{true,false}/*.jpg
```

날짜별 하위 구분 없이 라벨 폴더에 평면으로 모입니다.

## 실행

```bash
# 가장 최근 export 자동 선택 (전수, falldown+fire+smoke)
python src/evaluation/labeled_eval.py

# 특정 export 직접 지정
python src/evaluation/labeled_eval.py --path results/labeling/export/export_20260529_092418

# 카테고리/샘플 수 조정
python src/evaluation/labeled_eval.py --category fire,smoke --limit 50

# 결과 폴더 이름 직접 지정
python src/evaluation/labeled_eval.py --out results/eval_labeled_baseline
```

## 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--path` | `results/labeling/export/export_*` 최신 | 라벨 export 폴더 |
| `--category` | `falldown,fire,smoke` | 평가 카테고리 (violence 는 true 데이터 없음) |
| `--limit` | `0` (전수) | 카테고리/라벨당 샘플 수 |
| `--seed` | `0` | 샘플링 시드 |
| `--concurrency` | `.env` `MAX_NUM_SEQS` | 동시 요청 수 |
| `--max-tokens` | `1` | 응답 토큰 수 |
| `--url` | `http://localhost:8000` | 서버 주소 |
| `--model` | `.env` `VLLM_MODEL` → `/v1/models` | 모델 id 폴백 순서 |
| `--out` | `results/eval_labeled_<TS>/` | 결과 저장 폴더 |
| `--no-save` | off | 파일 저장 끄기 |
| `--collect` | `fp,fn` | 케이스 수집 모드. 항목 `tp/tn/fp/fn`, 특수값 `all`/`none` |
| `--symlink` | off | 수집 시 복사 대신 심볼릭 링크 |

## 결과 저장 구조

한 실행 = 한 폴더. `fp_reduction.py`와 동일 패턴, 폴더 prefix만 다름.

```
results/eval_labeled_20260529_092921/
├── eval.md            # 요약 + 서버 설정 + 사용 프롬프트
├── eval.csv           # 카테고리 × {total, pos, neg, TP, FN, TN, FP, 오탐감소율, 정탐률}
├── manifest.csv       # 수집된 케이스 메타데이터
├── fp/{category}/*.jpg   # 오분류: label=false → 모델 yes (필터 실패)
├── fn/{category}/*.jpg   # 오분류: label=true  → 모델 no  (정탐 놓침)
├── tp/{category}/*.jpg   # ✔ label=true → 모델 yes  (--collect 에 tp 포함 시)
└── tn/{category}/*.jpg   # ✔ label=false → 모델 no   (--collect 에 tn 포함 시)
```

수집 종류 4가지:

| 폴더 | 라벨 | 모델 응답 | 의미 | 기본 수집 |
|------|------|-----------|------|---|
| `tp/` | `true`  | `yes` | 정탐을 잘 유지 | (opt-in) |
| `fn/` | `true`  | `no`  | **정탐 놓침 ⚠️** | ✓ |
| `tn/` | `false` | `no`  | 오탐을 잘 거름 | (opt-in) |
| `fp/` | `false` | `yes` | **필터 실패** | ✓ |

기본은 실수 두 종류(`fp,fn`)만 모으고, 잘 맞춘 사례도 보고 싶으면 `--collect all` 또는 `--collect tp,fp,fn` 같이 명시. `tn`은 fire 카테고리 기준 1,500장 넘는 경우가 있어 디스크 부담이 큼 — 필요할 때만 켜기를 권장.

`manifest.csv` 컬럼: `kind, category, label, parsed, source, copied, raw_response`. `kind`는 `tp/tn/fp/fn` 중 하나.

## 출력 표 보는 법

```
=== 카테고리별 결과 ===
   카테고리   총수 true false   TP   FN    TN   FP  미파싱 에러   오탐감소율  정탐률
-----------------------------------------------------------------------------------
    smoke   164   73    91   53   20    76   15      0    0      83.5%   72.6%
-----------------------------------------------------------------------------------
     합계   164   73    91   53   20    76   15      0    0      83.5%   72.6%
```

- `오탐감소율 = TN / (TN+FP)` — 높을수록 오탐을 잘 거름
- `정탐률 = TP / (TP+FN)` — 높을수록 진짜 사건을 잘 유지
- `미파싱`/`에러`는 분모에서 제외됨 (`fp_reduction.py`와 동일 정책)
- 카테고리에 `true` 데이터가 없으면 정탐률은 `N/A` 표시

## sweep과 연동

`sweep.py`에 `--eval labeled` 플래그를 주면 여러 모델에 같은 라벨 평가를 반복하고, 모델 × 카테고리 매트릭스 두 개(오탐감소율, 정탐률)를 함께 출력합니다.

```bash
python src/evaluation/sweep.py --eval labeled
python src/evaluation/sweep.py --eval labeled --models 0.8B,9B --category fire,smoke
```

결과는 `results/sweep_labeled_<TS>/` 에 모이고, `summary.md`에 두 지표 표가 나란히 들어갑니다. 자세한 구조는 [sweep.md](sweep.md) 참고.
