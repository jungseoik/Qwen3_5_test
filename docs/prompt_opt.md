# 프롬프트 최적화 런북 (사람·에이전트 공통)

카테고리·모델별 yes/no 프롬프트를 반복 개선해 **정탐유지율**을 끌어올리는 절차입니다. 모든 단계가 동일한 CLI라서 **Claude(에이전트)가 하든 사람이 하든 똑같이** 재현됩니다. "판단"(오분류와 모델 설명을 읽고 다음 프롬프트 문구를 고치는 것)만 사람/에이전트가 하고, 평가·점수·기록은 CLI가 합니다.

## 배경

`describe` 진단을 보면 모델은 불·연기·낙상을 실제로는 잘 인식하는데(영어 묘사가 정확), base yes/no 프롬프트가 그 인식을 `no`로 떨어뜨려 FN(놓친 정탐)이 많습니다. 예: falldown 프롬프트의 "lying down → no" 규칙이 바닥에 쓰러진 진짜 낙상까지 배제. → **프롬프트 문구가 병목**이므로 문구를 고쳐 성능을 올립니다.

## 목표 지표

```
fp_reduction = TN/(TN+FP)   오탐감소율 (오탐 잘 거름)
retention    = TP/(TP+FN)   정탐유지율 (진짜 사건 유지)
```
**오탐감소율 ≥ 하한(기본 0.90)을 지키면서 정탐유지율 최대화.** `objective`가 이 규칙을 한 숫자로 표현하며, best = objective 최고 후보입니다.

## 전제

- 최적화할 그 모델로 vLLM 서버가 떠 있어야 합니다 (`docker compose up -d`, 모델은 `.env`/override). 반복 동안 **서버를 띄운 채** 후보만 바꿉니다(도커 재기동 없음).
- 라벨 export(`results/labeling/export/export_*`)가 있어야 합니다(labeled_eval 입력).
- 진단 근거로 쓰려면 그 모델의 `results/sweep_labeled_*/<Model>/describe/describe.jsonl`(한국어 설명)이 있으면 좋습니다(없어도 동작).

## 평가 셋

- **dev**(반복용): positive 전부 + negative 200개(seed 고정) → 빠르고 매번 동일. `eval` 기본.
- **full**(검증용): 전수. 최종 승자만 `--full`/`record`로 1회.

## 한 사이클 (카테고리 1개)

폴더: `results/prompt_opt/<Model>/<category>/`. 후보는 `vNN.txt`, 점수는 `vNN.json`.

### 1) 베이스라인 평가
```bash
python src/evaluation/prompt_opt.py eval \
  --model Qwen/Qwen3.5-27B --category falldown --baseline
```
→ `v00.json` 생성(현행 프롬프트의 dev 점수 + FN/FP 샘플 + describe 한국어 설명).

### 2) 진단 (사람/에이전트의 판단)
`v00.json`의 `samples.fn`(놓친 정탐)을 읽습니다. 각 항목: 이미지 경로, `base_verdict`(모델이 뭐라 했나), `desc_ko`(모델이 장면을 어떻게 봤나). 패턴을 찾습니다 — 예: "모델 설명은 '바닥에 쓰러져 있음'인데 base는 no" → 프롬프트의 어떤 규칙이 이걸 깎는지.

### 3) 후보 프롬프트 작성 → 평가
새 문구를 파일로 저장하고 평가:
```bash
# 예: 후보를 직접 편집기로 작성
$EDITOR /tmp/falldown_v1.txt
python src/evaluation/prompt_opt.py eval \
  --model Qwen/Qwen3.5-27B --category falldown --prompt-file /tmp/falldown_v1.txt --tag "lying-down 규칙 완화"
```
→ `v01.json`. dev 점수 비교.

### 4) 반복
개선되면 그 방향으로, 아니면 가설 수정해 v02, v03… 정지 기준(권장): 최대 6라운드 또는 2라운드 연속 개선 < 0.5%p.

### 5) 표로 보기
```bash
python src/evaluation/prompt_opt.py report --model Qwen/Qwen3.5-27B --category falldown
```
버전×지표 표와 현재 best 출력.

### 6) 최종 검증 + 기록
승자(예 v03)를 전수 검증하고 기록:
```bash
# 원인 분석을 markdown으로 적어두면 리포트에 삽입됨 (권장)
$EDITOR /tmp/falldown_why.md
python src/evaluation/prompt_opt.py record \
  --model Qwen/Qwen3.5-27B --category falldown --version v03 --notes-file /tmp/falldown_why.md
```
→ `record`가 하는 일:
- 승자와 베이스라인(v00)을 **전수**로 평가.
- `src/evaluation/optimized_prompts.json`에 `[모델][카테고리] = 최종 프롬프트` 기록 → 이후 labeled_eval/sweep/describe가 그 모델에 자동 사용.
- `best.txt`, `best.json`, `report.md`(베이스라인↔최종 지표·delta·원인·반복이력) 생성.

## 모델 순서

큰 모델부터: 27B → 35B-A3B → 9B → 4B → 2B → 0.8B. 모델을 바꿀 때만 서버 재기동:
```bash
docker compose down
VLLM_MODEL=Qwen/Qwen3.5-9B docker compose up -d   # 헬스 후 다음 모델 최적화
```
모델마다 falldown → smoke → fire 순. 모델별로 최적 프롬프트가 다른 게 정상입니다(레지스트리가 모델별로 보관).

## 주의

- **fire는 positive가 2건**뿐이라 retention 최적화는 통계적으로 약합니다. fp_reduction(오탐 안 깨짐) 유지선에서만 보고, 리포트에 캐비엇 표기.
- dev로 튜닝하므로 마지막 full 검증으로 과적합/실제 증가폭을 확인합니다.

## 명령 요약

| 단계 | 명령 |
|---|---|
| 베이스라인 | `prompt_opt.py eval --model M --category C --baseline` |
| 후보 평가 | `prompt_opt.py eval --model M --category C --prompt-file F [--tag T]` |
| 표 | `prompt_opt.py report --model M --category C` |
| 기록 | `prompt_opt.py record --model M --category C --version vNN [--notes-file N]` |
| 전수 단발 | 위 eval 에 `--full` 추가 |
