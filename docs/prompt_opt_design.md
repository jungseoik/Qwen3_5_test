# 프롬프트 최적화 파이프라인 구현 설계 (재현용)

이 문서만 보고 동일 도구를 재구현할 수 있도록 작성한 설계서다. 사용법은 [prompt_opt.md](prompt_opt.md).

## 목표

카테고리·모델별 yes/no 프롬프트를 반복 최적화한다. describe 진단상 모델은 사건을 인식하나 base 프롬프트가 `no`로 떨어뜨려 FN 이 많다(정탐유지율 하락). 프롬프트 문구를 고쳐 **오탐감소율 하한을 지키며 정탐유지율을 최대화**한다. 판단(프롬프트 작성)만 사람/에이전트가 하고 평가·점수·기록은 CLI(=Claude/인간 공통).

## 아키텍처 (추가 위주 + labeled_eval 소폭 리팩터)

```
src/evaluation/
├── labeled_eval.py        # (수정) --prompt-file, --max-neg, prompt_by_cat 주입, get_prompt(cat,model)
├── prompts.py             # (수정) get_prompt(category, model=None) + optimized_prompts.json 로더
├── optimized_prompts.json # (신규·커밋) {model: {category: prompt}} 모델별 최적 프롬프트
└── prompt_opt.py          # (신규) 오케스트레이터: eval / report / record
results/prompt_opt/<Model>/<category>/   # (gitignore) vNN.txt, vNN.json, v00_full.json, best.*, report.md
docs/prompt_opt.md, prompt_opt_design.md # 런북 / 설계
```

평가 엔진은 새로 만들지 않고 labeled_eval.py 를 subprocess 로 재사용한다. 떠 있는 vLLM 서버에 후보 프롬프트만 바꿔 보내므로 반복에 도커 재기동이 없다.

### labeled_eval.py 변경점

- `collect_images(root, categories, limit, seed, max_neg=0)`: `max_neg>0` 이면 positive(true) 전부 + negative(false) 만 `max_neg` 캡(seed 고정). dev 서브셋 모드. 정탐 표본 보존이 핵심(retention 측정).
- `one_request(..., prompt)`: 프롬프트 문자열을 인자로 받음(과거 내부 `get_prompt` 호출 제거).
- `main`: `--prompt-file`(있으면 그 텍스트를 --category 의 모든 카테고리에 적용), 없으면 `get_prompt(cat, model)`. `prompt_by_cat` dict 구성 후 one_request 에 전달. eval.md 에 실제 사용 프롬프트 기록.
- 산출물(eval.csv/manifest.csv)은 그대로. prompt_opt 가 이를 읽는다.

### prompts.py 변경점

- `get_prompt(category, model=None)`: model 의 최적 프롬프트가 `optimized_prompts.json` 에 있으면 그것, 없으면 baseline `PROMPTS[category]`. → record 후 labeled_eval/sweep/describe 가 자동으로 최적 프롬프트 사용.
- `OPTIMIZED_PROMPTS = _load_optimized()` (파일 없으면 `{}`).

### prompt_opt.py (오케스트레이터)

지표 계산:
```
fp_reduction = TN/(TN+FP);  retention = TP/(TP+FN)
objective = retention                              (fp_reduction >= floor)
          = retention - (floor - fp_reduction)*W   (미달, W=10)
best = objective 최대, 동률이면 fp_reduction 높은 쪽
floor 기본 0.90 (--floor)
```

서브커맨드:
- **eval**: 프롬프트(=`--baseline`→`PROMPTS[cat]`(v00) 또는 `--prompt-file`)를 `vNN.txt` 로 저장 → `run_labeled_eval` 로 dev(기본, `--max-neg`) 또는 `--full` 평가 → eval.csv 의 카테고리 행에서 counts → `score()` → `vNN.json` 저장. JSON 에 `samples.fn/fp`(최대 8건): manifest 의 source·parsed·raw + 그 이미지의 describe 한국어/영어 설명(같은 모델 describe.jsonl 에서 basename 매칭). 버전은 `next_version`(v00 은 baseline 고정).
- **report**: 해당 (model,category) 의 dev `vNN.json` 들을 표로(fp_reduction/retention/objective/floor) + `pick_best`.
- **record**: 승자 버전과 v00 을 `ensure_full`(전수 평가, 결과 캐시) → `optimized_prompts.json[model][category] = 승자 프롬프트` → `best.txt/json` → `report.md`(전수 baseline↔최종 지표·delta, 원인분석(`--notes-file` 삽입), 베이스/최종 프롬프트 전문, dev 반복 이력, positive≤3 캐비엇).

`run_labeled_eval(model, category, prompt_file, run_dir, full, max_neg, url, seed)`:
labeled_eval 를 `--category <c> --model <m> --prompt-file <txt> --out <run_dir> --collect fp,fn --symlink --seed S --max-tokens 1 [--max-neg N(=dev)]` 로 호출 → (category row, manifest rows) 반환.

### 데이터 구조 — vNN.json

```json
{"model","category","version","tag","set":"dev|full","max_neg",
 "prompt_file","prompt","ts",
 "counts":{"tp","fn","tn","fp","unparsed","error"},
 "positives","negatives","fp_reduction","retention","objective","passes_floor","floor",
 "samples":{"fn":[{"image","base_verdict","raw","desc_ko","desc_en"}],"fp":[...]}}
```

## dev / 검증

- dev: positive 전부 + negative `--max-neg`(기본 200), seed=0 → 매 반복 동일.
- full: 전수. 승자만. record 가 baseline·최종 모두 full 로 재서 delta 산출.
- 한계: positive 가 희소해(falldown 13, fire 2) holdout 불가 → retention 은 dev·full 동일 positive 로 측정. negative 는 dev 샘플/full 전수라 fp_reduction 일반화는 검증됨.

## 모델 순서 / 운용

큰 모델 먼저(27B→35B-A3B→9B→4B→2B→0.8B). 한 모델 띄운 채 카테고리(falldown→smoke→fire) 반복, 모델 교체 시에만 `docker compose down && VLLM_MODEL=<m> docker compose up -d`. 결과는 모델별 레지스트리에 분리 보관(모델마다 최적 프롬프트 상이).

## 검증 체크리스트

1. `labeled_eval.py --help` 에 `--prompt-file/--max-neg`, `prompt_opt.py {eval,report,record} --help` 정상.
2. `get_prompt("smoke")` == baseline; record 후 `get_prompt("smoke", "<model>")` == 기록된 프롬프트.
3. (서버 기동) `eval --baseline` → v00.json 에 counts + samples(describe 설명) + dev 건수(positive 전부 + max_neg).
4. `eval --prompt-file` 반복 → vNN.json 누적, `report` 표/ best 정확.
5. `record --version vNN` → optimized_prompts.json 갱신, report.md 에 baseline↔최종 delta·원인·프롬프트 전문.
