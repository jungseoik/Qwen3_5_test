# 카테고리 True/False 이미지 라벨링 도구

`src/labeling/label_server.py`는 카테고리별로 분류된 썸네일이 **진짜 그 카테고리가 맞는지** 사람이 키보드로 빠르게 검수하고, 결과를 `{카테고리}/true|false/` 구조로 복사·재정렬하는 웹 기반 도구입니다. 외부 의존성 없이 표준 라이브러리만 사용합니다.

## 개념

입력 데이터(`khonkaen/by_category/{category}/thumbnail/`)는 1차 탐지기가 카테고리별로 분류해 둔 썸네일입니다. 이 분류가 실제로 맞으면 **True**, 잘못 분류됐으면 **False**로 라벨합니다. 라벨 결과는 카테고리별 `true`/`false` 폴더로 원본을 복사해 재정렬합니다.

```
입력  by_category/{category}/thumbnail/{date}/*.jpg
출력  export/{category}/{true|false}/*.jpg
```

라벨은 찍는 즉시 `labels.json`에 저장되어, 서버를 껐다 켜거나 브라우저를 새로고침해도 **이어서** 작업할 수 있습니다.

## 전제

- 데이터가 `{input}/by_category/{category}/thumbnail/{date}/*.jpg` 구조여야 합니다.
- 경로 기본값은 `src/labeling/config.py`에서 가져옵니다.

## 경로 설정

`src/labeling/config.py` 한 파일만 고치면 입력/출력 위치가 바뀝니다.

| 상수 | 의미 | 기본값 |
|---|---|---|
| `INPUT_ROOT` | `by_category/`를 포함하는 데이터셋 루트 | `nas/data/khonkaen` |
| `OUTPUT_ROOT` | 라벨 상태와 복사본을 둘 위치 | `results` |
| `SESSION_NAME` | 세션 폴더 이름 | `labeling` |

→ 실제 산출물은 `{OUTPUT_ROOT}/{SESSION_NAME}/` (기본 `results/labeling/`)에 모입니다. CLI 인자(`--input`/`--output`/`--session`)로 그때그때 덮어쓸 수도 있습니다.

## 실행

```bash
# config.py 기본 경로로 기동 (포트 8800)
python src/labeling/label_server.py

# 포트 지정
python src/labeling/label_server.py --port 8800

# 일부 카테고리만
python src/labeling/label_server.py --category fire,smoke

# 출력 위치를 그때만 변경
python src/labeling/label_server.py --output /mnt/work
```

기동하면 콘솔에 총 이미지 수와 접속 URL이 출력됩니다. 브라우저에서 그 주소(`http://localhost:8800`)로 접속하면 됩니다.

```
입력      : .../nas/data/khonkaen
세션 폴더 : .../results/labeling
총 이미지 : 2660장 (falldown:300, fire:1508, smoke:164, violence:688)

서버 시작 → http://localhost:8800  (Ctrl+C 종료)
```

### 헤드리스/원격 서버 접속

GPU 서버에 디스플레이가 없으면 SSH 포트포워딩으로 로컬 브라우저에서 접속합니다.

```bash
# 로컬 PC에서
ssh -L 8800:localhost:8800 <서버주소>
# 그 후 로컬 브라우저에서 http://localhost:8800
```

## 키보드 조작

라벨은 한 손으로 연속해서 찍을 수 있습니다. True/False를 누르면 자동으로 다음 이미지로 넘어갑니다(`A`로 토글).

| 키 | 동작 |
|---|---|
| `F` / `1` / `↑` | **True** (카테고리 맞음) + 자동 다음 |
| `J` / `0` / `↓` | **False** (맞지 않음) + 자동 다음 |
| `→` / `Space` | 다음 이미지 |
| `←` | 이전 이미지 |
| `Backspace` | 현재 라벨 취소 |
| `Home` / `End` | 처음 / 끝으로 점프 |
| `[` / `]` | 이전 / 다음 카테고리로 점프 |
| `A` | 라벨 후 자동 다음 토글 |
| `E` | Export 실행 |

화면 상단(좌측)과 **이미지 바로 위 중앙에 현재 카테고리가 크게** 표시되어, 빠르게 넘기면서도 지금 어떤 카테고리를 보고 있는지 바로 알 수 있습니다. 그 밖에 위치·전체 진행 막대·라벨 집계(T/F)가 표시되고, 이미지 테두리 색으로 현재 라벨(초록=True, 빨강=False, 회색=미라벨)을 즉시 확인할 수 있습니다.

## 이어하기

라벨은 찍는 즉시 `results/labeling/labels.json`(정확히는 `{OUTPUT_ROOT}/{SESSION_NAME}/labels.json`)에 저장됩니다. 서버를 껐다 켜거나 브라우저를 새로고침하면 **첫 미라벨 이미지부터** 다시 시작하며, 이미 찍은 라벨은 그대로 복원됩니다. 진행 중 언제든 멈췄다가 이어서 할 수 있습니다. 이 파일은 Export와 무관하게 유지되므로 Export를 여러 번 해도 라벨 진행 상태는 보존됩니다.

## Export (결과 재정렬 복사)

`E` 키 또는 하단 **Export** 버튼을 누르면 라벨된 이미지를 카테고리별 `true`/`false` 폴더로 원본 복사합니다. **실행할 때마다 타임스탬프 폴더(`export_YYYYMMDD_HHMMSS/`)를 새로 만들어** 기존 결과를 덮어쓰지 않습니다.

```
results/labeling/
├── labels.json                          # 라벨 상태 (이어하기 원천)
└── export/
    ├── export_20260529_091313/          # 1회차 export
    │   ├── manifest.csv                 # category, label, source, copied
    │   ├── falldown/
    │   │   ├── true/*.jpg               # 진짜 falldown
    │   │   └── false/*.jpg              # 오분류
    │   ├── fire/{true,false}/*.jpg
    │   ├── smoke/{true,false}/*.jpg
    │   └── violence/{true,false}/*.jpg
    └── export_20260529_104522/          # 2회차 export (이전 것 보존)
        └── ...
```

- 원본은 **복사**(보존)됩니다 — 원본 폴더는 건드리지 않습니다.
- `manifest.csv`에 원본 경로와 복사 위치가 기록됩니다.
- Export를 다시 실행하면 **새 타임스탬프 폴더**가 생기므로 이전 결과가 남습니다. 라벨을 더 찍은 뒤 Export하면 그 시점의 전체 라벨이 새 폴더로 복사됩니다. 완료 시 화면 하단에 생성된 폴더명이 표시됩니다.
- 출력 위치를 바꾸려면 `config.py`의 `OUTPUT_ROOT`(또는 `--output`)를 수정합니다.

## 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--input` | `config.INPUT_ROOT` | 데이터셋 루트 (`by_category/` 포함) |
| `--output` | `config.OUTPUT_ROOT` | 출력 루트 |
| `--session` | `config.SESSION_NAME` | 세션 폴더 이름 |
| `--category` | 전체 | 라벨링할 카테고리 (콤마 구분) |
| `--host` | `0.0.0.0` | 바인드 호스트 |
| `--port` | `8800` | 포트 |

구현 세부 사항과 재구현 가이드는 [labeling_tool_design.md](labeling_tool_design.md)를 참고하세요.
