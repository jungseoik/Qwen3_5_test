# 라벨링 도구 구현 설계 (재현용)

이 문서만 보고 동일한 도구를 처음부터 재구현할 수 있도록 작성한 설계서다. 사용법은 [labeling_tool.md](labeling_tool.md) 를 본다.

## 목표

1차 탐지기가 카테고리별로 분류해 둔 썸네일이 **진짜 그 카테고리가 맞는지(True)/아닌지(False)** 사람이 키보드로 빠르게 검수하고, 결과를 `{category}/{true|false}/` 구조로 복사·재정렬한다. 외부 의존성 없이 **표준 라이브러리만** 사용한다 (프로젝트 전체 철학과 동일).

## 입력 / 출력 데이터 구조

입력 (읽기 전용):
```
{INPUT_ROOT}/by_category/{category}/thumbnail/{YYYYMMDD}/*.jpg
```
- 카테고리: `falldown`, `fire`, `smoke`, `violence` (스캔 시점에 동적으로 발견)
- `video/` 하위 mp4 는 대상이 아니다. `thumbnail/` 의 jpg 만 본다.

출력:
```
{OUTPUT_ROOT}/{SESSION_NAME}/
├── labels.json     # 라벨 상태(이어하기 원천)
└── export/         # Export 실행 시 생성
    └── export_{YYYYMMDD_HHMMSS}/   # 실행마다 새 폴더 (덮어쓰기 방지)
        ├── manifest.csv
        └── {category}/{true|false}/*.jpg
```

## 아키텍처 (3 파일 + 문서)

```
src/labeling/
├── config.py          # 경로 상수(INPUT_ROOT/OUTPUT_ROOT/SESSION_NAME). 사용자 편집 지점
├── label_server.py     # stdlib http.server 백엔드 + argparse CLI (메인)
└── static/index.html   # 단일 HTML(CSS/JS 인라인). 키보드 UI
```

단일 페이지 웹앱이다. 백엔드가 이미지 목록을 스캔해 JSON으로 내려주고, 브라우저 JS가 한 장씩 표시·키보드 입력·진행도를 처리하며, 라벨은 즉시 백엔드로 POST되어 `labels.json`에 저장된다. 이미지 바이트는 화면에 띄울 때만 `/api/image`로 스트리밍한다 (2,660장 전수를 메모리/브라우저에 올리지 않음).

### config.py

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = REPO_ROOT / "nas/data/khonkaen"   # by_category/ 포함 루트
OUTPUT_ROOT = REPO_ROOT / "results"            # 추후 여기만 바꿔 이동
SESSION_NAME = "labeling"                       # → {OUTPUT_ROOT}/labeling/
```
모든 CLI 기본값은 여기서 온다. CLI 인자(`--input/--output/--session/--category/--host/--port`)가 우선한다.

### label_server.py

핵심 구성요소:

1. **`collect_images(root, categories) -> (tasks, counts)`**
   `root/by_category/{cat}/thumbnail/` 아래 `rglob("*.jpg")`를 `sorted`로 모아 카테고리 순서대로 평탄화. 각 항목 `{"idx", "category", "rel"}`, `rel`은 `root` 기준 상대경로(POSIX). `fp_reduction.py:collect_images` (라인 68-82)와 동일한 스캔 규칙. `by_category/`가 없으면 `SystemExit`.

2. **`LabelStore`** — `labels.json`을 메모리에 들고 매 변경마다 디스크에 원자적 저장(`*.json.tmp` 작성 후 `replace`). 키 = 입력 루트 기준 상대경로(입력 절대경로가 바뀌어도 라벨 매칭 안정), 값 = `"true"`/`"false"`. 취소 시 키 삭제. `threading.Lock`으로 보호(ThreadingHTTPServer가 요청을 스레드로 처리하므로 필수).

3. **`export(input_root, store, export_base) -> {copied, skipped, out}`**
   매 호출마다 `export_base/export_{YYYYMMDD_HHMMSS}/`를 새로 만들어(덮어쓰기 방지) labels.json을 순회하며 `{category}/{label}/`로 `shutil.copy2` 원본 복사 + `manifest.csv` 작성. `category`는 `rel`의 두 번째 경로 조각(`by_category/{category}/...`)에서 추출. 원본 누락 파일은 `skipped` 카운트만 올리고 건너뜀. `out`은 생성된 타임스탬프 폴더 절대경로. `fp_reduction.py:collect_cases` (라인 341-368)의 복사+manifest 패턴 재사용.

4. **`make_handler(...)`** — 클로저로 상태(tasks/counts/store/export_dir/index.html 바이트)를 잡은 `BaseHTTPRequestHandler` 서브클래스를 반환. `log_message`를 오버라이드해 요청별 콘솔 로그를 끈다.

5. **`main()`** — argparse → `collect_images` → `LabelStore` → 배너 출력(입력/세션/총 이미지/이어하기 현황) → `ThreadingHTTPServer.serve_forever()`. 배너는 `sys.stdout.flush()`로 non-tty(로그 리다이렉트)에서도 즉시 보이게 한다.

### API 계약

| 메서드/경로 | 요청 | 응답 |
|---|---|---|
| `GET /` | — | `index.html` (text/html) |
| `GET /api/manifest` | — | `{total, counts:{cat:n}, labeled:{true,false,total}, items:[{idx,category,rel,label}]}` |
| `GET /api/image?idx=N` | — | 이미지 바이트 (`image/jpeg`, `Cache-Control: no-store`). 잘못된 idx → 404 |
| `POST /api/label` | `{idx, label}` (`label` ∈ `"true"`/`"false"`/`null`) | `{ok:true, labeled:{...}}` |
| `POST /api/export` | (본문 없음) | `{copied, skipped, out}` |

`label` 응답 예: `{"ok": true, "labeled": {"true": 2, "false": 1, "total": 3}}`

### labels.json 스키마

```json
{
  "by_category/fire/thumbnail/20260427/102_xxx.jpg": "true",
  "by_category/falldown/thumbnail/20260427/112_xxx.jpg": "false"
}
```
키 = 입력 루트 기준 상대경로, 값 = `"true"`/`"false"`. 취소된 항목은 키 자체가 없다.

### static/index.html

- **상태**: `items`(manifest), `cur`(현재 idx), `autoAdvance`, `cache`(idx→Image 프리페치).
- **부팅**: `/api/manifest` fetch → 첫 미라벨(`label==null`) 위치로 시작(이어하기). 전부 라벨됐으면 0.
- **렌더**: `img.src = /api/image?idx=cur`, 카테고리(좌상단 `#cat` + 이미지 위 중앙 `#catcenter` 큰 글씨)/위치/진행 막대/집계 갱신, 프레임 테두리색으로 라벨 표시(True=초록/False=빨강/미라벨=회색). `cur±1`, `cur+2` 프리페치.
- **라벨**: 낙관적 UI(즉시 화면 갱신) + 비동기 `POST /api/label`. True/False 입력 시 배지 플래시, `autoAdvance`면 다음 이미지로.
- **카테고리 점프**: `[`=현재 카테고리 시작(이미 시작이면 이전 카테고리 시작), `]`=다음 카테고리 첫 장.

### 키보드 스킴

| 키 | 동작 |
|---|---|
| `F` / `1` / `↑` | True 라벨 + (auto면) 다음 |
| `J` / `0` / `↓` | False 라벨 + (auto면) 다음 |
| `→` / `Space` | 다음 |
| `←` | 이전 |
| `Backspace` | 현재 라벨 취소 |
| `Home` / `End` | 처음 / 끝 |
| `[` / `]` | 이전 / 다음 카테고리 |
| `A` | 자동 다음 토글 |
| `E` | Export |

## 엣지케이스 처리 규칙

- **이미지 0장**: `main`에서 `SystemExit`. 프론트는 manifest가 비면 "이미지 없음" 표시.
- **빈 카테고리**: `thumbnail/`가 없으면 `counts[cat]=0`, tasks에 미포함.
- **중복 파일명**: 목적지가 `export_{타임스탬프}/{category}/{label}/`로 갈려 보통 충돌 없음. 같은 폴더 내 동명이 생기면 기존 파일을 `unlink` 후 덮어씀(마지막 라벨 우선).
- **재-Export**: 호출마다 새 타임스탬프 폴더가 생기므로 이전 export 결과를 덮어쓰지 않음(초 단위 해상도). labels.json은 그대로 유지되어 라벨 진행 상태와 무관.
- **이어하기**: 시작 시 `labels.json` 로드 → manifest의 `label` 필드로 복원. 서버 재시작/브라우저 새로고침해도 이어짐.
- **export 시 원본 누락**: NAS 마운트 해제 등으로 원본이 없으면 해당 항목만 `skipped`, 나머지는 정상 복사.
- **동시성**: `ThreadingHTTPServer` + `LabelStore.lock`으로 라벨 저장 경합 방지.

## 검증 체크리스트

1. 기동 시 콘솔에 총 이미지 수(카테고리 분포)와 접속 URL 출력.
2. `/api/manifest` 의 `total` == `find {input}/by_category -path '*/thumbnail/*' -name '*.jpg' | wc -l`.
3. `curl /api/image?idx=0` → JPEG.
4. 라벨 POST → `labels.json` 갱신, 취소 시 키 삭제.
5. 서버 재시작 후 manifest에 라벨 복원(이어하기).
6. Export → `{category}/{true,false}/`에 복사본 + `manifest.csv`, 장수가 라벨 집계와 일치.
