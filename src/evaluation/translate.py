#!/usr/bin/env python3
"""Gemini 기반 영어→한국어 번역 모듈.

describe_eval.py 의 번역 단계에서 사용한다. 작은 vLLM 모델이 한국어를 직접 생성하면
글자가 깨지거나 어색해서, 영어 묘사는 vLLM 이 만들고 번역만 Google Gemini API 에 맡긴다.

프로젝트 철학에 맞춰 외부 SDK(google-genai) 없이 표준 라이브러리 urllib 로 REST 호출한다.
엔드포인트: POST {BASE}/models/{model}:generateContent?key={API_KEY}

키는 .env 의 GEMINI_API_KEY, 모델은 GEMINI_TRANSLATE_MODEL(없으면 기본값)에서 읽는다.

사용 예
    python src/evaluation/translate.py --selftest          # API 작동 확인
    python src/evaluation/translate.py "Gray smoke rises."  # 임의 문장 번역
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from fp_reduction import ENV_PATH, load_env

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-2.5-flash"

TRANSLATE_INSTRUCTION = (
    "Translate the following English text into natural, fluent Korean. "
    "Output only the Korean translation — no preamble, no notes, no original text. "
    "Preserve the meaning faithfully.\n\n"
    "English:\n{text}"
)


class TranslateError(RuntimeError):
    pass


class BatchMismatch(TranslateError):
    """배치 번역에서 보낸 개수 != 받은 개수 (정렬 깨짐/응답 잘림). 호출부에서 분할 재시도."""
    pass


def get_key_and_model(api_key=None, model=None):
    """인자 우선, 없으면 .env 에서 GEMINI_API_KEY / GEMINI_TRANSLATE_MODEL 읽기."""
    env = load_env(ENV_PATH)
    key = api_key or env.get("GEMINI_API_KEY") or ""
    mdl = model or env.get("GEMINI_TRANSLATE_MODEL") or DEFAULT_MODEL
    return key.strip(), mdl.strip()


def translate_to_korean(text: str, api_key: str, model: str = DEFAULT_MODEL,
                        timeout: int = 60, retries: int = 4) -> str:
    """영어 text 를 한국어로 번역해 반환. 실패 시 TranslateError.

    429(rate limit)/5xx/네트워크 오류는 지수 백오프로 재시도한다.
    """
    if not text or not text.strip():
        return ""
    if not api_key:
        raise TranslateError("GEMINI_API_KEY 가 없습니다 (.env 확인).")

    url = f"{API_BASE}/models/{model}:generateContent?key={api_key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": TRANSLATE_INSTRUCTION.format(text=text)}]}],
        "generationConfig": {"temperature": 0},
    }).encode()

    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
            return _extract_text(data)
        except urllib.error.HTTPError as e:
            code = e.code
            detail = e.read().decode("utf-8", "replace")[:300]
            last_err = f"HTTP {code}: {detail}"
            # 429/5xx 만 재시도, 그 외(400/403 등)는 즉시 실패
            if code != 429 and code < 500:
                raise TranslateError(last_err)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = f"{type(e).__name__}: {e}"
        # 백오프 (마지막 시도면 생략)
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise TranslateError(f"번역 실패(재시도 {retries}회): {last_err}")


BATCH_INSTRUCTION = (
    "You are a translator. The input is a JSON array of English texts. "
    "Translate each element into natural, fluent Korean. "
    "Return a JSON array of strings of the SAME length and order — element i must be the "
    "Korean translation of input element i. Do not merge, split, skip, or reorder elements. "
    "Output only the JSON array.\n\nInput:\n{payload}"
)


def translate_batch(texts, api_key: str, model: str = DEFAULT_MODEL,
                    timeout: int = 180, retries: int = 4,
                    max_output_tokens: int = 65536) -> list:
    """여러 영어 텍스트를 한 번의 API 호출로 한국어 번역 (JSON 구조화 출력).

    반환: 입력과 같은 길이·순서의 한국어 리스트 (빈 입력은 "" 유지).
    개수가 어긋나거나 잘리면 BatchMismatch (호출부에서 분할 재시도).
    """
    if not texts:
        return []
    if not api_key:
        raise TranslateError("GEMINI_API_KEY 가 없습니다 (.env 확인).")

    # 빈 텍스트는 호출에서 제외하고 자리만 보존
    idx_nonempty = [i for i, t in enumerate(texts) if t and t.strip()]
    if not idx_nonempty:
        return ["" for _ in texts]
    payload_texts = [texts[i] for i in idx_nonempty]

    url = f"{API_BASE}/models/{model}:generateContent?key={api_key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": BATCH_INSTRUCTION.format(
            payload=json.dumps(payload_texts, ensure_ascii=False))}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": {"type": "array", "items": {"type": "string"}},
            "maxOutputTokens": max_output_tokens,
        },
    }).encode()

    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
            raw = _extract_text(data)
            try:
                arr = json.loads(raw)
            except json.JSONDecodeError:
                raise BatchMismatch("JSON 파싱 실패 (응답 잘림 추정)")
            if not isinstance(arr, list) or len(arr) != len(payload_texts):
                raise BatchMismatch(
                    f"개수 불일치: 보냄 {len(payload_texts)} 받음 "
                    f"{len(arr) if isinstance(arr, list) else 'non-list'}")
            # 자리 복원
            out = ["" for _ in texts]
            for j, i in enumerate(idx_nonempty):
                out[i] = (arr[j] or "").strip()
            return out
        except BatchMismatch:
            raise  # 분할은 호출부 담당 (재시도해도 같은 결과)
        except urllib.error.HTTPError as e:
            code = e.code
            last_err = f"HTTP {code}: {e.read().decode('utf-8','replace')[:300]}"
            if code != 429 and code < 500:
                raise TranslateError(last_err)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise TranslateError(f"배치 번역 실패(재시도 {retries}회): {last_err}")


def translate_batch_safe(texts, api_key: str, model: str = DEFAULT_MODEL) -> list:
    """translate_batch + 폴백. 어긋나면 반으로 분할 재시도, 1건이면 단건 번역.

    한 묶음 전체가 실패해도 다른 항목은 살리고, 실패 항목만 `<error: ...>` 로 남긴다.
    """
    try:
        return translate_batch(texts, api_key, model)
    except BatchMismatch:
        pass
    except TranslateError:
        pass
    if len(texts) <= 1:
        if not texts:
            return []
        try:
            return [translate_to_korean(texts[0], api_key, model)]
        except TranslateError as e:
            return [f"<error: {e}>"]
    mid = len(texts) // 2
    return (translate_batch_safe(texts[:mid], api_key, model)
            + translate_batch_safe(texts[mid:], api_key, model))


def _extract_text(data: dict) -> str:
    """generateContent 응답에서 텍스트 추출. 안전차단/빈응답은 TranslateError."""
    cands = data.get("candidates") or []
    if not cands:
        fb = data.get("promptFeedback", {})
        raise TranslateError(f"candidates 없음 (blockReason={fb.get('blockReason')})")
    parts = (cands[0].get("content") or {}).get("parts") or []
    txt = "".join(p.get("text", "") for p in parts).strip()
    if not txt:
        raise TranslateError(f"빈 응답 (finishReason={cands[0].get('finishReason')})")
    return txt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", default=None, help="번역할 영어 문장")
    ap.add_argument("--selftest", action="store_true", help="고정 문장으로 API 작동 확인")
    ap.add_argument("--model", default=None, help="Gemini 모델 (생략 시 .env 또는 기본값)")
    ap.add_argument("--api-key", default=None, help="API 키 (생략 시 .env GEMINI_API_KEY)")
    args = ap.parse_args()

    key, model = get_key_and_model(args.api_key, args.model)
    print(f"모델: {model}  키: {'설정됨(' + key[:6] + '…)' if key else '(없음)'}", file=sys.stderr)

    sample = args.text
    if args.selftest or not sample:
        sample = ("Yes, real smoke is visible. The smoke appears white to light gray "
                  "and rises from a specific point near the rooftop, spreading upward.")
        if not args.text:
            print("[selftest] 샘플 문장으로 테스트합니다.", file=sys.stderr)

    try:
        t0 = time.perf_counter()
        out = translate_to_korean(sample, key, model)
        dt = time.perf_counter() - t0
        print(f"--- 영어 ---\n{sample}\n--- 한국어 ({dt:.2f}s) ---\n{out}")
    except TranslateError as e:
        raise SystemExit(f"[실패] {e}")


if __name__ == "__main__":
    main()
