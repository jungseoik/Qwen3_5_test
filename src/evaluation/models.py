"""모델 sweep 평가용 모델 리스트.

이 파일만 수정해서 어떤 모델을 어떤 순서로 평가할지 정한다.
캐시(~/.cache/huggingface)에 weight 가 없으면 sweep 이 자동 스킵한다.
값은 HuggingFace repo id 이며 .env 의 VLLM_MODEL 자리에 들어가는 값과 동일하다.
"""

MODELS = [
    "Qwen/Qwen3.5-0.8B",      # ~1.6GB
    "Qwen/Qwen3.5-2B",        # ~4GB
    "Qwen/Qwen3.5-4B",        # ~8GB
    "Qwen/Qwen3.5-9B",        # ~18GB
    "Qwen/Qwen3.5-27B",       # ~54GB
    "Qwen/Qwen3.5-35B-A3B",   # ~70GB  (MoE)
]
