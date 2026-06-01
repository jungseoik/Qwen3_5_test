"""오탐감소 평가용 카테고리별 VQA 프롬프트.

이 파일만 수정해서 프롬프트를 튜닝한 뒤 src/evaluation/fp_reduction.py 를 재실행한다.
키는 데이터셋 by_category/ 아래 폴더 이름과 일치해야 한다
(khonkaen: falldown, fire, smoke, violence).

각 프롬프트는 "진짜 이벤트면 yes, 아니면 no" 를 한 토큰으로 답하게 한다.
입력 데이터가 전부 오탐이므로, 모델이 "no" 라고 답할수록 오탐이 잘 걸러진 것이다.

fire / smoke / falldown 은 pe_vqa_2stage/validation_server/prompts.py 를 시작점으로
가져왔고, violence 는 동일한 톤으로 새로 작성했다.
"""

FIRE_PROMPT = (
    'Is there a real flame or active fire burning in this image?\n\n'
    'Answer "yes" if you see an actual flame or active combustion.\n\n'
    'Answer "no" if the image only contains:\n'
    '- lights, headlights, sunlight, or reflections\n'
    '- fire trucks, fire extinguishers, or fire-related equipment\n'
    '- smoke without a visible flame\n\n'
    'Answer only: "yes" or "no"'
)

# SMOKE_PROMPT = (
#     'Analyze the image carefully.\n\n'
#     'Task:\n'
#     'Determine whether the image shows real visible smoke in the scene.\n\n'
#     'Return "yes" only if there is actual smoke physically present in the environment, such as:\n'
#     '- a localized smoke plume\n'
#     '- rising, drifting, or spreading smoke\n'
#     '- smoke with a visible origin, direction, or shape\n'
#     '- white, gray, or black smoke emitted from a specific area or object\n\n'
#     'Return "no" for:\n'
#     '- lens fog\n'
#     '- condensation\n'
#     '- water droplets\n'
#     '- humidity haze\n'
#     '- camera blur\n'
#     '- dirty or smeared lens\n'
#     '- low contrast or washed-out frames\n'
#     '- glare or overexposure\n'
#     '- general misty appearance without a clear smoke source\n'
#     '- any ambiguous or uncertain case\n\n'
#     'Important rules:\n'
#     '- The smoke must be part of the real scene, not caused by the camera or lens condition.\n'
#     '- Do not classify smoke based on a globally foggy or low-visibility image alone.\n'
#     '- A localized smoke shape, direction, or source must be visible.\n'
#     '- If uncertain, return "no".\n\n'
#     'Answer only with:\n'
#     '"yes"\n'
#     'or\n'
#     '"no"'
# )
SMOKE_PROMPT = (
    'Analyze the image carefully.\n\n'
    'Task:\n'
    'Determine whether the image shows real visible smoke in the scene.\n\n'
    'Return "yes" only if there is actual smoke physically present in the environment, such as:\n'
    '- a localized smoke plume\n'
    '- rising, drifting, or spreading smoke\n'
    '- smoke with a visible origin, direction, or shape\n'
    '- white, gray, or black smoke emitted from a specific area or object\n\n'
    'Return "no" for:\n'
    '- lens fog\n'
    '- condensation\n'
    '- water droplets\n'
    '- humidity haze\n'
    '- camera blur\n'
    '- dirty or smeared lens\n'
    '- low contrast or washed-out frames\n'
    '- glare or overexposure\n'
    '- general misty appearance without a clear smoke source\n'
    '- clouds in the sky\n'
    '- natural fog, mist, sea fog, haze, or steam without a clear smoke source\n'
    '- any ambiguous or uncertain case\n\n'
    'Important rules:\n'
    '- The smoke must be part of the real scene, not caused by the camera or lens condition.\n'
    '- Do not classify smoke based on a globally foggy or low-visibility image alone.\n'
    '- Clouds in the sky are not smoke. Do not classify clouds as smoke unless there is a clear ground-level source emitting smoke.\n'
    '- A localized smoke shape, direction, or source must be visible.\n'
    '- If uncertain, return "no".\n\n'
    'Answer only with:\n'
    '"yes"\n'
    'or\n'
    '"no"'
)

FALLDOWN_PROMPT = (
    'Analyze the image and determine whether the person\'s situation should be classified as a fall.\n\n'
    'Focus primarily on the person\'s body posture, balance, support, and whether the pose appears '
    'controlled and intentional or collapsed and unintentional.\n\n'
    'Use the following rules:\n'
    '- Output `no` if the person appears to be standing, walking, sitting, crouching, squatting, '
    'kneeling, bending, reclining, or lying down in a controlled, stable, supported, or clearly '
    'intentional manner.\n'
    '- Output `yes` only if the person appears collapsed, sprawled, limp, tumbled, or unintentionally '
    'down due to loss of body control.\n'
    '- Do not mistake voluntary low postures or resting poses for a fall.\n\n'
    'Output requirements:\n'
    '- Return only one token.\n'
    '- Return `yes` for falldown.\n'
    '- Return `no` for normal.\n'
    '- Do not provide any explanation, description, JSON, punctuation, or additional text.'
)

# violence: 참조 prompts.py 에 없어 동일한 톤으로 신규 작성.
VIOLENCE_PROMPT = (
    'Analyze the image and determine whether real physical violence is happening.\n\n'
    'Answer "yes" only if there is an actual physical assault or fight, such as:\n'
    '- punching, kicking, hitting, or striking another person\n'
    '- grappling, choking, or pinning someone down aggressively\n'
    '- a clear act of physical aggression between people\n\n'
    'Answer "no" for:\n'
    '- people standing, walking, talking, or gathering normally\n'
    '- hugging, shaking hands, or other friendly contact\n'
    '- sports, exercise, or playful interaction\n'
    '- a single person with no aggressive contact\n'
    '- any ambiguous or uncertain case\n\n'
    'Answer only: "yes" or "no"'
)

PROMPTS = {
    "fire": FIRE_PROMPT,
    "smoke": SMOKE_PROMPT,
    "falldown": FALLDOWN_PROMPT,
    "violence": VIOLENCE_PROMPT,
}

# 위 맵에 없는 카테고리에 쓰는 기본 프롬프트.
DEFAULT_PROMPT = (
    "Look at this image carefully. Is there a real abnormal event or safety hazard? "
    "Answer only: \"yes\" or \"no\""
)


def get_prompt(category: str) -> str:
    return PROMPTS.get(category, DEFAULT_PROMPT)
