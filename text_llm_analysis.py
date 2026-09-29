"""
Анализ текста звонка через LLM (YandexGPT, через OpenAI-совместимый API
Yandex AI Studio) — оценка соблюдения скрипта, отработки возражений,
вежливости и т.д.

Настройка (один раз):
    1. Зарегистрируйся в Yandex Cloud: https://cloud.yandex.ru
    2. Создай каталог (folder) в консоли, если ещё нет
    3. Создай API-ключ: https://console.cloud.yandex.ru -> сервисные аккаунты
       -> создать ключ доступа (или API-ключ для быстрого старта)
    4. В .env файле добавь:
        YANDEX_API_KEY=твой_ключ
        YANDEX_FOLDER_ID=твой_id_каталога
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from stt_diarization_pipeline import TranscriptSegment
from global_constants import DATA_DIR

load_dotenv(verbose=False)

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

client = OpenAI(
    api_key=YANDEX_API_KEY,
    base_url="https://ai.api.cloud.yandex.net/v1",
)

MODEL_URI = f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest"


# ---------------------------------------------------------------------------
# ПРОМПТ — системная инструкция с критериями оценки + few-shot пример
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Ты — опытный супервайзер отдела контроля качества call-центра.
Тебе дают транскрипт телефонного звонка с разметкой, кто говорит —
МЕНЕДЖЕР или КЛИЕНТ. Оцени работу МЕНЕДЖЕРА по следующим критериям:

1. script_adherence (1-5): соблюдение структуры звонка (приветствие,
   выявление потребности, презентация, работа с возражениями, закрытие)
2. objection_handling: было ли возражение от клиента, и если да — как
   менеджер его отработал (detected: true/false, quality: строка с оценкой,
   score: 1-5 или null если возражений не было)
3. forbidden_phrases: список грубых/непрофессиональных фраз, если есть
4. tone: общий тон менеджера одним-двумя словами (например "вежливый",
   "раздражённый", "уверенный")
5. overall_score (0-10): итоговая оценка звонка
6. justification: краткое обоснование оценки (2-3 предложения)

Отвечай СТРОГО в формате JSON, без markdown-разметки, без пояснений до или
после JSON — только сам объект.

Пример входа:
МЕНЕДЖЕР: Добрый день! Компания ТехноСтрой, меня зовут Анна. По вашей заявке
на кровельные материалы звоню.
КЛИЕНТ: Да, слушаю.
МЕНЕДЖЕР: Подскажите, какая площадь крыши?
КЛИЕНТ: 150 метров, но у меня бюджет ограничен.
МЕНЕДЖЕР: Понимаю, у нас как раз сейчас акция минус 10% — уложимся в ваш
бюджет с хорошим качеством.
КЛИЕНТ: Хорошо, давайте подробнее.

Пример ответа:
{"script_adherence": 5, "objection_handling": {"detected": true, "quality": "отработал через конкретное предложение (акция), а не просто скидку", "score": 4}, "forbidden_phrases": [], "tone": "уверенный", "overall_score": 8.5, "justification": "Менеджер представился, выявил потребность, грамотно отработал возражение по бюджету конкретным предложением, клиент заинтересован продолжить разговор."}
"""


def format_transcript_for_llm(segments: list[TranscriptSegment], manager_label: str) -> str:
    """Превращает список сегментов в читаемый текст для промпта."""
    lines = []
    for seg in segments:
        role = "МЕНЕДЖЕР" if seg.speaker == manager_label else "КЛИЕНТ"
        lines.append(f"{role}: {seg.text}")
    return "\n".join(lines)


def analyze_transcript_with_llm(segments: list[TranscriptSegment], manager_label: str) -> dict:
    transcript_text = format_transcript_for_llm(segments, manager_label)

    response = client.chat.completions.create(
        model=MODEL_URI,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript_text},
        ],
        temperature=0.2,  # низкая температура — нужна стабильная, воспроизводимая оценка
    )

    raw_output = response.choices[0].message.content.strip()

    # На случай, если модель всё же обернёт JSON в markdown-блок ```json ... ```
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`").removeprefix("json").strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        print("Не удалось распарсить JSON от LLM. Сырой ответ:")
        print(raw_output)
        raise


if __name__ == "__main__":
    from stt_diarization_pipeline import process_call

    segments, manager_label = process_call(audio_path=DATA_DIR + r"\crowd_test\dialogs\Phone_ARU_OFF.wav")
    result = analyze_transcript_with_llm(segments, manager_label)

    print("\n--- Текстовый анализ звонка ---\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))