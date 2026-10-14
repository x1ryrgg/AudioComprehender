"""Разбор ответа LLM в типизированный объект. Без тяжёлых зависимостей — легко тестировать."""

import json
from dataclasses import dataclass, field


@dataclass
class TextAnalysis:
    # критерии 1-5, None — критерий к разговору неприменим
    greeting_and_introduction: float | None
    needs_identification: float | None
    answer_quality: float | None
    empathy_and_politeness: float | None
    next_steps_and_closing: float | None
    unanswered_questions: list[str]
    risky_statements: list[str]
    facts_to_verify: list[str]      # для проверки человеком по правилам приёма
    tone: str | None
    overall_score: float | None     # 0-10
    justification: str
    raw: dict = field(default_factory=dict)   # сырой ответ LLM — для отладки


def _to_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value.replace(",", ".").strip())
        except ValueError:
            return None
    return None


def _to_str_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item]


def extract_json(raw_output: str) -> dict:
    """Достаёт JSON из ответа модели, даже если она обернула его в ```json ... ``` или добавила текст."""
    text = raw_output.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def parse_text_analysis(raw_output: str) -> TextAnalysis:
    data = extract_json(raw_output)
    if not isinstance(data, dict):
        raise ValueError(f"Ожидался JSON-объект, получено: {type(data).__name__}")

    overall = _to_number(data.get("overall_score"))
    if overall is not None:
        overall = min(10, max(0, overall))

    return TextAnalysis(
        greeting_and_introduction=_to_number(data.get("greeting_and_introduction")),
        needs_identification=_to_number(data.get("needs_identification")),
        answer_quality=_to_number(data.get("answer_quality")),
        empathy_and_politeness=_to_number(data.get("empathy_and_politeness")),
        next_steps_and_closing=_to_number(data.get("next_steps_and_closing")),
        unanswered_questions=_to_str_list(data.get("unanswered_questions")),
        risky_statements=_to_str_list(data.get("risky_statements")),
        facts_to_verify=_to_str_list(data.get("facts_to_verify")),
        tone=data.get("tone"),
        overall_score=overall,
        justification=str(data.get("justification", "")),
        raw=data,
    )
