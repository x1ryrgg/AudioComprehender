"""Публичная точка входа в модуль текстового анализа: CallTranscript на входе,
структурированная оценка (TextAnalysis) на выходе."""

import json
from dataclasses import dataclass, field

from audio_comprehender.asr.types import CallTranscript
from audio_comprehender.text_analysis.llm_client import get_client, get_model_uri
from audio_comprehender.text_analysis.prompts import SYSTEM_PROMPT


@dataclass
class ObjectionHandling:
    detected: bool
    quality: str | None = None
    score: float | None = None


@dataclass
class TextAnalysis:
    script_adherence: int | None
    objection_handling: ObjectionHandling
    forbidden_phrases: list[str]
    tone: str | None
    overall_score: float | None
    justification: str
    raw: dict = field(default_factory=dict)  # сырой ответ LLM — на случай отладки/логов


def _parse_llm_json(raw_output: str) -> dict:
    raw_output = raw_output.strip()
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`").removeprefix("json").strip()
    return json.loads(raw_output)


def analyze_transcript_with_llm(transcript: CallTranscript) -> TextAnalysis:
    client = get_client()

    response = client.chat.completions.create(
        model=get_model_uri(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript.as_dialogue_text()},
        ],
        temperature=0.2,
    )

    raw_output = response.choices[0].message.content
    data = _parse_llm_json(raw_output)

    oh = data.get("objection_handling") or {}

    return TextAnalysis(
        script_adherence=data.get("script_adherence"),
        objection_handling=ObjectionHandling(
            detected=bool(oh.get("detected", False)),
            quality=oh.get("quality"),
            score=oh.get("score"),
        ),
        forbidden_phrases=data.get("forbidden_phrases") or [],
        tone=data.get("tone"),
        overall_score=data.get("overall_score"),
        justification=data.get("justification", ""),
        raw=data,
    )
