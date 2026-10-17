"""Объединение текстовой и эмоциональной оценок в единый балл консультации.

Веса и пороги — стартовые значения, требуют калибровки на размеченных
звонках (оценки эксперта приёмной комиссии vs оценки системы).
"""

from dataclasses import dataclass

from audio_comprehender.emotion.service import EmotionAnalysis
from audio_comprehender.text_analysis.parsing import TextAnalysis

W_TEXT = 0.7
W_EMOTION = 0.3
NEGATIVE_FLAG_THRESHOLD = 0.3


@dataclass
class ScoringResult:
    final_score: float | None
    flags: list[str]


def combine_scores(text: TextAnalysis, emotion: EmotionAnalysis) -> ScoringResult:
    text_score = text.overall_score
    emotion_score = emotion.score

    if text_score is not None and emotion_score is not None:
        final = W_TEXT * text_score + W_EMOTION * emotion_score
    else:
        final = text_score if text_score is not None else emotion_score

    flags = []

    if text.risky_statements:
        flags.append("Рискованные формулировки: " + "; ".join(text.risky_statements))

    if text.unanswered_questions:
        flags.append("Вопросы звонящего без ответа: " + "; ".join(text.unanswered_questions))

    if emotion.negative_share is not None and emotion.negative_share > NEGATIVE_FLAG_THRESHOLD:
        flags.append(f"Высокая доля негативных эмоций в голосе консультанта: {emotion.negative_share:.0%}")

    if emotion_score is None:
        flags.append("Оценка тона не рассчитана: нет достаточно надёжных реплик консультанта")

    return ScoringResult(
        final_score=round(final, 1) if final is not None else None,
        flags=flags,
    )
