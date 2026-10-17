"""
Публичная точка входа в модуль эмоций: транскрипт менеджера + путь к аудио
на входе, эмоциональная траектория и агрегированная оценка тона на выходе.
"""

from collections import Counter
from dataclasses import dataclass, field

import soundfile as sf

from audio_comprehender.asr.types import CallTranscript
from audio_comprehender.emotion.model import SERScorer, extract_segment_waveform

MIN_SEGMENT_SEC = 0.5   # реплики короче SER не оценивает (ненадёжно)
MIN_CONFIDENCE = 0.5    # предсказания с меньшей уверенностью не учитываем

# Насколько "плохой" каждая эмоция в речи менеджера, для расчёта итогового балла
NEGATIVE_WEIGHTS = {"angry": 1.0, "sad": 0.5}


@dataclass
class EmotionPoint:
    start: float
    end: float
    text: str
    emotion: str
    confidence: float


@dataclass
class EmotionAnalysis:
    score: float | None                  # 0-10, None если надёжных данных не было
    negative_share: float | None         # доля "плохих" эмоций по времени
    distribution: dict[str, float]       # доля времени на каждую эмоцию
    timeline: list[EmotionPoint] = field(default_factory=list)


def analyze_manager_emotions(audio_path: str, transcript: CallTranscript, ser: SERScorer) -> EmotionAnalysis:
    full_audio, sr = sf.read(audio_path, dtype="float32")
    if full_audio.ndim > 1:
        full_audio = full_audio.mean(axis=1)

    timeline: list[EmotionPoint] = []
    for seg in transcript.manager_segments():
        duration = seg.end - seg.start
        if duration < MIN_SEGMENT_SEC:
            continue

        probs = ser.predict(extract_segment_waveform(full_audio, sr, seg.start, seg.end), sr)
        emotion = max(probs, key=probs.get)

        timeline.append(EmotionPoint(
            start=round(seg.start, 1),
            end=round(seg.end, 1),
            text=seg.text,
            emotion=emotion,
            confidence=round(probs[emotion], 3),
        ))

    reliable = [p for p in timeline if p.confidence >= MIN_CONFIDENCE]
    total_duration = sum(p.end - p.start for p in reliable)

    if total_duration == 0:
        return EmotionAnalysis(score=None, negative_share=None, distribution={}, timeline=timeline)

    negative = sum(NEGATIVE_WEIGHTS.get(p.emotion, 0.0) * (p.end - p.start) for p in reliable)
    negative_share = negative / total_duration

    by_emotion = Counter()
    for p in reliable:
        by_emotion[p.emotion] += p.end - p.start

    return EmotionAnalysis(
        score=round(10 * (1 - negative_share), 1),
        negative_share=round(negative_share, 3),
        distribution={e: round(d / total_duration, 3) for e, d in by_emotion.items()},
        timeline=timeline,
    )
