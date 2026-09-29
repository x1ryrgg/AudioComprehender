"""
Единственная точка входа в ASR-модуль. Всё остальное в пакете asr/ —
детали реализации, наружу торчит только эта функция.
"""

from audio_comprehender.asr.diarization import diarize
from audio_comprehender.asr.transcription import transcribe_words
from audio_comprehender.asr.alignment import (
    assign_speakers_by_words,
    guess_manager_speaker,
    guess_manager_speaker_manual,
)
from audio_comprehender.asr.types import CallTranscript


def transcribe_call(audio_path: str, manager_speaker: str | None = None) -> CallTranscript:
    """
    Полный ASR-пайплайн: диаризация + распознавание + объединение.

    manager_speaker: передай "SPEAKER_00" / "SPEAKER_01", если роль известна
    заранее (надёжнее эвристики) — например, из метаданных звонка.
    """
    diarization_segments = diarize(audio_path)
    words = transcribe_words(audio_path)
    segments = assign_speakers_by_words(words, diarization_segments)

    if manager_speaker is not None:
        manager_label = guess_manager_speaker_manual(manager_speaker)
    else:
        manager_label = guess_manager_speaker(segments)

    return CallTranscript(segments=segments, manager_speaker=manager_label)
