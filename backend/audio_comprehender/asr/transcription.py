"""Speech-to-Text: пословное распознавание речи через faster-whisper.

Настройки VAD подобраны экспериментально (см. ablation в истории чата/диплома) —
мягкий VAD (threshold=0.3) даёт больше распознанных слов, чем настройки
по умолчанию. initial_prompt намеренно НЕ используется — эксперимент показал,
что он склеивает короткие реплики и теряет слова вроде "Да"/"Спасибо".
"""

from faster_whisper import WhisperModel

from audio_comprehender.config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

_model = None

_VAD_PARAMS = dict(threshold=0.3, speech_pad_ms=500, min_silence_duration_ms=500)


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    return _model


def transcribe_words(audio_path: str) -> list[tuple[float, float, str]]:
    """Возвращает список (start_sec, end_sec, word)."""
    model = _get_model()

    segments, info = model.transcribe(
        audio_path,
        language="ru",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=_VAD_PARAMS,
    )

    words = []
    for seg in segments:
        if seg.words:
            words.extend((w.start, w.end, w.word) for w in seg.words)
        else:
            # запасной вариант, если у сегмента почему-то нет пословных тайм-кодов
            words.append((seg.start, seg.end, " " + seg.text.strip()))

    print(f"[ASR] распознано слов: {len(words)}, язык: {info.language} ({info.language_probability:.2f})")
    return words
