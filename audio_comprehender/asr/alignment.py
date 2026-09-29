"""
Объединение результатов диаризации и распознавания речи:
  - какому говорящему принадлежит каждое слово
  - слова -> реплики
  - какая из меток SPEAKER_00/01 — менеджер (эвристика, см. ограничения ниже)
"""

from audio_comprehender.asr.types import TranscriptSegment

MANAGER_KEYWORDS = [
        "компания",
        "меня зовут",
        "чем могу помочь",
        "оператор",
        "служба",
        "добрый день, вы позвонили",
        "стоимость составит",
        "стоимость поездки",
        "с вами говорит",
    ]

CLIENT_KEYWORDS = [
    "это вызов такси?",
    "хочу узнать",
    "подскажите",
    "сколько будет стоить",
    "мне нужно",
    "я звоню по поводу",
    "у меня вопрос",
]


def _speaker_for_interval(start, end, diarization_segments):
    best_speaker, best_overlap = None, 0.0
    for speaker, d_start, d_end in diarization_segments:
        overlap = max(0.0, min(end, d_end) - max(start, d_start))
        if overlap > best_overlap:
            best_speaker, best_overlap = speaker, overlap

    if best_speaker is None and diarization_segments:
        mid = (start + end) / 2
        best_speaker = min(
            diarization_segments,
            key=lambda s: min(abs(mid - s[1]), abs(mid - s[2])),
        )[0]

    return best_speaker or "UNKNOWN"


def assign_speakers_by_words(
    words: list[tuple[float, float, str]],
    diarization_segments: list[tuple[str, float, float]],
    max_gap: float = 1.5,
) -> list[TranscriptSegment]:
    """
    Каждому слову назначаем говорящего по тайм-коду (максимум перекрытия
    с сегментом диаризации), затем склеиваем подряд идущие слова одного
    говорящего в реплику.

    ИЗВЕСТНОЕ ОГРАНИЧЕНИЕ: на моно-записи с короткими репликами (<1-2с)
    диаризация может "поглощать" их соседними длинными сегментами — точность
    пословного назначения не выше точности самой диаризации на таком аудио.
    В проде эта проблема решается двухканальной записью звонков.
    """
    turns: list[TranscriptSegment] = []

    for start, end, word in words:
        speaker = _speaker_for_interval(start, end, diarization_segments)

        if turns and turns[-1].speaker == speaker and start - turns[-1].end <= max_gap:
            turns[-1].text += word
            turns[-1].end = end
        else:
            turns.append(TranscriptSegment(speaker=speaker, start=start, end=end, text=word))

    for t in turns:
        t.text = t.text.strip()

    return turns


def guess_manager_speaker(segments: list[TranscriptSegment]) -> str:
    """
    Эвристика: ищем характерные фразы менеджера/клиента среди первых реплик.
    Если ничего не нашли — берём того, кто заговорил первым (запасной вариант).

    ОГРАНИЧЕНИЕ: для входящих звонков (клиент звонит первым и сразу задаёт
    вопрос) это правило может ошибаться — см. раздел "Ограничения" в дипломе.
    Для достоверного результата используй manual override ниже, когда роль
    известна заранее (например, из метаданных телефонии).
    """
    if not segments:
        return "SPEAKER_00"

    scores: dict[str, int] = {}
    for seg in segments:
        speaker = seg.speaker
        text_lower = seg.text.lower()

        scores.setdefault(speaker, 0)

        for phrase in MANAGER_KEYWORDS:
            if phrase.lower() in text_lower:
                scores[speaker] += 1

        for phrase in CLIENT_KEYWORDS:
            if phrase.lower() in text_lower:
                scores[speaker] -= 1

    print("Оценки спикеров:", scores)

    if len(set(scores.values())) > 1:
        return max(scores, key=scores.get)

    return segments[0].speaker


def guess_manager_speaker_manual(manager_speaker_label: str) -> str:
    """Явное указание роли — используй, когда знаешь её заранее (CRM/телефония)."""
    return manager_speaker_label
