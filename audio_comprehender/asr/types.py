"""Общие типы данных для модуля распознавания речи."""

from dataclasses import dataclass, field


@dataclass
class TranscriptSegment:
    """Одна реплика с привязкой к говорящему."""
    speaker: str   # "SPEAKER_00" / "SPEAKER_01" — сырая метка от диаризации
    start: float
    end: float
    text: str


@dataclass
class CallTranscript:
    """Результат работы ASR-модуля целиком: транскрипт + кто есть кто."""
    segments: list[TranscriptSegment]
    manager_speaker: str  # какая из сырых меток (SPEAKER_00/01) — менеджер

    def manager_segments(self) -> list[TranscriptSegment]:
        return [s for s in self.segments if s.speaker == self.manager_speaker]

    def client_segments(self) -> list[TranscriptSegment]:
        return [s for s in self.segments if s.speaker != self.manager_speaker]

    def as_dialogue_text(self) -> str:
        """Текст вида 'МЕНЕДЖЕР: ...\\nКЛИЕНТ: ...' — то, что уходит в LLM."""
        lines = []
        for seg in self.segments:
            role = "МЕНЕДЖЕР" if seg.speaker == self.manager_speaker else "КЛИЕНТ"
            lines.append(f"{role}: {seg.text}")
        return "\n".join(lines)
