"""Диаризация: разбивка аудио по говорящим без текста."""

import torch
import soundfile as sf
from pyannote.audio import Pipeline as DiarizationPipeline

from audio_comprehender.config import DIARIZATION_MODEL, HF_TOKEN

_pipeline = None  # ленивая загрузка — модель грузится один раз на процесс


def _get_pipeline() -> DiarizationPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DiarizationPipeline.from_pretrained(DIARIZATION_MODEL, token=HF_TOKEN)
    return _pipeline


def diarize(audio_path: str, num_speakers: int = 2) -> list[tuple[str, float, float]]:
    """
    Возвращает список (speaker_label, start_sec, end_sec).
    num_speakers=2 — звонок менеджер-клиент, ровно два говорящих;
    поставь None, если заранее не известно сколько людей в записи.
    """
    pipeline = _get_pipeline()

    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T.copy())

    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)

    annotation = diarization.speaker_diarization

    return [
        (speaker, turn.start, turn.end)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
