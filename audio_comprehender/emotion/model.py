"""Инференс обученной SER-модели. Обучение — в training/train_ser.py, не здесь."""

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

from audio_comprehender.config import SER_MODEL_DIR, SAMPLE_RATE

EMOTIONS = ["neutral", "positive", "angry", "sad", "other"]


class SERScorer:
    """Обёртка над SER-моделью — грузим один раз, применяем много раз."""

    def __init__(self, model_dir: str = SER_MODEL_DIR):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, waveform: torch.Tensor, sr: int) -> dict[str, float]:
        """waveform — одномерный тензор (моно). Возвращает {эмоция: вероятность}."""
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform.unsqueeze(0)).squeeze(0)

        inputs = self.feature_extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(inputs["input_values"]).logits
            probs = torch.softmax(logits, dim=-1).squeeze()

        return {emotion: prob.item() for emotion, prob in zip(EMOTIONS, probs)}


def extract_segment_waveform(full_waveform: np.ndarray, sr: int, start: float, end: float) -> torch.Tensor:
    """Вырезает кусок аудио по временным меткам (в секундах)."""
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    chunk = full_waveform[start_sample:end_sample]
    return torch.from_numpy(chunk.astype(np.float32))
