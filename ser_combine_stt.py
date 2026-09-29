"""
Объединяем SER (эмоция по голосу) и STT+диаризацию (текст + кто говорит).

Идея: вместо одной оценки эмоции на весь звонок — прогоняем SER-модель
ОТДЕЛЬНО по каждому сегменту речи менеджера (границы уже знает диаризация),
получаем эмоциональную траекторию по ходу разговора.
"""

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

from stt_diarization_pipeline import process_call, TranscriptSegment

from global_constants import DATA_DIR, SAMPLE_RATE, EMOTIONS


SER_MODEL_DIR = "./ser_model_final"


class SERScorer:
    """Обёртка над SER-моделью — грузим один раз, применяем много раз."""

    def __init__(self, model_dir: str = SER_MODEL_DIR):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, waveform: torch.Tensor, sr: int) -> dict:
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


def analyze_call(audio_path: str):
    # --- STT + диаризация (уже готовый пайплайн) ---
    merged_segments, manager_label = process_call(audio_path)

    # --- Загружаем аудио целиком один раз для нарезки кусков ---
    full_audio, sr = sf.read(audio_path, dtype="float32")
    if full_audio.ndim > 1:
        full_audio = full_audio.mean(axis=1)  # моно

    # --- SER модель ---
    ser = SERScorer()

    print("\n--- Эмоциональная траектория менеджера ---\n")

    manager_emotions_over_time = []

    for seg in merged_segments:
        if seg.speaker != manager_label:
            continue  # интересует только менеджер

        # Пропускаем слишком короткие сегменты — SER на них ненадёжен
        if seg.end - seg.start < 0.5:
            continue

        chunk_waveform = extract_segment_waveform(full_audio, sr, seg.start, seg.end)
        emotion_probs = ser.predict(chunk_waveform, sr)
        dominant_emotion = max(emotion_probs, key=emotion_probs.get)

        manager_emotions_over_time.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "dominant_emotion": dominant_emotion,
            "confidence": emotion_probs[dominant_emotion],
        })

        print(
            f"[{seg.start:6.1f}s -> {seg.end:6.1f}s] "
            f"эмоция={dominant_emotion} ({emotion_probs[dominant_emotion]:.2f}) | "
            f"текст: {seg.text}"
        )

    # --- Сводка по звонку ---
    if manager_emotions_over_time:
        from collections import Counter
        emotion_counts = Counter(e["dominant_emotion"] for e in manager_emotions_over_time)
        print("\n--- Сводка эмоций менеджера за звонок ---")
        for emotion, count in emotion_counts.most_common():
            pct = 100 * count / len(manager_emotions_over_time)
            print(f"  {emotion}: {count} сегментов ({pct:.0f}%)")

        negative_share = (
            emotion_counts.get("angry", 0) + emotion_counts.get("sad", 0)
        ) / len(manager_emotions_over_time)
        if negative_share > 0.3:
            print(f"\n  ⚠ Высокая доля негативных эмоций у менеджера: {negative_share:.0%}")

    return merged_segments, manager_emotions_over_time


if __name__ == "__main__":
    analyze_call(audio_path=DATA_DIR + r"\crowd_test\dialogs\Phone_ARU_OFF.wav")