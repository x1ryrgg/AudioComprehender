import soundfile as sf
import torch
import torchaudio

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

from audio_comprehender.config import SER_MODEL_DIR, SAMPLE_RATE, DATA_DIR, EMOTIONS


def predict_single_file(audio_path: str, model_dir: str = SER_MODEL_DIR) -> dict:
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    data, sr = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=-1)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform.unsqueeze(0)).squeeze(0)

    inputs = feature_extractor(
        waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(inputs["input_values"]).logits
        probs = torch.softmax(logits, dim=-1).squeeze()

    predictions = {}

    for emotion, prob in zip(EMOTIONS, probs.tolist()):
        print(f"{emotion}: {prob:.3f}")
        predictions[emotion] = prob

    return predictions


if __name__ == "__main__":
    predict_single_file(audio_path=DATA_DIR / "crowd_test/dialogs/aggressive_01.wav")