"""
Пайплайн дообучения (fine-tuning) модели Wav2Vec2 для распознавания
эмоций в речи на датасете Dusha (часть crowd).

Рекомендуется запускать в Google Colab с GPU (T4 хватит для старта,
для полного датасета лучше A100 — как и делали авторы модели на HF).

Установка зависимостей (в Colab выполнить первой ячейкой):
    !pip install transformers torch torchaudio datasets pandas scikit-learn evaluate
"""

import os
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

from global_constants import (DATA_DIR, TRAIN_TSV, TEST_TSV, WAV_DIR_TRAIN, WAV_DIR_TEST,
                              SAMPLE_RATE, EMOTIONS)

# ---------------------------------------------------------------------------
# 1. КОНФИГУРАЦИЯ
# ---------------------------------------------------------------------------


# Базовая предобученная модель (можно заменить на facebook/wav2vec2-xls-r-300m,
# как в модели KELONMYOSA, если хватит вычислительных ресурсов)
BASE_MODEL = "facebook/wav2vec2-base"

MAX_SECONDS = 6  # обрезаем/дополняем аудио до 6 секунд для батчинга

LABEL2ID = {label: i for i, label in enumerate(EMOTIONS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# 2. ЗАГРУЗКА И АГРЕГАЦИЯ МЕТАДАННЫХ
# ---------------------------------------------------------------------------
#
# raw_crowd_*.tsv — это СЫРЫЕ данные: один и тот же аудиофайл (hash_id)
# оценивало по несколько асессоров (annotator_id), каждый — отдельной
# строкой в колонке annotator_emo. Для обучения классификатора нужна
# ОДНА метка на один файл, поэтому агрегируем через majority vote
# (самый частый ответ среди асессоров для данного hash_id).

def load_and_aggregate(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"Загружено {len(df)} сырых записей (голосов асессоров) из {tsv_path}")

    # Убираем строки без метки эмоции от асессора (на всякий случай)
    df = df.dropna(subset=["annotator_emo"])

    # Majority vote по каждому уникальному файлу
    aggregated = (
        df.groupby(["hash_id", "audio_path", "duration"])["annotator_emo"]
        .agg(lambda votes: votes.value_counts().idxmax())
        .reset_index()
        .rename(columns={"annotator_emo": "emotion"})
    )

    print(f"После агрегации: {len(aggregated)} уникальных аудиофайлов")
    print("Распределение классов после агрегации:")
    print(aggregated["emotion"].value_counts())

    return aggregated


# ---------------------------------------------------------------------------
# 3. DATASET
# ---------------------------------------------------------------------------

class DushaEmotionDataset(Dataset):
    """
    Каждый элемент — обработанный тензор аудио + числовая метка эмоции.
    Ресемплирует все файлы к 16kHz (требование wav2vec2) и приводит
    к единой длине для батчинга.
    """

    def __init__(self, df: pd.DataFrame, wav_dir: str, feature_extractor):
        self.df = df.reset_index(drop=True)
        self.wav_dir = wav_dir
        self.feature_extractor = feature_extractor
        self.max_length = MAX_SECONDS * SAMPLE_RATE

    def __len__(self):
        return len(self.df)

    def _load_audio(self, relative_path: str) -> torch.Tensor:
        # relative_path уже вида "wavs/<hash>.wav" — просто добавляем базу
        #
        # Используем soundfile вместо torchaudio.load(), потому что новые
        # версии torchaudio грузят аудио через torchcodec, которому нужен
        # отдельно установленный FFmpeg с DLL — на Windows это часто ломается
        # (см. ошибку "Could not load libtorchcodec"). soundfile работает
        # напрямую через libsndfile, которая ставится вместе с пакетом,
        # без внешних зависимостей.
        path = os.path.join(self.wav_dir, relative_path)
        data, sr = sf.read(path, dtype="float32")  # data: (samples,) или (samples, channels)
        waveform = torch.from_numpy(data)

        # моно (на случай стерео)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=-1)

        # ресемплинг к 16kHz
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        return waveform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform = self._load_audio(row["audio_path"])

        inputs = self.feature_extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        label = LABEL2ID[row["emotion"]]

        return {
            "input_values": inputs["input_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 4. МЕТРИКИ
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }


# ---------------------------------------------------------------------------
# 5. ОСНОВНОЙ ПАЙПЛАЙН
# ---------------------------------------------------------------------------

def main():
    # --- Данные (с агрегацией голосов асессоров) ---
    train_df = load_and_aggregate(TRAIN_TSV)
    test_df = load_and_aggregate(TEST_TSV)

    # ВАЖНО: класс "other" встречается очень редко (~1% в train) —
    # на старте можно исключить его из обучения, если модель будет
    # плохо его предсказывать из-за нехватки примеров:
    # train_df = train_df[train_df["emotion"] != "other"]
    # test_df = test_df[test_df["emotion"] != "other"]

    # SMOKE TEST: раскомментируй на первом запуске, чтобы быстро проверить,
    # что весь пайплайн отрабатывает без ошибок, прежде чем гонять на
    # полном датасете (182k файлов — это часы обучения на CPU).
    # train_df = train_df.sample(500)
    # test_df = test_df.sample(500)

    # --- Feature extractor и модель ---
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(BASE_MODEL)

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(EMOTIONS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    # Замораживаем feature extractor слои — дообучаем только классификатор
    # и верхние слои энкодера (экономит память и время, как делали
    # в референсной HuBERT-модели на Dusha)
    model.freeze_feature_encoder()

    # --- Датасеты ---
    train_dataset = DushaEmotionDataset(train_df, WAV_DIR_TRAIN, feature_extractor)
    test_dataset = DushaEmotionDataset(test_df, WAV_DIR_TEST, feature_extractor)

    # --- Аргументы обучения ---
    # batch_size=8 подобран под 8GB видеопамяти (RTX 3050) — если вылетит
    # CUDA out of memory, уменьши до 4 и увеличь gradient_accumulation_steps до 4
    training_args = TrainingArguments(
        output_dir="./ser_model_checkpoints",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        learning_rate=5e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),  # ускорение на GPU
        dataloader_num_workers=2,  # параллельная загрузка аудио
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # --- Обучение ---
    # Продолжаем с последнего чекпоинта, если он есть — не нужно обучать
    # заново с нуля. Trainer сам найдёт последний чекпоинт в output_dir
    # при resume_from_checkpoint=True (или передай точный путь строкой).
    import glob

    existing_checkpoints = glob.glob("./ser_model_checkpoints/checkpoint-*")
    if existing_checkpoints:
        print(f"Найдены чекпоинты, продолжаю обучение с последнего: {existing_checkpoints}")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Чекпоинтов не найдено, обучение с нуля")
        trainer.train()

    # --- Финальная оценка ---
    results = trainer.evaluate()
    print("Финальные метрики на тесте:", results)

    # --- Сохранение модели ---
    trainer.save_model("./ser_model_final")
    feature_extractor.save_pretrained("./ser_model_final")
    print("Модель сохранена в ./ser_model_final")


# ---------------------------------------------------------------------------
# 6. ИНФЕРЕНС (проверка на одном файле после обучения)
# ---------------------------------------------------------------------------

def predict_single_file(audio_path: str, model_dir: str = "./ser_model_final"):
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

    for emotion, prob in zip(EMOTIONS, probs.tolist()):
        print(f"{emotion}: {prob:.3f}")


if __name__ == "__main__":
    # main()
    predict_single_file(audio_path=DATA_DIR + r"\crowd_test\dialogs\holodniy_zvonok.mp3")