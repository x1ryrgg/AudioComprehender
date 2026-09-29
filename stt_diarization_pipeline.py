"""
Распознавание речи (Speech-to-Text) + диаризация (разделение по говорящим)
для звонков менеджер-клиент.

Установка (в активированном окружении, где уже стоит CUDA-версия torch):
    pip install faster-whisper pyannote.audio

Диаризация (pyannote) требует токен Hugging Face и согласие с условиями
использования модели — см. ШАГ 0 ниже, без этого скрипт не запустится.
"""

import os

import nvidia.cublas
import nvidia.cudnn


import torch
import soundfile as sf
from dataclasses import dataclass
from dotenv import load_dotenv

if os.name == "nt": # Для Windows из за плохого поиска библиотек
    cublas_bin = os.path.join(list(nvidia.cublas.__path__)[0], "bin")
    cudnn_bin = os.path.join(list(nvidia.cudnn.__path__)[0], "bin")

    os.add_dll_directory(cublas_bin)
    os.add_dll_directory(cudnn_bin)

    os.environ["PATH"] = cublas_bin + os.pathsep + cudnn_bin + os.pathsep + os.environ["PATH"]

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as DiarizationPipeline

from global_constants import DATA_DIR, TRAIN_TSV, TEST_TSV, WAV_DIR_TRAIN, WAV_DIR_TEST

load_dotenv(verbose=False)

HF_TOKEN = os.getenv("HF_TOKEN")

# ---------------------------------------------------------------------------
# КОНФИГУРАЦИЯ
# ---------------------------------------------------------------------------

WHISPER_MODEL_SIZE = "medium"  # варианты: tiny/base/small/medium/large-v3
                                 # medium — разумный баланс качества/скорости
                                 # для русского языка на GPU 8GB
DEVICE = "cuda"  # поставь "cpu", если запускаешь без GPU (будет медленнее)
COMPUTE_TYPE = "float16"  # для GPU; на CPU используй "int8"


@dataclass
class TranscriptSegment:
    speaker: str      # "SPEAKER_00", "SPEAKER_01" и т.д. (пока без привязки
                       # к роли менеджер/клиент — см. функцию guess_manager_speaker)
    start: float
    end: float
    text: str


# ---------------------------------------------------------------------------
# 1. ДИАРИЗАЦИЯ — кто говорит и когда
# ---------------------------------------------------------------------------

def diarize(audio_path: str):
    """
    Возвращает список сегментов вида (speaker_label, start_sec, end_sec)
    без текста — просто разметку "кто говорил в какой промежуток времени".
    """
    pipeline = DiarizationPipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN,
    )

    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(audio.T.copy())

    diarization = pipeline({
        "waveform": waveform,
        "sample_rate": sample_rate,
    })

    annotation = diarization.speaker_diarization

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append((speaker, turn.start, turn.end))

    return segments


# ---------------------------------------------------------------------------
# 2. РАСПОЗНАВАНИЕ РЕЧИ — что было сказано
# ---------------------------------------------------------------------------

def transcribe(audio_path: str):
    """
    Возвращает список сегментов вида (start_sec, end_sec, text)
    от Whisper, без привязки к говорящему.
    """
    model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    segments, info = model.transcribe(
        audio_path,
        language="ru",
        beam_size=5,
        vad_filter=True,  # отсекает тишину/шум, экономит время
    )

    result = []
    for seg in segments:
        result.append((seg.start, seg.end, seg.text.strip()))

    print(f"Распознано, определён язык: {info.language} (вероятность {info.language_probability:.2f})")
    return result


# ---------------------------------------------------------------------------
# 3. ОБЪЕДИНЕНИЕ — сопоставляем текст с говорящим по времени
# ---------------------------------------------------------------------------

def merge_transcript_with_speakers(whisper_segments, diarization_segments):
    """
    Для каждого текстового сегмента от Whisper находим говорящего,
    чей диаризационный сегмент перекрывается с ним больше всего по времени.
    """
    merged = []

    for w_start, w_end, text in whisper_segments:
        best_speaker = None
        best_overlap = 0.0

        for speaker, d_start, d_end in diarization_segments:
            overlap = max(0.0, min(w_end, d_end) - max(w_start, d_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        merged.append(
            TranscriptSegment(
                speaker=best_speaker or "UNKNOWN",
                start=w_start,
                end=w_end,
                text=text,
            )
        )

    return merged


# ---------------------------------------------------------------------------
# 4. ЭВРИСТИКА: кто из спикеров — менеджер
# ---------------------------------------------------------------------------
#
# pyannote не знает ролей "менеджер"/"клиент", только абстрактные метки.
# Для исходящих звонков колл-центра почти всегда ПЕРВЫМ говорит менеджер
# (приветствие) — это разумное упрощение для старта, но не железное правило
# (клиент может поднять трубку и сказать "алло" первым). Более надёжный
# способ — искать характерные фразы приветствия ("компания", "меня зовут")
# в начале звонка, это можно добавить позже.

def guess_manager_speaker(merged_segments: list[TranscriptSegment]) -> str:
    """
    Улучшенная эвристика: ищем ключевые фразы в первых нескольких репликах
    каждого спикера. Если явных признаков нет — откатываемся на старое
    правило "кто говорил первым".

    НЕРЕШЕННЫЙ МОМЕНТ ОЧЕНЬ НЕНАДЕЖНО
    """

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

    if not merged_segments:
        return "SPEAKER_00"

    scores = {}
    for seg in merged_segments:
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

    # Фолбэк — старое правило, если ключевых фраз не нашлось
    return merged_segments[0].speaker


def guess_manager_speaker_manual(merged_segments: list, manager_speaker_label: str) -> str:
    """
    Ручное указание роли — использовать, когда заранее известно кто менеджер а кто не менеджер (например,
    из метаданных CRM/телефонии: направление звонка, номер канала).
    В реальной системе это почти всегда надёжнее любой эвристики.
    """
    return manager_speaker_label


def transcribe_words(audio_path: str, use_vad: bool = True):
    """
    Возвращает список слов вида (start_sec, end_sec, word).

    Диагностика обрезанного начала/конца: запусти один раз с use_vad=False.
    Если приветствие появилось — виноват VAD, оставляй мягкие параметры ниже.
    """
    model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    segments, info = model.transcribe(
        audio_path,
        language="ru",
        beam_size=5,
        word_timestamps=True,
        vad_filter=use_vad,
        vad_parameters=dict(
            threshold=0.3,
            speech_pad_ms=500,
            min_silence_duration_ms=500,
        ),
    )

    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append((w.start, w.end, w.word))
        else:
            # запасной вариант: целый сегмент как одно "слово"
            words.append((seg.start, seg.end, " " + seg.text.strip()))

    print(f"Распознано слов: {len(words)}, язык: {info.language} ({info.language_probability:.2f})")
    return words


# ---------------------------------------------------------------------------
# 3. ПОСЛОВНОЕ НАЗНАЧЕНИЕ ГОВОРЯЩИХ
# ---------------------------------------------------------------------------

def _speaker_for_interval(start, end, diarization_segments):
    best_speaker, best_overlap = None, 0.0
    for speaker, d_start, d_end in diarization_segments:
        overlap = max(0.0, min(end, d_end) - max(start, d_start))
        if overlap > best_overlap:
            best_speaker, best_overlap = speaker, overlap

    if best_speaker is None and diarization_segments:
        # слово попало в "дыру" между сегментами диаризации — берём ближайший по времени
        mid = (start + end) / 2
        best_speaker = min(
            diarization_segments,
            key=lambda s: min(abs(mid - s[1]), abs(mid - s[2])),
        )[0]

    return best_speaker or "UNKNOWN"


def assign_speakers_by_words(words, diarization_segments, max_gap: float = 1.5):
    """
    Каждому слову назначаем говорящего по тайм-коду, затем склеиваем подряд
    идущие слова одного говорящего в реплику. Новая реплика начинается,
    когда меняется говорящий или пауза между словами больше max_gap секунд.
    """
    turns = []

    for start, end, word in words:
        speaker = _speaker_for_interval(start, end, diarization_segments)

        if turns and turns[-1].speaker == speaker and start - turns[-1].end <= max_gap:
            turns[-1].text += word  # у слов faster-whisper уже есть ведущий пробел
            turns[-1].end = end
        else:
            turns.append(TranscriptSegment(speaker=speaker, start=start, end=end, text=word))

    for t in turns:
        t.text = t.text.strip()

    return turns



# ---------------------------------------------------------------------------
# ОСНОВНОЙ ПАЙПЛАЙН
# ---------------------------------------------------------------------------

def process_call(audio_path: str):
    print(f"Обрабатываю: {audio_path}")

    print("Шаг 1/3: диаризация (кто говорит)...")
    diarization_segments = diarize(audio_path)

    print("Шаг 2/3: распознавание речи...")
    words = transcribe_words(audio_path)

    print("Шаг 3/3: объединение...")
    merged = assign_speakers_by_words(words, diarization_segments)

    manager_label = guess_manager_speaker(merged)

    print("\n--- Итоговый транскрипт ---\n")
    for seg in merged:
        role = "МЕНЕДЖЕР" if seg.speaker == manager_label else "КЛИЕНТ"
        print(f"[{seg.start:6.1f}s -> {seg.end:6.1f}s] {role}: {seg.text}")

    return merged, manager_label


if __name__ == "__main__":
    process_call(audio_path=DATA_DIR + r"\crowd_test\dialogs\holodniy_zvonok.mp3")
