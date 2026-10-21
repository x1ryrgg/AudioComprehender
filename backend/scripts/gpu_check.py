"""Проверка GPU и моделей внутри контейнера.

Запуск: docker compose run --rm worker python scripts/gpu_check.py
Проверяет по очереди: torch (pyannote, SER), cuDNN, CTranslate2 (faster-whisper),
реальный запуск Whisper на GPU и загрузку модели диаризации из кэша Hugging Face.
"""
import sys

import numpy as np

failed = False


def step(name, fn):
    global failed
    try:
        print(f"[ OK ] {name}: {fn()}", flush=True)
    except Exception as exc:  # noqa: BLE001
        failed = True
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)


def torch_cuda():
    import torch

    assert torch.cuda.is_available(), "CUDA недоступна (проверь, что у контейнера есть GPU)"
    return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}"


def torch_cudnn():
    import torch

    conv = torch.nn.Conv1d(1, 4, 3).cuda()
    with torch.no_grad():
        conv(torch.randn(1, 1, 1600, device="cuda"))
    return f"cuDNN {torch.backends.cudnn.version()}"


def ct2_cuda():
    import ctranslate2

    count = ctranslate2.get_cuda_device_count()
    assert count > 0, "CTranslate2 не видит GPU"
    return f"устройств: {count}"


def whisper_infer():
    from faster_whisper import WhisperModel

    from audio_comprehender import config

    model = WhisperModel(
        config.WHISPER_MODEL_SIZE, device="cuda", compute_type=config.WHISPER_COMPUTE_TYPE
    )
    audio = (np.random.randn(16000 * 3) * 0.01).astype("float32")
    segments, _ = model.transcribe(audio)
    list(segments)
    return f"модель {config.WHISPER_MODEL_SIZE} отработала на GPU"


def diarization_load():
    from pyannote.audio import Pipeline

    from audio_comprehender import config

    try:
        Pipeline.from_pretrained(config.DIARIZATION_MODEL, token=config.HF_TOKEN)
    except TypeError:  # старые версии pyannote принимают use_auth_token
        Pipeline.from_pretrained(config.DIARIZATION_MODEL, use_auth_token=config.HF_TOKEN)
    return f"{config.DIARIZATION_MODEL} загружена"


step("torch + CUDA", torch_cuda)
step("torch + cuDNN", torch_cudnn)
step("CTranslate2 + CUDA", ct2_cuda)
step("Whisper на GPU", whisper_infer)
step("Модель диаризации", diarization_load)

sys.exit(1 if failed else 0)