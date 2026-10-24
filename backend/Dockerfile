# Один образ для web, worker, beat и flower: отличается только команда запуска (см. docker/entrypoint.sh).
# Контекст сборки: папка backend/ (внутри неё и Django-проект, и пакет audio_comprehender).
# Вычисления идут на CPU: CUDA-библиотеки в образ не ставятся.

# ---------- builder: установка зависимостей ----------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential: на случай пакетов, которые собираются из исходников
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Окружение лежит в /app/.venv и целиком копируется в итоговый образ
RUN python -m venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN pip install packaging

WORKDIR /build
COPY poetry.lock docker/lock_to_requirements.py ./

# 1) torch и torchaudio: CPU-сборка с индекса PyTorch, версии те же, что в poetry.lock.
#    Ставим первыми: тогда pip не потянет за другими пакетами CUDA-вариант torch с PyPI.
RUN python lock_to_requirements.py torch-specs poetry.lock > torch.txt \
    && cat torch.txt \
    && pip install --no-deps -r torch.txt --index-url https://download.pytorch.org/whl/cpu

# 2) Остальные зависимости с теми же версиями, что в poetry.lock (без torch, triton и nvidia-*;
#    пакеты только для Windows отбрасываются по маркерам)
RUN python lock_to_requirements.py requirements poetry.lock > requirements.txt \
    && pip install -r requirements.txt

# 3) Проверка, что в образ попал именно CPU-torch; pip check только выводит замечания
RUN python -c "import torch; assert torch.version.cuda is None, 'в образ попала CUDA-сборка torch'; print('torch', torch.__version__, '(CPU)')" \
    && (pip check || true) \
    && rm -rf /root/.cache


# ---------- runtime ----------
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models/hf

# ffmpeg и libsndfile: чтение аудио (torchcodec, soundfile); libgomp1: OpenMP для расчётов на CPU;
# tzdata: часовой пояс Europe/Moscow
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgomp1 tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# Проверка, что ключевые библиотеки анализатора импортируются
RUN python -c "import torchaudio, ctranslate2, av, transformers, faster_whisper, pyannote.audio; print('imports ok')"

COPY . /app

RUN sed -i 's/\r$//' /app/docker/entrypoint.sh \
    && chmod +x /app/docker/entrypoint.sh \
    && mkdir -p /app/media /app/staticfiles /models/hf

EXPOSE 8000

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["web"]