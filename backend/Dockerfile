# Один образ для web, worker, beat и flower: отличается только команда запуска (см. docker/entrypoint.sh).
# Контекст сборки: папка backend/ (внутри неё и Django-проект, и пакет audio_comprehender).

# ---------- builder: установка зависимостей ----------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# build-essential: на случай пакетов, которые собираются из исходников
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install poetry

WORKDIR /app

# Окружение лежит в /app/.venv и целиком копируется в итоговый образ
RUN python -m venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml poetry.lock ./

# torch и torchaudio ставим pip-ом с индекса PyTorch (cu130) под версии из poetry.lock.
# Причина: poetry.lock собран на Windows, а Linux-сборки torch тянут за собой пакеты nvidia-*
# (cuDNN, cuBLAS и др.), которых в таком lock может не оказаться. pip подтянет их сам,
# а `poetry install` дальше увидит торч уже установленным.
RUN python - <<'PY'
import subprocess, sys, tomllib

with open("poetry.lock", "rb") as f:
    versions = {p["name"].lower(): p["version"] for p in tomllib.load(f)["package"]}

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    f"torch=={versions['torch']}", f"torchaudio=={versions['torchaudio']}",
    "--index-url", "https://download.pytorch.org/whl/cu130",
    "--extra-index-url", "https://pypi.org/simple",
])
PY

# Остальные зависимости по poetry.lock + проверка, что ключевые библиотеки импортируются
RUN poetry install --no-root \
    && python -c "import torch, torchaudio, ctranslate2, av; print('torch', torch.__version__, '| cuda', torch.version.cuda)" \
    && rm -rf /root/.cache


# ---------- runtime ----------
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    LD_LIBRARY_PATH="/app/.venv/lib/python3.13/site-packages/nvidia/cublas/lib:/app/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib" \
    HF_HOME=/models/hf

# ffmpeg и libsndfile: чтение аудио (torchcodec, soundfile); tzdata: часовой пояс Europe/Moscow
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY . /app

# sed: на случай, если скрипт сохранён с Windows-переводами строк (CRLF)
RUN sed -i 's/\r$//' /app/docker/entrypoint.sh \
    && chmod +x /app/docker/entrypoint.sh \
    && mkdir -p /app/media /app/staticfiles /models/hf

EXPOSE 8000

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["web"]