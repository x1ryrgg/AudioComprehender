import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

from django.conf import settings


# Папка анализатора (лежит внутри backend). По ней считается версия:
# правки остального кода backend версию анализатора не меняют.
ANALYZER_PATHS = ["audio_comprehender"]

class AnalysisError(RuntimeError):
    pass


def get_analyzer_version() -> str:
    """Версия анализатора.

    В Docker задаётся при сборке через ANALYZER_VERSION (внутри образа нет .git).
    Локально считается по git: короткий хэш последнего коммита, затронувшего audio_comprehender/,
    с суффиксом -dirty, если в отслеживаемых файлах есть незакоммиченные правки.
    """
    if getattr(settings, "ANALYZER_VERSION", ""):
        return settings.ANALYZER_VERSION

    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=settings.BASE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=True,
        ).stdout.strip()

    try:
        commit = git("log", "-1", "--format=%h", "--", *ANALYZER_PATHS)
        dirty = git("status", "--porcelain", "--untracked-files=no", "--", *ANALYZER_PATHS)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not commit:
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def run_analysis(call) -> dict:
    """Анализ звонка. Возвращает словарь CallReport.to_dict()."""

    # Ленивый импорт: torch и модели загружаются только в воркере и только при первом звонке.
    # Не переносить импорт наверх файла: иначе torch импортируется до fork воркера Celery.
    from audio_comprehender.report import analyze_call

    data = analyze_call(call.audio_file.path).to_dict()
    data.setdefault("analyzer_version", get_analyzer_version())
    return data

