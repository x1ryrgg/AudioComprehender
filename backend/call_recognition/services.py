import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

from django.conf import settings


class AnalysisError(RuntimeError):
    pass


def run_analysis(call) -> dict:
    """Анализ звонка. Возвращает словарь CallReport.to_dict().

    Анализатор (audio_comprehender) живёт в своём проекте и своём виртуальном окружении,
    поэтому backend запускает его отдельным процессом через run_analysis.py
    и читает готовый JSON-отчёт.
    """
    python = Path(settings.ANALYZER_PYTHON)
    script = Path(settings.ANALYZER_SCRIPT)
    for path in (python, script):
        if not path.exists():
            raise AnalysisError(f"Не найден файл: {path}")

    # Переменные окружения backend не должны влиять на интерпретатор анализатора
    env = {k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"}}
    # Без этого print() с кириллицей/символами падает в Windows при выводе в pipe
    env["PYTHONUTF8"] = "1"

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "report.json"
        cmd = [str(python), str(script), call.audio_file.path, "--out", str(out_path)]
        try:
            proc = subprocess.run(
                cmd,
                cwd=settings.ANALYZER_CWD,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=settings.ANALYZER_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise AnalysisError(f"Анализ превысил лимит {settings.ANALYZER_TIMEOUT} с")

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-1500:]
            raise AnalysisError(f"Анализатор завершился с кодом {proc.returncode}:\n{tail}")

        return json.loads(out_path.read_text(encoding="utf-8"))

