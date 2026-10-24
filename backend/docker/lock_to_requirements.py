"""Преобразует poetry.lock в списки зависимостей для pip (сборка Docker-образа без CUDA).

poetry.lock собран на Windows под GPU: torch там с индекса cu130, а рядом лежат пакеты nvidia-*.
В Docker-образе вычисления идут на CPU, поэтому torch ставится отдельно (CPU-сборка), а всё остальное
берётся из lock с теми же версиями, что и на твоём ПК.

Использование:
    python lock_to_requirements.py requirements poetry.lock   # зависимости для Linux без torch и nvidia-*
    python lock_to_requirements.py torch-specs poetry.lock    # torch и torchaudio тех же версий без суффикса +cu130
"""
import sys
import tomllib

from packaging.markers import Marker
from packaging.version import Version

TORCH_PACKAGES = {"torch", "torchaudio"}
# Пакеты, нужные только для CUDA: в CPU-образ не попадают
SKIP_NAMES = TORCH_PACKAGES | {"triton", "pytorch-triton"}
SKIP_PREFIXES = ("nvidia-", "cuda-")


def load_packages(path):
    with open(path, "rb") as f:
        return tomllib.load(f)["package"]


def applies(package) -> bool:
    """Нужен ли пакет в основном окружении на текущей платформе (Linux, текущий Python)."""
    groups = package.get("groups")
    if groups is not None and "main" not in groups:
        return False
    markers = package.get("markers")
    if isinstance(markers, dict):
        markers = markers.get("main")
    if markers:
        return Marker(markers).evaluate()
    return True


def requirements(path):
    chosen = {}
    for package in load_packages(path):
        name = package["name"].lower()
        if name in SKIP_NAMES or name.startswith(SKIP_PREFIXES):
            continue

        source = package.get("source") or {}
        source_type = source.get("type")
        if source_type == "legacy" and str(source.get("reference", "")).startswith("pytorch"):
            continue  # пакеты с индекса PyTorch ставятся отдельно
        if source_type is not None:
            raise SystemExit(f"Источник пакета {name} ({source_type}) скрипт не поддерживает")

        if not applies(package):
            continue

        version = package["version"]
        if name in chosen and Version(version) <= Version(chosen[name]):
            continue  # несколько версий под разные условия: берём самую новую
        chosen[name] = version

    for name in sorted(chosen):
        print(f"{name}=={chosen[name]}")


def torch_specs(path):
    versions = {
        p["name"].lower(): p["version"].split("+")[0]
        for p in load_packages(path)
        if p["name"].lower() in TORCH_PACKAGES
    }
    missing = TORCH_PACKAGES - versions.keys()
    if missing:
        raise SystemExit(f"В poetry.lock не найдены: {', '.join(sorted(missing))}")
    for name in sorted(versions):
        print(f"{name}=={versions[name]}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("requirements", "torch-specs"):
        raise SystemExit(__doc__)
    mode, lock_path = sys.argv[1], sys.argv[2]
    (requirements if mode == "requirements" else torch_specs)(lock_path)