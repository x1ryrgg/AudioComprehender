import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(verbose=False)

# ---------------------------------------------------------------------------
# Windows: CTranslate2 (faster-whisper) не всегда находит cuBLAS/cuDNN
# через os.add_dll_directory — дублируем путь через PATH.
# На Linux (прод/сервер) этот блок просто не выполняется.
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    import nvidia.cublas
    import nvidia.cudnn

    _cublas_bin = os.path.join(list(nvidia.cublas.__path__)[0], "bin")
    _cudnn_bin = os.path.join(list(nvidia.cudnn.__path__)[0], "bin")

    os.add_dll_directory(_cublas_bin)
    os.add_dll_directory(_cudnn_bin)
    os.environ["PATH"] = _cublas_bin + os.pathsep + _cudnn_bin + os.pathsep + os.environ["PATH"]


# ---------------------------------------------------------------------------
# Пути к моделям
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent

MODELS_DIR = PACKAGE_DIR / "models"

SER_MODEL_DIR = Path(os.getenv("SER_MODEL_DIR", str(MODELS_DIR / "ser_model_final")))
SER_CHECKPOINTS_DIR = Path(os.getenv("SER_CHECKPOINTS_DIR",str(MODELS_DIR / "ser_model_checkpoints")))

DATA_DIR = Path(os.getenv("DATA_DIR", r'Disk:\crowd_data\crowd')) # DATA_DIR=D:\crowd_data\crowd
TRAIN_TSV     = Path(os.getenv("TRAIN_TSV",     DATA_DIR / "crowd_train" / "raw_crowd_train.tsv"))
TEST_TSV      = Path(os.getenv("TEST_TSV",      DATA_DIR / "crowd_test"  / "raw_crowd_test.tsv"))
WAV_DIR_TRAIN = Path(os.getenv("WAV_DIR_TRAIN", DATA_DIR / "crowd_train"))
WAV_DIR_TEST  = Path(os.getenv("WAV_DIR_TEST",  DATA_DIR / "crowd_test"))

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

SAMPLE_RATE = 16000

EMOTIONS = ["neutral", "positive", "angry", "sad", "other"]

# ---------------------------------------------------------------------------
# Секреты / внешние сервисы
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
