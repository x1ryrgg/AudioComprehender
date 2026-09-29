# Пути датасетов
DATA_DIR = r"D:\crowd_data\crowd"
TRAIN_TSV = rf"{DATA_DIR}\crowd_train\raw_crowd_train.tsv"
TEST_TSV = rf"{DATA_DIR}\crowd_test\raw_crowd_test.tsv"
WAV_DIR_TRAIN = rf"{DATA_DIR}\crowd_train"
WAV_DIR_TEST = rf"{DATA_DIR}\crowd_test"

# wav2vec2 всегда работает с 16kHz
SAMPLE_RATE = 16000

# распознаваемые эмоции
EMOTIONS = ["neutral", "positive", "angry", "sad", "other"]