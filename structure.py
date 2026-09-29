"""
Запусти этот скрипт первым, чтобы увидеть реальную структуру данных
после распаковки crowd.tar. Пришли мне вывод — я поправлю основной
пайплайн (ser_finetune_pipeline.py) под точные названия колонок.
"""

import pandas as pd
import os

TRAIN_TSV = r"D:\crowd_data\crowd\crowd_train\raw_crowd_train.tsv"
TEST_TSV = r"D:\crowd_data\crowd\crowd_test\raw_crowd_test.tsv"

for name, path in [("TRAIN", TRAIN_TSV), ("TEST", TEST_TSV)]:
    print(f"\n{'=' * 50}")
    print(f"{name}: {path}")
    print('=' * 50)

    if not os.path.exists(path):
        print("ФАЙЛ НЕ НАЙДЕН — проверь путь")
        continue

    df = pd.read_csv(path, sep="\t")
    print(f"Всего строк: {len(df)}")
    print(f"\nКолонки: {df.columns.tolist()}")
    print(f"\nПервые 3 строки:")
    print(df.head(3).to_string())

    # Если есть колонка с эмоцией — покажем распределение классов
    for col in df.columns:
        if "emo" in col.lower():
            print(f"\nРаспределение по колонке '{col}':")
            print(df[col].value_counts())

# Также проверим, где физически лежат .wav файлы
print(f"\n{'=' * 50}")
print("Поиск .wav файлов рядом с tsv")
print('=' * 50)
for base in [r"D:\crowd_data\crowd_extracted\crowd_train",
             r"D:\crowd_data\crowd_extracted\crowd_test"]:
    if os.path.exists(base):
        for root, dirs, files in os.walk(base):
            wavs = [f for f in files if f.endswith(".wav")]
            if wavs:
                print(f"{root}: найдено {len(wavs)} .wav файлов, пример: {wavs[0]}")
                break