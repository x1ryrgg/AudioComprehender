# import torch
#
# print(torch.cuda.is_available())       # должно быть True
# print(torch.cuda.get_device_name(0))
#
# EMOTIONS = ["neutral", "positive", "angry", "sad", "other"]
# LABEL2ID = {label: i for i, label in enumerate(EMOTIONS)}
# ID2LABEL = {i: label for label, i in LABEL2ID.items()}
#
# print(LABEL2ID)
# print(ID2LABEL)
#
# """
# Смотрим метрики обучения из уже сохранённого trainer_state.json —
# не нужно ничего переобучать, чтобы узнать, как отработала модель.
# """
#
# ########################################################################################################
# import json
#
# STATE_PATH = r"C:\python\AudioComprehender\ser_model_checkpoints\checkpoint-11382\trainer_state.json"
#
# with open(STATE_PATH, "r", encoding="utf-8") as f:
#     state = json.load(f)
#
# print("История метрик по эпохам:\n")
# for entry in state["log_history"]:
#     if "eval_accuracy" in entry:
#         print(
#             f"Эпоха {entry.get('epoch', '?')}: "
#             f"accuracy={entry['eval_accuracy']:.4f}, "
#             f"f1_macro={entry['eval_f1_macro']:.4f}, "
#             f"loss={entry['eval_loss']:.4f}"
#         )
#
# print(f"\nЛучшая метрика (best_metric): {state.get('best_metric')}")
# print(f"Лучший чекпоинт: {state.get('best_model_checkpoint')}")


"""
Смотрим "сырые" данные, чтобы понять, где именно ломается разметка:
в самой диаризации (pyannote) или в тайм-кодах слов (Whisper).

Запуск: python -m debug_alignment
"""

from stt_diarization_pipeline import diarize, transcribe_words

AUDIO = r"D:\crowd_data\crowd\crowd_test\dialogs\Phone_ARU_ON.wav"


def main():
    print("--- Сегменты диаризации (pyannote) ---")
    for speaker, start, end in diarize(AUDIO):
        print(f"  {speaker}: {start:5.1f} -> {end:5.1f}")

    print("\n--- Слова Whisper с тайм-кодами ---")
    for start, end, word in transcribe_words(AUDIO):
        print(f"  [{start:5.1f} -> {end:5.1f}] {word.strip()}")


def reader(file_path: str):
    import json
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    print(json.dumps(data, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    reader(file_path=r"D:\crowd_data\crowd\crowd_test\dialogs\holodniy_zvonok_report.json")



