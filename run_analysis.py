"""
Точка входа для ручных прогонов из консоли — временная замена
разрозненным if __name__ == "__main__" в старых файлах.

Использование:
    python run_analysis.py "D:\путь\к\звонку.wav"
    python run_analysis.py "D:\путь\к\звонку.wav" --manager SPEAKER_00
"""

import argparse
import json

from audio_comprehender.report import analyze_call


def main():
    parser = argparse.ArgumentParser(description="Анализ звонка менеджер-клиент")
    parser.add_argument("audio_path", help="Путь к .wav файлу звонка")
    parser.add_argument(
        "--manager", default=None,
        help="Явно указать метку менеджера (SPEAKER_00/SPEAKER_01), "
             "если известна заранее — иначе определяется эвристикой",
    )
    parser.add_argument("--out", default=None, help="Путь для сохранения JSON-отчёта")
    args = parser.parse_args()

    report = analyze_call(args.audio_path, manager_speaker=args.manager)

    print("\n=== ИТОГОВЫЙ ОТЧЁТ ===")
    print(f"Итоговая оценка: {report.final_score} / 10")
    print(f"  текст (LLM): {report.text_analysis.overall_score}")
    print(f"  тон (SER):   {report.emotion_analysis.score}")
    for flag in report.flags:
        print(f"  ⚠ {flag}")

    out_path = args.out or args.audio_path.rsplit(".", 1)[0] + "_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"\nПолный отчёт сохранён: {out_path}")


if __name__ == "__main__":
    main()
