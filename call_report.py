"""
Модуль объединения: итоговый отчёт по звонку.

Пайплайн:
  1. STT + диаризация — ОДИН раз (process_call)
  2. SER по репликам менеджера -> эмоциональная траектория и оценка тона
  3. LLM-анализ текста -> скрипт, возражения, запрещённые фразы, оценка
  4. Объединение в итоговый балл 0-10 + флаги + JSON-отчёт

Запуск: python -m call_report
"""

import json
from collections import Counter

import soundfile as sf

from stt_diarization_pipeline import process_call
from ser_combine_stt import SERScorer, extract_segment_waveform
from text_llm_analysis import analyze_transcript_with_llm

# ---------------------------------------------------------------------------
# ПАРАМЕТРЫ ОЦЕНКИ — это проектные решения, их нужно обосновать в дипломе
# (например, подобрав веса по сравнению с оценками эксперта на нескольких звонках)
# ---------------------------------------------------------------------------

W_TEXT = 0.7      # вес текстовой оценки LLM в итоговом балле
W_EMOTION = 0.3   # вес оценки тона (SER)

MIN_SEGMENT_SEC = 0.5   # реплики короче SER не оценивает (ненадёжно)
MIN_CONFIDENCE = 0.5    # предсказания SER с меньшей уверенностью не учитываем

# Насколько "плохой" каждая эмоция в речи менеджера
NEGATIVE_WEIGHTS = {"angry": 1.0, "sad": 0.5}
NEGATIVE_FLAG_THRESHOLD = 0.3   # доля негатива, после которой ставим флаг


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 1. ЭМОЦИИ МЕНЕДЖЕРА (SER по каждой реплике)
# ---------------------------------------------------------------------------

def analyze_manager_emotions(audio_path, segments, manager_label, ser) -> dict:
    full_audio, sr = sf.read(audio_path, dtype="float32")
    if full_audio.ndim > 1:
        full_audio = full_audio.mean(axis=1)

    timeline = []
    for seg in segments:
        if seg.speaker != manager_label:
            continue
        duration = seg.end - seg.start
        if duration < MIN_SEGMENT_SEC:
            continue

        probs = ser.predict(extract_segment_waveform(full_audio, sr, seg.start, seg.end), sr)
        emotion = max(probs, key=probs.get)
        timeline.append({
            "start": round(seg.start, 1),
            "end": round(seg.end, 1),
            "duration": duration,
            "text": seg.text,
            "emotion": emotion,
            "confidence": round(probs[emotion], 3),
        })

    reliable = [t for t in timeline if t["confidence"] >= MIN_CONFIDENCE]
    total = sum(t["duration"] for t in reliable)

    if total == 0:
        return {"score": None, "negative_share": None, "distribution": {}, "timeline": timeline}

    negative = sum(NEGATIVE_WEIGHTS.get(t["emotion"], 0.0) * t["duration"] for t in reliable)
    negative_share = negative / total

    by_emotion = Counter()
    for t in reliable:
        by_emotion[t["emotion"]] += t["duration"]

    return {
        "score": round(10 * (1 - negative_share), 1),
        "negative_share": round(negative_share, 3),
        "distribution": {e: round(d / total, 3) for e, d in by_emotion.items()},
        "timeline": timeline,
    }


# ---------------------------------------------------------------------------
# 2. ОБЪЕДИНЕНИЕ ОЦЕНОК
# ---------------------------------------------------------------------------

def combine_scores(text_result: dict, emotion_result: dict):
    text_score = _to_float(text_result.get("overall_score"))
    emotion_score = emotion_result["score"]

    if text_score is not None and emotion_score is not None:
        final = W_TEXT * text_score + W_EMOTION * emotion_score
    else:
        final = text_score if text_score is not None else emotion_score

    flags = []

    forbidden = text_result.get("forbidden_phrases") or []
    if forbidden:
        flags.append(f"Запрещённые фразы: {', '.join(forbidden)}")

    share = emotion_result["negative_share"]
    if share is not None and share > NEGATIVE_FLAG_THRESHOLD:
        flags.append(f"Высокая доля негативных эмоций в голосе менеджера: {share:.0%}")

    if emotion_score is None:
        flags.append("Оценка тона не рассчитана: нет достаточно надёжных реплик менеджера")

    return (round(final, 1) if final is not None else None), flags


# ---------------------------------------------------------------------------
# 3. ПОЛНЫЙ ОТЧЁТ
# ---------------------------------------------------------------------------

def build_report(audio_path: str) -> dict:
    segments, manager_label = process_call(audio_path)   # STT + диаризация — один раз

    ser = SERScorer()
    emotion_result = analyze_manager_emotions(audio_path, segments, manager_label, ser)
    text_result = analyze_transcript_with_llm(segments, manager_label)

    final_score, flags = combine_scores(text_result, emotion_result)

    return {
        "audio": audio_path,
        "final_score": final_score,
        "weights": {"text": W_TEXT, "emotion": W_EMOTION},
        "flags": flags,
        "text_analysis": text_result,
        "emotion_analysis": emotion_result,
        "transcript": [
            {
                "role": "manager" if s.speaker == manager_label else "client",
                "start": round(s.start, 1),
                "end": round(s.end, 1),
                "text": s.text,
            }
            for s in segments
        ],
    }


if __name__ == "__main__":
    AUDIO = r"D:\crowd_data\crowd\crowd_test\dialogs\holodniy_zvonok.mp3"

    report = build_report(AUDIO)

    print("\n=== ИТОГОВЫЙ ОТЧЁТ ===")
    print(f"Итоговая оценка: {report['final_score']} / 10")
    print(f"  текст (LLM):  {report['text_analysis'].get('overall_score')}")
    print(f"  тон (SER):    {report['emotion_analysis']['score']}")
    for flag in report["flags"]:
        print(f"  ⚠ {flag}")

    out_path = AUDIO.rsplit(".", 1)[0] + "_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nПолный отчёт сохранён: {out_path}")
