"""
ГЛАВНАЯ ТОЧКА ВХОДА В СЕРВИС.

Любая обёртка (Django view, Telegram-хендлер, CLI, Celery-задача) должна
дёргать ТОЛЬКО analyze_call() из этого файла — и ничего не знать про то,
что внутри используются whisper/pyannote/YandexGPT. Это и есть та граница,
о которой шла речь: "логика как сервис, который можно подключить к чему
угодно сверху".

Пример использования в Django view:

    from audio_comprehender.report import analyze_call

    def upload_call(request):
        audio_path = save_uploaded_file(request.FILES["audio"])
        report = analyze_call(audio_path)
        return JsonResponse(report.to_dict())

Пример использования в Celery-задаче:

    @shared_task
    def process_call_task(audio_path: str, call_id: int):
        report = analyze_call(audio_path)
        Call.objects.filter(id=call_id).update(
            score=report.final_score,
            report_json=report.to_dict(),
        )
"""

from dataclasses import dataclass, asdict

from audio_comprehender.asr.service import transcribe_call
from audio_comprehender.asr.types import CallTranscript
from audio_comprehender.emotion.model import SERScorer
from audio_comprehender.emotion.service import analyze_manager_emotions, EmotionAnalysis
from audio_comprehender.text_analysis.service import analyze_transcript_with_llm, TextAnalysis
from audio_comprehender.scoring import combine_scores

# SER-модель тяжёлая (веса wav2vec2) — грузим один раз на процесс,
# а не при каждом вызове analyze_call()
_ser_scorer: SERScorer | None = None


def _get_ser_scorer() -> SERScorer:
    global _ser_scorer
    if _ser_scorer is None:
        _ser_scorer = SERScorer()
    return _ser_scorer


@dataclass
class CallReport:
    audio_path: str
    final_score: float | None
    flags: list[str]
    text_analysis: TextAnalysis
    emotion_analysis: EmotionAnalysis
    transcript: CallTranscript

    def to_dict(self) -> dict:
        """Плоский словарь, готовый под JsonResponse / json.dump / сохранение в БД."""
        return {
            "audio_path": self.audio_path,
            "final_score": self.final_score,
            "flags": self.flags,
            "text_analysis": asdict(self.text_analysis),
            "emotion_analysis": {
                "score": self.emotion_analysis.score,
                "negative_share": self.emotion_analysis.negative_share,
                "distribution": self.emotion_analysis.distribution,
                "timeline": [asdict(p) for p in self.emotion_analysis.timeline],
            },
            "transcript": [
                {
                    "role": "manager" if s.speaker == self.transcript.manager_speaker else "client",
                    "start": round(s.start, 1),
                    "end": round(s.end, 1),
                    "text": s.text,
                }
                for s in self.transcript.segments
            ],
        }


def analyze_call(audio_path: str, manager_speaker: str | None = None) -> CallReport:
    """
    Полный анализ звонка: ASR -> эмоции + текстовый LLM-анализ -> итоговый балл.

    manager_speaker: передай явно ("SPEAKER_00"/"SPEAKER_01"), если роль
    известна заранее — надёжнее автоматической эвристики.
    """
    transcript = transcribe_call(audio_path, manager_speaker=manager_speaker)

    ser = _get_ser_scorer()
    emotion_result = analyze_manager_emotions(audio_path, transcript, ser)
    text_result = analyze_transcript_with_llm(transcript)

    scoring = combine_scores(text_result, emotion_result)

    return CallReport(
        audio_path=audio_path,
        final_score=scoring.final_score,
        flags=scoring.flags,
        text_analysis=text_result,
        emotion_analysis=emotion_result,
        transcript=transcript,
    )
