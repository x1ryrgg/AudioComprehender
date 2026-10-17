import logging

from celery import shared_task
from django.db import transaction
from django.utils import timezone

from call_recognition.models.call import Call, Evaluation, Transcript
from .services import run_analysis

logger = logging.getLogger(__name__)


def _save_report(call: Call, report: dict) -> None:
    segments = report.get("transcript") or []
    text = {k: v for k, v in (report.get("text_analysis") or {}).items() if k != "raw"}
    emotion = report.get("emotion_analysis") or {}

    with transaction.atomic():
        Transcript.objects.update_or_create(call=call, defaults={"segments": segments})
        Evaluation.objects.update_or_create(
            call=call,
            defaults={
                "final_score": report.get("final_score"),
                "text_score": text.get("overall_score"),
                "emotion_score": emotion.get("score"),
                "negative_share": emotion.get("negative_share"),
                "analyzer_version": report.get("analyzer_version") or "",
                "flags": report.get("flags") or [],
                "text_analysis": text,
                "emotion_analysis": emotion,
            },
        )
        call.duration = segments[-1]["end"] if segments else None
        call.status = Call.Status.DONE
        call.error = ""
        call.processed_at = timezone.now()
        call.save(update_fields=["duration", "status", "error", "processed_at"])


@shared_task(name="call_recognition.tasks.process_call",
             soft_time_limit=60 * 25,
             time_limit=60 * 30)
def process_call(call_id: int) -> None:
    call = Call.objects.filter(pk=call_id).first()
    if call is None or call.status == Call.Status.DONE:
        return

    call.status = Call.Status.PROCESSING
    call.error = ""
    call.save(update_fields=["status", "error"])

    try:
        report = run_analysis(call)
        _save_report(call, report)
    except Exception as exc:  # включая SoftTimeLimitExceeded
        logger.exception("Не удалось обработать звонок %s", call_id)
        call.status = Call.Status.FAILED
        call.error = str(exc)[:2000]
        call.processed_at = timezone.now()
        call.save(update_fields=["status", "error", "processed_at"])