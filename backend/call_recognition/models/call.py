from django.db import models
from django.utils import timezone


class Call(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "В очереди"
        PROCESSING = "processing", "Обрабатывается"
        DONE = "done", "Готов"
        FAILED = "failed", "Ошибка"

    audio_file = models.FileField(upload_to="calls/%Y/%m/")
    original_filename = models.CharField(max_length=255, blank=True)
    consultant = models.ForeignKey(
        "User",
        on_delete=models.PROTECT,
        related_name="calls",
    )
    # Дата самого звонка (можно передать при загрузке). Нужна для графиков по дням.
    called_at = models.DateTimeField(default=timezone.now, db_index=True)
    duration = models.FloatField(null=True, blank=True, help_text="секунды")
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING, db_index=True
    )
    error = models.TextField(blank=True)
    uploaded_by = models.ForeignKey(
        "User",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-called_at"]

    def __str__(self):
        return f"Call #{self.pk} ({self.status})"


class Transcript(models.Model):
    call = models.OneToOneField(Call, on_delete=models.CASCADE, related_name="transcript")
    # [{"role": "manager" | "client", "start": 0.0, "end": 4.2, "text": "..."}]
    segments = models.JSONField(default=list)


class Evaluation(models.Model):
    call = models.OneToOneField(Call, on_delete=models.CASCADE, related_name="evaluation")

    # Числовые поля вынесены отдельно, чтобы строить графики обычными агрегациями
    final_score = models.FloatField(null=True, blank=True, db_index=True)
    text_score = models.FloatField(null=True, blank=True)
    emotion_score = models.FloatField(null=True, blank=True)
    negative_share = models.FloatField(null=True, blank=True)

    # Версия анализатора (код LLM-части, веса), которой получена оценка
    analyzer_version = models.CharField(max_length=64, blank=True, db_index=True)

    flags = models.JSONField(default=list, blank=True)
    text_analysis = models.JSONField(default=dict, blank=True)  # asdict(TextAnalysis)
    emotion_analysis = models.JSONField(default=dict, blank=True)  # score, distribution, timeline...

    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Evaluation for call #{self.call_id}: {self.final_score}"