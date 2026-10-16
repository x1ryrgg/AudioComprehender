from django.contrib import admin

from .models.call import Call, Evaluation, Transcript


@admin.register(Call)
class CallAdmin(admin.ModelAdmin):
    list_display = ("id", "original_filename", "consultant", "status", "called_at")
    list_filter = ("status", "consultant")
    search_fields = ("original_filename", "consultant__username", "consultant__last_name")


@admin.register(Evaluation)
class EvaluationAdmin(admin.ModelAdmin):
    list_display = ("call", "final_score", "text_score", "emotion_score")


admin.site.register(Transcript)