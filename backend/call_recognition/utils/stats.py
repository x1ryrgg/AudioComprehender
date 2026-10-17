from collections import Counter, defaultdict

from django.db.models import Avg, Count
from django.db.models.functions import TruncDate

# Критерии текстового анализа (ключи из text_analysis) и подписи для графиков
CRITERIA = {
    "greeting_and_introduction": "Приветствие и представление",
    "needs_identification": "Выяснение потребности",
    "answer_quality": "Качество ответа",
    "empathy_and_politeness": "Вежливость и эмпатия",
    "next_steps_and_closing": "Следующие шаги и завершение",
}


def _round(value, digits=2):
    return round(value, digits) if isinstance(value, float) else value


def _criteria_avgs(analyses):
    """Список text_analysis -> {ключ критерия: среднее значение}."""
    result = {}
    for key in CRITERIA:
        values = [a[key] for a in analyses if isinstance(a.get(key), (int, float))]
        result[key] = round(sum(values) / len(values), 2) if values else None
    return result


def _flag_type(flag: str) -> str:
    # Флаги имеют вид "Тип флага: подробности", группируем по типу
    return flag.split(":", 1)[0].strip()


def _consultant_name(row) -> str:
    full = f"{row['consultant__first_name']} {row['consultant__last_name']}".strip()
    return full or row["consultant__username"]


def build_stats(qs) -> dict:
    """Агрегаты для графиков. qs: уже отфильтрованные обработанные звонки (status=done)."""
    overall = {
        key: _round(value)
        for key, value in qs.aggregate(
            calls=Count("id"),
            avg_score=Avg("evaluation__final_score"),
            avg_text_score=Avg("evaluation__text_score"),
            avg_emotion_score=Avg("evaluation__emotion_score"),
            avg_negative_share=Avg("evaluation__negative_share"),
        ).items()
    }

    by_day = [
        {**row, "avg_score": _round(row["avg_score"])}
        for row in qs.annotate(day=TruncDate("called_at"))
        .values("day")
        .annotate(calls=Count("id"), avg_score=Avg("evaluation__final_score"))
        .order_by("day")
    ]

    consultant_rows = (
        qs.values(
            "consultant_id",
            "consultant__first_name",
            "consultant__last_name",
            "consultant__username",
        )
        .annotate(calls=Count("id"), avg_score=Avg("evaluation__final_score"))
        .order_by("-avg_score")
    )

    # Критерии и флаги лежат в JSON, поэтому считаем их в Python
    # (для объёмов дипломной работы этого достаточно)
    rows = list(qs.values_list("consultant_id", "evaluation__text_analysis", "evaluation__flags"))

    analyses = [analysis or {} for _, analysis, _ in rows]
    overall_criteria = _criteria_avgs(analyses)
    criteria = [
        {"key": key, "label": label, "avg": overall_criteria[key]}
        for key, label in CRITERIA.items()
    ]

    analyses_by_consultant = defaultdict(list)
    for consultant_id, analysis, _ in rows:
        analyses_by_consultant[consultant_id].append(analysis or {})

    by_consultant = [
        {
            "consultant_id": row["consultant_id"],
            "name": _consultant_name(row),
            "calls": row["calls"],
            "avg_score": _round(row["avg_score"]),
            "criteria": _criteria_avgs(analyses_by_consultant[row["consultant_id"]]),
        }
        for row in consultant_rows
    ]

    flag_counter = Counter(_flag_type(flag) for _, _, flags in rows for flag in (flags or []))
    flag_types = [{"type": t, "count": c} for t, c in flag_counter.most_common()]

    # Оценки разных версий анализатора несравнимы, поэтому отдаём список версий для фильтра на фронте
    versions = [
        {"version": row["evaluation__analyzer_version"] or "unknown", "calls": row["calls"]}
        for row in qs.values("evaluation__analyzer_version")
        .annotate(calls=Count("id"))
        .order_by("-calls")
    ]

    return {
        "overall": overall,
        "criteria": criteria,
        "by_day": by_day,
        "by_consultant": by_consultant,
        "flag_types": flag_types,
        "analyzer_versions": versions,
    }