from django.utils.dateparse import parse_date

from call_recognition.models.call import Call


def calls_for_user(user):
    """Руководитель видит все звонки, консультант только свои."""
    qs = Call.objects.select_related("consultant", "evaluation")
    return qs if user.is_supervisor else qs.filter(consultant=user)


def _parse_date(value):
    try:
        return parse_date(value) if value else None
    except ValueError:
        return None


def filter_calls(qs, params):
    """Общие фильтры для списка звонков и статистики:
    ?status=done&consultant=<id>&analyzer_version=<версия>&date_from=2026-10-01&date_to=2026-10-31"""
    if params.get("status"):
        qs = qs.filter(status=params["status"])
    consultant = params.get("consultant")
    if consultant and str(consultant).isdigit():
        qs = qs.filter(consultant_id=int(consultant))
    if params.get("analyzer_version"):
        qs = qs.filter(evaluation__analyzer_version=params["analyzer_version"])
    if d := _parse_date(params.get("date_from")):
        qs = qs.filter(called_at__date__gte=d)
    if d := _parse_date(params.get("date_to")):
        qs = qs.filter(called_at__date__lte=d)
    return qs