from django.core.management.base import BaseCommand, CommandError

from call_recognition.models.call import Call
from utils.processing import retry_processing


class Command(BaseCommand):
    help = "Ставит звонки на повторную обработку (например, после доработки LLM или весов оценки)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--status", choices=[s.value for s in Call.Status],
            help="только звонки с этим статусом (например, failed)",
        )
        parser.add_argument(
            "--exclude-version",
            help="пропустить звонки, уже оценённые этой версией анализатора",
        )
        parser.add_argument("--all", action="store_true", help="перезапустить вообще все звонки")

    def handle(self, *args, **opts):
        if not (opts["status"] or opts["exclude_version"] or opts["all"]):
            raise CommandError("Укажите --status, --exclude-version или --all")

        qs = Call.objects.exclude(status=Call.Status.PROCESSING)
        if opts["status"]:
            qs = qs.filter(status=opts["status"])
        if opts["exclude_version"]:
            qs = qs.exclude(evaluation__analyzer_version=opts["exclude_version"])

        count = sum(1 for call in qs.iterator() if retry_processing(call))
        self.stdout.write(self.style.SUCCESS(f"Поставлено на повторную обработку: {count}"))