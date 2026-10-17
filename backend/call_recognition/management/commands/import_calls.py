import random
from datetime import timedelta
from pathlib import Path

from call_recognition.models.user import User
from django.core.files import File
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from call_recognition.models.call import Call
from call_recognition.utils.processing import enqueue_processing
from call_recognition.utils.constants import ALLOWED_EXTENSIONS


# python manage.py import_calls D:\crowd_data\crowd\crowd_test\dialogs --consultants 2 3 4 --spread-days 30 --skip-existing

class Command(BaseCommand):
    help = "Загружает все аудиофайлы из папки как звонки и ставит их на обработку"

    def add_arguments(self, parser):
        parser.add_argument("directory", help="папка с аудиофайлами")
        parser.add_argument(
            "--consultants", nargs="+", type=int, required=True,
            help="id консультантов; звонки распределяются между ними по кругу",
        )
        parser.add_argument("--recursive", action="store_true", help="искать и во вложенных папках")
        parser.add_argument(
            "--spread-days", type=int, default=0,
            help="проставить звонкам случайные даты за последние N дней (для наглядных графиков)",
        )
        parser.add_argument("--limit", type=int, default=0, help="загрузить не больше N файлов")
        parser.add_argument(
            "--skip-existing", action="store_true",
            help="пропускать файлы, чьё имя уже есть среди загруженных звонков",
        )
        parser.add_argument("--uploaded-by", type=int, help="id пользователя, от имени которого загружаем")

    def handle(self, *args, **opts):
        directory = Path(opts["directory"])
        if not directory.is_dir():
            raise CommandError(f"Папка не найдена: {directory}")

        found = {
            u.pk: u
            for u in User.objects.filter(pk__in=opts["consultants"], role=User.UserRole.CONSULTANT)
        }
        missing = sorted(set(opts["consultants"]) - set(found))
        if missing:
            raise CommandError(f"Не найдены консультанты с id: {missing}")
        consultants = [found[pk] for pk in opts["consultants"]]

        uploaded_by = None
        if opts["uploaded_by"]:
            uploaded_by = User.objects.filter(pk=opts["uploaded_by"]).first()
            if uploaded_by is None:
                raise CommandError(f"Не найден пользователь с id {opts['uploaded_by']}")

        pattern = "**/*" if opts["recursive"] else "*"
        files = sorted(
            p for p in directory.glob(pattern)
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
        )
        if opts["skip_existing"]:
            existing = set(Call.objects.values_list("original_filename", flat=True))
            files = [p for p in files if p.name not in existing]
        if opts["limit"]:
            files = files[: opts["limit"]]
        if not files:
            self.stdout.write("Подходящих файлов не найдено")
            return

        now = timezone.now()
        spread = opts["spread_days"]
        for index, path in enumerate(files):
            called_at = now - timedelta(days=random.uniform(0, spread)) if spread else now
            call = Call(
                consultant=consultants[index % len(consultants)],
                uploaded_by=uploaded_by,
                original_filename=path.name,
                called_at=called_at,
            )
            with path.open("rb") as fh:
                call.audio_file.save(path.name, File(fh))  # сохраняет файл и запись в БД
            enqueue_processing(call)
            self.stdout.write(f"[{index + 1}/{len(files)}] {path.name} -> звонок #{call.pk} ({call.consultant})")

        self.stdout.write(self.style.SUCCESS(
            f"Поставлено в очередь: {len(files)}. Следи за воркером Celery, обработка идёт по одному звонку."
        ))