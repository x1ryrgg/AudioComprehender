from django.db import transaction

from call_recognition.models.call import Call
from call_recognition.tasks import process_call


def enqueue_processing(call: Call) -> None:
    # on_commit: задача уйдёт в очередь только после записи звонка в БД,
    # иначе воркер может не найти объект
    transaction.on_commit(lambda: process_call.delay(call.pk))


def retry_processing(call: Call) -> bool:
    """Ставит звонок на повторную обработку. False, если он уже обрабатывается."""
    if call.status == Call.Status.PROCESSING:
        return False
    call.status = Call.Status.PENDING
    call.error = ""
    call.save(update_fields=["status", "error"])
    enqueue_processing(call)
    return True