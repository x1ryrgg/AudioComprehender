#!/bin/sh
# Точка входа контейнера: первый аргумент выбирает роль сервиса.
set -e

case "$1" in
  web)
    python manage.py migrate --noinput
    python manage.py collectstatic --noinput
    exec gunicorn config.wsgi:application \
      --bind 0.0.0.0:8000 \
      --workers "${GUNICORN_WORKERS:-2}" \
      --timeout 120 \
      --access-logfile -
    ;;
  worker)
    concurrency="${WORKER_CONCURRENCY:-1}"
    # Расчёты идут на CPU: делим ядра между процессами воркера, чтобы они не мешали друг другу.
    # Можно задать вручную переменной OMP_NUM_THREADS в .env.
    if [ -z "$OMP_NUM_THREADS" ]; then
      threads=$(( $(nproc) / concurrency ))
      if [ "$threads" -lt 1 ]; then
        threads=1
      fi
      export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads"
    fi
    echo "worker: процессов $concurrency, потоков CPU на процесс $OMP_NUM_THREADS"
    exec celery -A config worker -l info --concurrency="$concurrency"
    ;;
  beat)
    exec celery -A config beat -l info -s /tmp/celerybeat-schedule
    ;;
  flower)
    exec celery -A config flower --port=5555
    ;;
  *)
    exec "$@"
    ;;
esac