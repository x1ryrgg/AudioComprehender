#!/bin/sh

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
    exec celery -A config worker -l info --concurrency="${WORKER_CONCURRENCY:-2}"
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