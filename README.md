Проверка эмоций - poetry run python -m audio_comprehender.training.train_ser

тестирование poetry run python -m audio_comprehender.tests 

полный анализ - python\AudioComprehender> python run_analysis.py "D:\crowd_data\crowd\crowd_test\dialogs\Phone_ARU_ON.wav"

# Audio Comprehender

Сервис оценки качества звонков консультантов приёмной комиссии. Загруженная запись проходит конвейер:
распознавание речи и разделение по ролям, оценка тона голоса (SER), LLM-анализ текста, итоговый балл.
Результаты сохраняются в БД и доступны через API для графиков и статистики.

Всё работает в Docker: `web` (Django + gunicorn), `worker` (Celery + анализатор на GPU), `beat`,
`postgres`, `redis`, `nginx`, а также `flower` (по желанию).

## Первый запуск

В корне репозитория (PowerShell):

```powershell
copy .env.example .env    # заполни SECRET_KEY, DB_PASSWORD, HF_TOKEN, YANDEX_API_KEY, YANDEX_FOLDER_ID
docker compose build      # первый раз долго: скачиваются torch и библиотеки CUDA
docker compose up --no-start

# один раз перенести кэш моделей Hugging Face с ПК в том (около 2 ГБ);
# без этого шага модели скачаются при первом звонке
docker run --rm -v audio-comprehender_hf_cache:/dst -v "$env:USERPROFILE\.cache\huggingface:/src:ro" alpine sh -c "cp -a /src/. /dst/"

docker compose up -d
docker compose exec web python manage.py createsuperuser
```

Файл `.env` содержит секреты и в git не коммитится.

## Адреса

| Что | Адрес |
|---|---|
| API и Swagger | http://localhost/api/v1/docs/ |
| Проверка работоспособности | http://localhost/health/ |
| Аудиофайлы | http://localhost/media/... |
| Flower (мониторинг Celery) | http://localhost:5555, запускается отдельно (см. ниже) |

Если порт 80 занят, поставь в `.env` `HTTP_PORT=8080` и добавь `http://localhost:8080` в `CSRF_TRUSTED_ORIGINS`.

## Повседневная работа

```powershell
docker compose up -d                          # запустить
docker compose stop                           # остановить (данные сохраняются)
docker compose ps                             # состояние сервисов
docker compose logs -f worker                 # ход обработки звонков (логи web: logs -f web)
docker compose up -d --build                  # применить изменения кода
docker compose --profile tools up -d flower   # запустить Flower
```

`docker compose down` удаляет контейнеры, но данные сохраняются. Команда `docker compose down -v`
удаляет ещё и тома: БД, загруженные звонки и кэш моделей.

## Как пользоваться

1. Создай пользователей в админке Django: консультантов (роль «Консультант») и при необходимости руководителя.
2. Открой Swagger, получи JWT-токен, нажми Authorize и вставь токен (без слова `Bearer`).
3. Загрузи звонок (`multipart/form-data`, поле `audio_file`). Руководитель указывает ещё и консультанта,
   консультант загружает звонки от своего имени.
4. Статус звонка меняется `pending → processing → done` (или `failed` с текстом ошибки).
   Готовые оценки, транскрипт и статистика доступны через API.

Пакетная загрузка папки со звонками:

```powershell
docker compose cp D:\crowd_data\crowd\crowd_test\dialogs web:/tmp/dialogs
docker compose exec web python manage.py import_calls /tmp/dialogs --consultants 2 3 --limit 3
```

Пересчитать звонки после доработки анализатора: `docker compose exec web python manage.py reprocess_calls --all`
(или `--status failed`, `--exclude-version <версия>`).

## Версия анализатора

Внутри образа нет git, поэтому версию (поле `analyzer_version`) передаём при запуске. Сначала закоммить
изменения в `backend/audio_comprehender`, затем:

```powershell
$env:ANALYZER_VERSION = (git log -1 --format=%h -- backend/audio_comprehender)
docker compose up -d
```

## Если что-то не работает

- Проверка GPU и моделей: `docker compose run --rm worker python scripts/gpu_check.py`
- `CUDA out of memory` в воркере: поставь в `.env` `WORKER_CONCURRENCY=1`
- `DisallowedHost`: проверь `DJANGO_ALLOWED_HOSTS` в `.env`

Подробности и таблица типовых проблем: [DOCKER.md](DOCKER.md).