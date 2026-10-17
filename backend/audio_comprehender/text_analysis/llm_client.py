from openai import OpenAI

from audio_comprehender.config import YANDEX_API_KEY, YANDEX_FOLDER_ID

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=YANDEX_API_KEY, base_url="https://ai.api.cloud.yandex.net/v1")
    return _client


def get_model_uri() -> str:
    return f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest"
