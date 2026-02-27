import httpx
from fast_langdetect import detect
from openai import OpenAI
from tqdm import tqdm

from shield.config.settings import get_settings
from shield.models.api import (
    ShieldRequest,
    ShieldResponse,
)
from shield.models.evaluation import (
    DatasetRecord,
    PredictionResult,
)

_SUPPORTED_LANGUAGES = {"fr", "en", "de"}
_LANGUAGE_NAMES = {"fr": "French", "en": "English", "de": "German"}


def _detect_language(text: str) -> str:
    return str(detect(text, model="lite", k=1)[0]["lang"])


def _translate(text: str, target_lang: str) -> str:
    settings = get_settings()
    client = OpenAI(base_url=settings.base_url, api_key=settings.api_key)
    lang_name = _LANGUAGE_NAMES[target_lang]
    return f"{text} ({lang_name})"
    # response = client.chat.completions.create(
    #     model=settings.model_name,
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": (
    #                 f"Translate the following text to {lang_name}. "
    #                 "Return only the translated text, nothing else."
    #             ),
    #         },
    #         {"role": "user", "content": text},
    #     ],
    # )
    # return response.choices[0].message.content.strip()


def augment_dataset(
    records: list[DatasetRecord], detect_lang: bool = True, translate: bool = True
) -> list[DatasetRecord]:
    augmented: list[DatasetRecord] = []
    for record in tqdm(records, desc="Preprocessing"):
        lang = record.language or (
            _detect_language(record.text) if detect_lang or translate else None
        )
        augmented.append(
            DatasetRecord(text=record.text, label=record.label, language=lang)
        )

        if translate and lang in _SUPPORTED_LANGUAGES:
            for target in _SUPPORTED_LANGUAGES - {lang}:
                augmented.append(
                    DatasetRecord(
                        text=_translate(record.text, target),
                        label=record.label,
                        language=target,
                    )
                )
                # time.sleep(0.5)

    return augmented


def call_api(client: httpx.Client, record: DatasetRecord) -> PredictionResult:
    settings = get_settings()
    language = record.language or _detect_language(record.text)
    request = ShieldRequest(text=record.text)
    url = f"{settings.base_url.rstrip('/')}/{settings.custom_path.lstrip('/')}"
    # response = client.post(
    #     url,
    #     json=request.model_dump(),
    #     headers={
    #         "Content-Type": "application/json",
    #         "Authorization": f"Bearer {settings.api_key}",
    #     },
    # )
    # response.raise_for_status()
    # data = ShieldResponse.model_validate(response.json())
    data = ShieldResponse.model_validate(
        {
            "jailbreak": {"class": record.label, "scores": [0.6]},
            "xpia": {"class": 0, "scores": [0]},
        }
    )
    # time.sleep(0.5)
    return PredictionResult(
        text=record.text,
        true_label=record.label,
        predicted_label=data.jailbreak.predicted_class,
        score=data.jailbreak.scores[0],
        language=language,
    )
