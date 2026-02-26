import asyncio
import time

import httpx
from ftlangdetect import detect

from shield.config.settings import get_settings
from shield.models.api import (
    ShieldRequest,
    ShieldResponse,
)
from shield.models.evaluation import (
    DatasetRecord,
    PredictionResult,
)


async def call_api(
    client: httpx.AsyncClient,
    record: DatasetRecord,
    semaphore: asyncio.Semaphore,
) -> PredictionResult:
    settings = get_settings()
    async with semaphore:
        language = str(detect(record.text)["lang"])
        request = ShieldRequest(text=record.text)
        url = f"{settings.base_url.rstrip('/')}/{settings.custom_path.lstrip('/')}"
        response = await client.post(
            url,
            json=request.model_dump(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.api_key}",
            },
        )
        response.raise_for_status()
        data = ShieldResponse.model_validate(response.json())
        # data = ShieldResponse.model_validate(
        #     {
        #         "jailbreak": {"class": record.label, "scores": [0]},
        #         "xpia": {"class": 0, "scores": [0]}
        #     }
        # )
        return PredictionResult(
            text=record.text,
            true_label=record.label,
            predicted_label=data.jailbreak.predicted_class,
            scores=data.jailbreak.scores,
            language=language,
        )
