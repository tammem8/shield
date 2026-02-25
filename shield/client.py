import asyncio

import httpx

from shield.config.settings import get_settings
from shield.models import (
    DatasetRecord,
    PredictionResult,
    ShieldRequest,
    ShieldResponse,
)


async def call_api(
    client: httpx.AsyncClient,
    record: DatasetRecord,
    semaphore: asyncio.Semaphore,
) -> PredictionResult:
    settings = get_settings()
    async with semaphore:
        request = ShieldRequest(text=record.text)
        response = await client.post(
            settings.base_url,
            json=request.model_dump(),
            headers={"Authorization": f"Bearer {settings.api_key}"},
        )
        response.raise_for_status()
        data = ShieldResponse.model_validate(response.json())
        # data = ShieldResponse.model_validate({"jailbreak": {"class": record.label, "scores": [0]}})
        return PredictionResult(
            text=record.text,
            true_label=record.label,
            predicted_label=data.jailbreak.predicted_class,
            scores=data.jailbreak.scores,
        )
