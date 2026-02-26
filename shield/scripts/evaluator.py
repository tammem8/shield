import asyncio

import httpx
from tqdm.asyncio import tqdm_asyncio

from shield.client import call_api
from shield.config.settings import get_settings
from shield.models.evaluation import DatasetRecord, EvaluationMetrics, PredictionResult


class Evaluator:
    def __init__(self, records: list[DatasetRecord]) -> None:
        self.records = records

    async def run(self) -> tuple[list[PredictionResult], EvaluationMetrics]:
        settings = get_settings()
        semaphore = asyncio.Semaphore(settings.concurrency)

        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            tasks = [call_api(client, record, semaphore) for record in self.records]
            results: list[PredictionResult] = await tqdm_asyncio.gather(
                *tasks, desc="Evaluating"
            )

        return results, self.compute_metrics(results)

    def compute_metrics(self, results: list[PredictionResult]) -> EvaluationMetrics:
        tp = sum(1 for r in results if r.true_label == 1 and r.predicted_label == 1)
        fp = sum(1 for r in results if r.true_label == 0 and r.predicted_label == 1)
        fn = sum(1 for r in results if r.true_label == 1 and r.predicted_label == 0)
        tn = sum(1 for r in results if r.true_label == 0 and r.predicted_label == 0)
        total = len(results)

        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return EvaluationMetrics(
            total=total,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
        )
