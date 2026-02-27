import httpx
from tqdm import tqdm

from shield.client import call_api
from shield.models.evaluation import DatasetRecord, EvaluationMetrics, PredictionResult


class Evaluator:
    def __init__(self, records: list[DatasetRecord]) -> None:
        self.records = records

    def run(
        self, threshold: float | None = None
    ) -> tuple[list[PredictionResult], EvaluationMetrics]:
        with httpx.Client(timeout=30.0, verify=False) as client:
            results: list[PredictionResult] = [
                call_api(client, record)
                for record in tqdm(self.records, desc="Evaluating")
            ]

        if threshold is not None:
            results = [
                r.model_copy(update={"predicted_label": int(r.score > threshold)})
                for r in results
            ]

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
