import csv
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from shield.models.evaluation import EvaluationMetrics, PredictionResult

RESULTS_DIR = Path("results")


def save_results(
    results: list[PredictionResult], metrics: EvaluationMetrics, dataset_name: str
) -> Path:
    output_dir = RESULTS_DIR / dataset_name.split(".")[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_csv(results, output_dir)
    _save_confusion_matrix(metrics, output_dir)
    _save_json(results, metrics, output_dir)
    return output_dir


def _save_csv(results: list[PredictionResult], output_dir: Path) -> None:
    path = output_dir / "predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "predicted", "score", "language"])
        for r in results:
            writer.writerow(
                [r.text, r.true_label, r.predicted_label, r.score, r.language]
            )


def _compute_metrics_dict(group: pd.DataFrame) -> dict:
    tp = int(((group["true_label"] == 1) & (group["predicted_label"] == 1)).sum())
    fp = int(((group["true_label"] == 0) & (group["predicted_label"] == 1)).sum())
    fn = int(((group["true_label"] == 1) & (group["predicted_label"] == 0)).sum())
    tn = int(((group["true_label"] == 0) & (group["predicted_label"] == 0)).sum())
    total = len(group)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "count": total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _save_json(results: list[PredictionResult], metrics: EvaluationMetrics, output_dir: Path) -> None:
    df = pd.DataFrame([r.model_dump() for r in results])
    per_language = {
        lang: _compute_metrics_dict(group) for lang, group in df.groupby("language")
    }
    output = {
        "global": {
            "count": metrics.total,
            "accuracy": round(metrics.accuracy, 4),
            "precision": round(metrics.precision, 4),
            "recall": round(metrics.recall, 4),
            "f1": round(metrics.f1, 4),
        },
        "per_language": per_language,
    }
    path = output_dir / "metrics.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def _save_confusion_matrix(m: EvaluationMetrics, output_dir: Path) -> None:
    z = [[m.tn, m.fp], [m.fn, m.tp]]
    labels = ["Benign", "Jailbreak"]
    text = [[f"TN = {m.tn}", f"FP = {m.fp}"], [f"FN = {m.fn}", f"TP = {m.tp}"]]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"Predicted: {l}" for l in labels],
            y=[f"Actual: {l}" for l in labels],
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis={"side": "bottom"},
        yaxis={"autorange": "reversed"},
    )

    path = output_dir / "confusion_matrix.html"
    fig.write_html(str(path))
