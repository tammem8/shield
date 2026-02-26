import csv
from pathlib import Path

import plotly.graph_objects as go

from shield.models.evaluation import EvaluationMetrics, PredictionResult

RESULTS_DIR = Path("results")


def save_results(results: list[PredictionResult], metrics: EvaluationMetrics) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    _save_csv(results)
    _save_confusion_matrix(metrics)
    return RESULTS_DIR


def _save_csv(results: list[PredictionResult]) -> None:
    path = RESULTS_DIR / "predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "predicted", "language"])
        for r in results:
            writer.writerow([r.text, r.true_label, r.predicted_label, r.language])


def _save_confusion_matrix(m: EvaluationMetrics) -> None:
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

    path = RESULTS_DIR / "confusion_matrix.html"
    fig.write_html(str(path))
