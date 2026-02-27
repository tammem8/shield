from pydantic import BaseModel


class DatasetRecord(BaseModel):
    text: str
    label: int  # 0 = benign, 1 = jailbreak
    language: str | None = None


class PredictionResult(BaseModel):
    text: str
    true_label: int
    predicted_label: int
    score: float
    language: str


class EvaluationMetrics(BaseModel):
    total: int
    tp: int
    fp: int
    fn: int
    tn: int
    accuracy: float
    precision: float
    recall: float
    f1: float
