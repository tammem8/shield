from pydantic import BaseModel, Field


class ShieldRequest(BaseModel):
    text: str


class ShieldClassification(BaseModel):
    predicted_class: int = Field(alias="class")
    scores: list[int]

    model_config = {"populate_by_name": True}


class ShieldResponse(BaseModel):
    jailbreak: ShieldClassification


class DatasetRecord(BaseModel):
    text: str
    label: int  # 0 = benign, 1 = jailbreak


class PredictionResult(BaseModel):
    text: str
    true_label: int
    predicted_label: int
    scores: list[int]


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
