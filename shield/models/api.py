from typing import Annotated

from pydantic import BaseModel, Field


class ShieldRequest(BaseModel):
    text: str


class ShieldClassification(BaseModel):
    predicted_class: int = Field(alias="class")
    scores: list[Annotated[float, Field(ge=0.0, le=1.0)]]

    model_config = {"populate_by_name": True}


class ShieldResponse(BaseModel):
    jailbreak: ShieldClassification
    xpia: ShieldClassification
