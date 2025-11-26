from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from datetime import datetime


class CheckRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Утверждение для проверки")
    style: Literal["formal", "simple"] = "simple"


class NewsSource(BaseModel):
    title: str
    description: str | None = None
    url: str
    published_at: datetime | None = None
    source_name: str | None = None


class ClaimAnalysis(BaseModel):
    status: Literal["confirmed", "not_found", "uncertain"]
    probability: float | None = Field(
        None, ge=0.0, le=1.0, description="Оценка вероятности события"
    )
    explanation: str
    matched_sources: List[NewsSource] = []


class CheckResponse(BaseModel):
    claim: str
    style: Literal["formal", "simple"]
    analysis: ClaimAnalysis
    cached: bool = False
