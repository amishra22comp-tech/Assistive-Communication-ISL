from pydantic import BaseModel

class PredictionResponse(BaseModel):
    success: bool
    sentence: str | None = None
    confidence: float | None = None
    message: str | None = None
