"""Pydantic schemas for API request/response."""
from typing import Optional, Any
from pydantic import BaseModel


class MovieSchema(BaseModel):
    """Movie schema."""
    id: Any
    title: str
    originalTitle: Optional[str] = None
    description: Optional[str] = None
    categories: Optional[list[dict]] = None
    country: Optional[str] = None
    year: Optional[int] = None
    metadata: Optional[str] = None

    class Config:
        from_attributes = True


class MovieHistorySchema(BaseModel):
    """Movie history item schema."""
    movie: MovieSchema
    timesWatched: Optional[int] = 0
    lastWatchSeconds: Optional[int] = 0
    modifiedDate: Optional[str] = None

    class Config:
        from_attributes = True


class RecommendationResponse(BaseModel):
    """Recommendation API response schema."""
    data: list[dict]
