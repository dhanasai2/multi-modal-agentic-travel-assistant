from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class WeatherPoint(BaseModel):
    date: str
    temp_c: float
    condition: str


class TravelResponse(BaseModel):
    city: str
    city_summary: str
    weather_forecast: list[WeatherPoint] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    source: Literal["vector_store", "web_search", "memory"]
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
