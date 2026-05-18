import random
import time
from datetime import datetime, timedelta


KNOWN_IMAGE_SETS: dict[str, list[str]] = {
    "paris": [
        "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1499856871958-5b9627545d1a?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1522093007474-d86e9bf7ba6f?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1431274172761-fca41d930114?auto=format&fit=crop&w=1200&q=80",
    ],
    "tokyo": [
        "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1513407030348-c983a97b98d8?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1493514789931-586cb221d7a7?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1536098561742-ca998e48cbcc?auto=format&fit=crop&w=1200&q=80",
    ],
    "new york": [
        "https://images.unsplash.com/photo-1496588152823-e98c45dd467f?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1485871981521-5b1fd3805eee?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?auto=format&fit=crop&w=1200&q=80",
        "https://images.unsplash.com/photo-1448317846460-907988886b33?auto=format&fit=crop&w=1200&q=80",
    ],
}

KNOWN_WEB_FACTS: dict[str, str] = {
    "kyoto": (
        "Kyoto blends quiet temple districts, tea houses, and seasonal gardens with a "
        "walkable city core. Spring cherry blossoms and autumn foliage are peak travel "
        "periods, while neighborhoods like Gion and Arashiyama are favorites for first-time visitors."
    ),
    "snohomish": (
        "Snohomish is a small Washington town known for historic downtown charm, "
        "river views, antique shops, and quick access to outdoor trails and mountain day trips."
    ),
    "barcelona": (
        "Barcelona offers beach-meets-city energy with Gaudi landmarks, tapas streets, "
        "and lively evening culture concentrated around the Gothic Quarter and Eixample."
    ),
}


def _simulate_latency(min_seconds: float = 0.2, max_seconds: float = 0.7) -> None:
    time.sleep(random.uniform(min_seconds, max_seconds))


def mock_web_search(city: str) -> str:
    _simulate_latency()
    key = city.strip().lower()
    if key in KNOWN_WEB_FACTS:
        return KNOWN_WEB_FACTS[key]
    return (
        f"{city.title()} is a compelling destination with local culture, food, and landmarks. "
        "Travelers usually benefit from planning around weather seasonality and neighborhood-based itineraries."
    )


def get_location_images(city: str, count: int = 4) -> list[str]:
    _simulate_latency()
    key = city.strip().lower()
    if key in KNOWN_IMAGE_SETS:
        urls = KNOWN_IMAGE_SETS[key]
    else:
        urls = [
            f"https://picsum.photos/seed/{key.replace(' ', '-')}-travel/1200/800",
            f"https://picsum.photos/seed/{key.replace(' ', '-')}-city/1200/800",
            f"https://picsum.photos/seed/{key.replace(' ', '-')}-landmark/1200/800",
            f"https://picsum.photos/seed/{key.replace(' ', '-')}-nature/1200/800",
        ]
    return urls[: max(1, min(count, len(urls)))]


def get_weather_forecast(city: str, days: int = 7) -> list[dict]:
    _simulate_latency()
    safe_days = max(1, min(days, 7))
    seed = sum(ord(ch) for ch in city.lower())
    rnd = random.Random(seed)
    start_temp = rnd.uniform(10, 30)
    base_date = datetime.now()
    conditions = ["Sunny", "Cloudy", "Rain", "Partly Cloudy", "Windy"]

    forecast: list[dict] = []
    for i in range(safe_days):
        trend = rnd.uniform(-2.5, 2.5)
        temp = round(start_temp + trend + (i * 0.2), 1)
        forecast.append(
            {
                "date": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                "temp_c": temp,
                "condition": rnd.choice(conditions),
            }
        )
    return forecast
