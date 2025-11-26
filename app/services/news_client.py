from __future__ import annotations

from typing import List, Optional
from datetime import datetime, timedelta
import httpx
from dateutil.parser import parse as parse_date

from ..config import settings
from ..schemas import NewsSource


class NewsClient:
    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.newsapi_key

    async def search(self, query: str, from_days: int = 7, limit: int = 5) -> List[NewsSource]:
        if not self.api_key:
            # без ключа NewsAPI просто возвращаем пустой список
            return []

        params = {
            "q": query,
            "pageSize": limit,
            "sortBy": "relevancy",
            "language": "en",
            "from": (datetime.utcnow() - timedelta(days=from_days)).isoformat(timespec="seconds") + "Z",
        }

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                self.BASE_URL,
                params=params,
                headers={"X-Api-Key": self.api_key},
            )
            resp.raise_for_status()
            data = resp.json()

        articles = data.get("articles", [])
        result: List[NewsSource] = []
        for art in articles:
            published_at = art.get("publishedAt")
            dt = parse_date(published_at) if published_at else None
            result.append(
                NewsSource(
                    title=art.get("title") or "",
                    description=art.get("description"),
                    url=art.get("url"),
                    published_at=dt,
                    source_name=(art.get("source") or {}).get("name"),
                )
            )
        return result
