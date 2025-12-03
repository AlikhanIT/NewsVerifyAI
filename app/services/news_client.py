from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List

import httpx
from dateutil.parser import parse as parse_date

from ..config import settings
from ..schemas import NewsSource


logger = logging.getLogger(__name__)


class NewsClient:
    def __init__(self):
        self.api_key = settings.NEWSAPI_KEY
        self.api_base = "https://newsapi.org/v2"
        self.timeout = 10.0

        print("📰 NewsClient init:")
        print(f"   API Base: {self.api_base}")
        print(f"   API Key present: {bool(self.api_key)}")
        logger.info("📰 NewsClient init:")
        logger.info(f"   API Base: {self.api_base}")
        logger.info(f"   API Key present: {bool(self.api_key)}")

    async def search(self, query: str, from_days: int = 7, limit: int = 5) -> List[NewsSource]:
        if not self.api_key:
            logger.error("❌ NEWSAPI_KEY not set")
            return []

        from_date = (datetime.now() - timedelta(days=from_days)).strftime("%Y-%m-%d")

        logger.info("📰 NEWS API ЗАПРОС:")
        logger.info(f"   🔗 URL: {self.api_base}/everything")
        logger.info(f"   📋 Параметры:")
        logger.info(f"      - query: {query}")
        logger.info(f"      - from: {from_date}")
        logger.info(f"      - sortBy: relevancy")
        logger.info(f"      - pageSize: {limit}")
        logger.info(f"      - language: ru")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{self.api_base}/everything",
                    params={
                        "q": query,
                        "from": from_date,
                        "sortBy": "relevancy",
                        "pageSize": limit,
                        "apiKey": self.api_key,
                        "language": "ru",
                    },
                )

                logger.info(f"   📡 ОТВЕТ News API:")
                logger.info(f"      - Статус: {resp.status_code}")
                logger.info(f"      - Время ответа: {resp.elapsed.total_seconds():.2f}s")

                if resp.status_code != 200:
                    logger.error(f"      ❌ Ошибка: {resp.text[:200]}")
                    return []

                data = resp.json()
                articles = data.get("articles", [])
                total_results = data.get("totalResults", 0)

                logger.info(f"      ✅ Найдено статей: {total_results}")
                logger.info(f"      📄 Получено статей: {len(articles)}")

                results: List[NewsSource] = []
                for idx, a in enumerate(articles, 1):
                    title = a.get("title") or "Без названия"
                    desc = a.get("description")
                    content = a.get("content")
                    url = a.get("url")
                    published_at = a.get("publishedAt")
                    source_name = a.get("source", {}).get("name")

                    summary = content or desc
                    published_dt = parse_date(published_at) if published_at else None

                    news_source = NewsSource(
                        title=title,
                        description=desc,
                        summary=summary,
                        url=url,
                        published_at=published_dt,
                        source_name=source_name,
                    )
                    results.append(news_source)
                    logger.info(
                        "         %s. [%s] %s",
                        idx,
                        source_name or "Unknown",
                        title[:60],
                    )

                return results
        except Exception as e:
            logger.error(f"   ❌ Ошибка запроса к News API: {e}")
            return []
