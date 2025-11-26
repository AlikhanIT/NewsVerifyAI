from __future__ import annotations
from typing import Optional, Tuple
from ..config import settings

from openai import OpenAI  # новое
import json


class LLMClient:
    def __init__(self):
        import openai
        from app.config import settings

        openai.api_key = settings.openai_api_key or ""
        self.client = openai if settings.openai_api_key else None
        self.model = "gpt-3.5-turbo"  # или твоя

    async def analyze(self, prompt: str):
        if not self.client:
            return 0.3, "AI-анализ недоступен. Использована эвристическая оценка."

        try:
            response = await self.client.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            result = response.choices[0].message.content
            return 0.7, result  # парсить если хочешь
        except Exception:
            return 0.3, "Анализ через AI временно недоступен. Использован fallback."

