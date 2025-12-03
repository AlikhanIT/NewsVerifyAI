from __future__ import annotations

import logging
from typing import Optional, Tuple

import httpx

from ..config import settings


logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        # Используем только OPENAI_API_KEY из config
        self.api_key = settings.OPENAI_API_KEY
        # Базовый URL и модель жёстко заданы
        self.api_base = "https://api.openai.com/v1"
        self.model = "gpt-3.5-turbo"
        self.timeout = 30.0

        print("🤖 LLMClient init:")
        print(f"   API Base: {self.api_base}")
        print(f"   Model: {self.model}")
        print(f"   API Key present: {bool(self.api_key)}")

    async def analyze(self, prompt: str) -> Optional[Tuple[float, str]]:
        """
        Анализирует утверждение через LLM API.
        Возвращает (вероятность, объяснение) или None при ошибке.
        """
        if not self.api_key:
            logger.error("❌ OPENAI_API_KEY not set")
            return None

        logger.info("🤖 OPENAI API ЗАПРОС:")
        logger.info(f"   🔗 URL: {self.api_base}/chat/completions")
        logger.info(f"   📋 Параметры:")
        logger.info(f"      - model: {self.model}")
        logger.info(f"      - temperature: 0.3")
        logger.info(f"      - max_tokens: 500")
        logger.info(f"   💬 Промпт (первые 200 символов):")
        logger.info(f"      {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

        try:
            request_payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a fact-checker."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 500,
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                )

                logger.info(f"   📡 ОТВЕТ OpenAI API:")
                logger.info(f"      - Статус: {resp.status_code}")
                logger.info(f"      - Время ответа: {resp.elapsed.total_seconds():.2f}s")

                if resp.status_code != 200:
                    logger.error(f"      ❌ Ошибка: {resp.text[:300]}")
                    return None

                data = resp.json()

                # Логируем использование токенов
                usage = data.get("usage", {})
                if usage:
                    logger.info(f"      📊 Использовано токенов:")
                    logger.info(f"         - prompt: {usage.get('prompt_tokens', 0)}")
                    logger.info(f"         - completion: {usage.get('completion_tokens', 0)}")
                    logger.info(f"         - total: {usage.get('total_tokens', 0)}")

                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                finish_reason = data.get("choices", [{}])[0].get("finish_reason", "unknown")

                logger.info(f"      ✅ Ответ получен (finish_reason: {finish_reason})")
                logger.info(f"      💡 Ответ (первые 200 символов):")
                logger.info(f"         {content[:200]}{'...' if len(content) > 200 else ''}")

                # Простая попытка извлечь вероятность из текста
                prob = 0.5
                if "probability" in content.lower():
                    for line in content.splitlines():
                        if "probability" in line.lower() and ":" in line:
                            try:
                                prob = float(line.split(":")[-1].strip())
                                logger.info(f"      🎯 Извлечена вероятность: {prob}")
                                break
                            except:
                                pass

                return prob, content
        except Exception as e:
            logger.error(f"   ❌ Ошибка запроса к OpenAI API: {e}")
            return None
