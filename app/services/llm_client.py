from __future__ import annotations

import json
import logging
import re
from typing import Optional, Tuple

import httpx

from ..config import settings


logger = logging.getLogger(__name__)

_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
MAX_LOG_LEN = 1000

SYSTEM_PROMPT = (
    "Ты — ведущий аналитик по проверке фактов. Отвечай ТОЛЬКО корректным JSON вида "
    '{"probability": <float 0..1>, "explanation": "подробное русское объяснение"}. '
    "probability — вероятность истинности утверж��ения (0..1). "
    "explanation — 3-4 насыщенных предложения на русском языке, где ты кратко описываешь контекст, логические аргументы, упоминаешь найденные или отсутствующие источники."
)


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
            user_prompt = (
                "Проанализируй утверждение и контекст ниже. "
                "Определи вероятность истинности (0..1) и сформулируй развёрнутое объяснение на русском языке, "
                "которое отражает основные аргументы, найденные источники или их отсутствие.\n\n"
                f"{prompt}"
            )

            request_payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 600,
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
                logger.info(f"      🧾 Полный ответ (до {MAX_LOG_LEN} символов): {content[:MAX_LOG_LEN]}")

                parsed = self._parse_llm_response(content)
                if parsed is None:
                    logger.warning(
                        "      ⚠️ Не удалось разобрать JSON LLM, используем эвристику. Сырой ответ: %s",
                        content[:MAX_LOG_LEN],
                    )
                    return 0.3, "AI-анализ вернул ответ, который не удалось разобрать."

                prob, explanation = parsed
                logger.info(f"      🎯 Извлечена вероятность: {prob}")
                return prob, explanation
        except Exception as e:
            logger.error(f"   ❌ Ошибка запроса к OpenAI API: {e}")
            return 0.3, f"AI-анализ временно недоступен: {e}"

    def _parse_llm_response(self, content: str) -> Optional[Tuple[float, str]]:
        try:
            json_candidate = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.debug("LLM JSON decode error (primary): %s | raw=%s", exc, content[:MAX_LOG_LEN])
            match = _JSON_PATTERN.search(content)
            if not match:
                logger.debug("LLM JSON regex search не нашёл подходящего блока")
                return None
            try:
                json_candidate = json.loads(match.group())
            except json.JSONDecodeError as exc:
                logger.debug("LLM JSON decode error (regex match): %s | raw=%s", exc, match.group()[:MAX_LOG_LEN])
                return None

        probability = json_candidate.get("probability")
        explanation = json_candidate.get("explanation")

        try:
            probability = float(probability)
        except (TypeError, ValueError):
            logger.debug("LLM probability не float: %s (подставляем 0.5)", probability)
            probability = 0.5

        probability = max(0.0, min(1.0, probability))
        if not explanation:
            logger.debug("LLM explanation отсутствует, используем заглушку")
            explanation = "LLM вернул пустое объяснение."

        return probability, str(explanation)
