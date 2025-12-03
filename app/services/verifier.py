from __future__ import annotations

import hashlib
import json
import logging
from typing import Tuple

from sqlalchemy.orm import Session

from ..schemas import CheckRequest, CheckResponse, ClaimAnalysis
from ..models import CachedResult
from .nlp import extract_entities, extract_ngrams
from .news_client import NewsClient
from .llm_client import LLMClient

# Configure logger
logger = logging.getLogger(__name__)


def _hash_claim(text: str, style: str) -> str:
    h = hashlib.sha256()
    h.update(text.strip().encode("utf-8"))
    h.update(style.encode("utf-8"))
    return h.hexdigest()


def _format_formal_explanation(
    claim: str,
    status: str,
    probability: float | None,
    entities: dict,
    news_count: int,
    base_explanation: str,
) -> str:
    prob_text = f"{probability:.2f}" if probability is not None else "не определена"
    ent_parts = []
    for k, vals in entities.items():
        if vals:
            ent_parts.append(f"{k}: {', '.join(vals)}")
    entities_text = "; ".join(ent_parts) if ent_parts else "ключевые сущности не выделены"

    status_human = {
        "confirmed": "подтверждено новостными источниками",
        "not_found": "подтверждений в новостях не найдено (низкая вероятность события)",
        "uncertain": "прямых подтверждений нет (событие умеренно правдоподобно)",
    }.get(status, status)

    return (
        f"Проведена автоматическая проверка утверждения:\n\n"
        f"«{claim}».\n\n"
        f"На основе анализа новостных публикаций и извлечённых сущностей "
        f"система сформировала следующий вывод: {status_human}. "
        f"Оценочная вероятность истинности события: {prob_text}.\n\n"
        f"Выделенные сущности: {entities_text}. "
        f"Количество релевантных публикаций за последнюю неделю: {news_count}. "
        f"Дополнительный анализ (AI):\n\n{base_explanation}\n\n"
        f"Вывод носит вероятностный характер и не является юридически значимым."
    )


def _format_simple_explanation(status: str, probability: float | None) -> str:
    if probability is None:
        return "Нет данных для оценки."

    if status == "confirmed":
        base = "Нашли подтверждение в новостях."
    elif status == "not_found":
        base = "В новостях не нашли подтверждений. Событие маловероятно."
    else:
        base = "Прямых подтверждений нет."

    if probability >= 0.8:
        prob_text = "Вероятно, это правда."
    elif probability >= 0.5:
        prob_text = "Похоже на правду, но не точно."
    elif probability >= 0.3:
        prob_text = "Сомнительно."
    else:
        prob_text = "Маловероятно."

    return f"{base} {prob_text}".strip()


async def verify_claim(
    db: Session,
    payload: CheckRequest,
    news_client: NewsClient | None = None,
    llm_client: LLMClient | None = None,
) -> CheckResponse:
    news_client = news_client or NewsClient()
    llm_client = llm_client or LLMClient()

    logger.info("🔍 НАЧАЛО ПРОВЕРКИ УТВЕРЖДЕНИЯ")
    
    claim = payload.text.strip()
    style = payload.style

    print(f"\n=== 🚀 Verifying claim: {claim} ===")

    claim_hash = _hash_claim(claim, style)
    cached = db.query(CachedResult).filter_by(claim_hash=claim_hash).first()
    if cached:
        print("⚡ Loaded from CACHE")
        data = json.loads(cached.result_json)
        return CheckResponse(**data, cached=True)

    print("🔍 Cache not found, running fresh analysis...")

    entities = extract_entities(claim)
    ngrams = extract_ngrams(claim, n=2)
    print(f"🔎 Entities extracted: {entities}")
    print(f"🔗 N-grams: {ngrams}")

    news_results = await news_client.search(claim, from_days=7, limit=5)

    print("\n=== 📡 NEWS RESULTS DEBUG ===")
    if news_results:
        print(f"📰 Found {len(news_results)} result(s):")
        for nr in news_results:
            print(f"   • {nr}")
    else:
        print("🗞 No news results found.")

    if news_results:
        status = "confirmed"
        probability = 0.9
        explanation_base = (
            f"По найденным новостным публикациям утверждение вероятно подтверждается. "
            f"Найдено совпадений: {len(news_results)}."
        )
    else:
        context_summary = "No matching news articles found for this claim within the last 7 days."
        try:
            probability, llm_explanation = await llm_client.analyze(
                f"Claim: {claim}\nContext: {context_summary}"
            )

            print("\n=== 🤖 LLM ANALYSIS DEBUG ===")
            print(f"💬 Claim: {claim}")
            print(f"📊 Probability: {probability}")
            print(f"📝 Explanation:\n{llm_explanation}")

        except Exception as e:
            probability = 0.3
            llm_explanation = f"AI-анализ временно недоступен: {e}"
            print(f"\n=== ❌ LLM ERROR ===\n{e}")

        status = "uncertain" if probability >= 0.4 else "not_found"
        explanation_base = llm_explanation

    print(f"\n📌 Final status: {status} (prob={probability:.2f})")

    explanation = (
        _format_formal_explanation(claim, status, probability, entities, len(news_results), explanation_base)
        if style == "formal"
        else _format_simple_explanation(status, probability)
    )

    analysis = ClaimAnalysis(
        status=status,
        probability=probability,
        explanation=explanation,
        matched_sources=news_results,
    )

    resp_obj = CheckResponse(
        claim=claim,
        style=style,
        analysis=analysis,
        cached=False,
    )

    print("\n💾 Saving result to cache...")
    db.add(CachedResult(claim_hash=claim_hash, result_json=resp_obj.json()))
    db.commit()

    print("✅ Done.\n")
    return resp_obj
