from __future__ import annotations

import hashlib
import json
import asyncio
import logging
import re

from sqlalchemy.orm import Session

from ..schemas import CheckRequest, CheckResponse, ClaimAnalysis
from ..models import CachedResult
from .nlp import extract_entities, extract_ngrams
from .news_client import NewsClient
from .llm_client import LLMClient


logger = logging.getLogger(__name__)

STOP_WORDS = {
    "на", "и", "в", "во", "к", "ко", "по", "из", "за", "от", "для", "как", "что",
    "это", "этот", "эта", "эти", "последний", "месяц", "месяца", "год", "года",
}
WORD_PATTERN = re.compile(r"[\w%]+", re.UNICODE)


def _hash_claim(text: str, style: str) -> str:
    h = hashlib.sha256()
    h.update(text.strip().encode("utf-8"))
    h.update(style.encode("utf-8"))
    return h.hexdigest()


def _is_cache_entry_modern(data: dict) -> bool:
    analysis = data.get("analysis")
    if not isinstance(analysis, dict):
        return False

    matched_sources = analysis.get("matched_sources")
    if matched_sources is None:
        return True
    if not isinstance(matched_sources, list):
        return False

    for item in matched_sources:
        if not isinstance(item, dict):
            return False
        if not item.get("title"):
            return False
        # url может быть пустым у старых записей, но структура должна присутствовать
        if "url" not in item:
            return False
    return True


def _normalize_cached_result(data: dict) -> dict:
    analysis = data.get("analysis")
    if not isinstance(analysis, dict):
        return data

    matched_sources = analysis.get("matched_sources")
    if not isinstance(matched_sources, list):
        return data

    normalized = []
    changed = False
    for item in matched_sources:
        if isinstance(item, dict):
            normalized.append(item)
        else:
            changed = True
            text = str(item)
            normalized.append(
                {
                    "title": text,
                    "description": None,
                    "summary": text,
                    "url": None,
                    "published_at": None,
                    "source_name": None,
                }
            )

    if changed:
        analysis["matched_sources"] = normalized

    return data


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


def _collect_keywords(claim: str, entities: dict[str, list[str]]) -> list[str]:
    tokens: set[str] = set()
    for token in WORD_PATTERN.findall(claim.lower()):
        if len(token) < 3 or token in STOP_WORDS:
            continue
        tokens.add(token)

    for values in entities.values():
        for value in values:
            for token in WORD_PATTERN.findall(value.lower()):
                if len(token) < 3 or token in STOP_WORDS:
                    continue
                tokens.add(token)

    return list(tokens)


def _filter_news_by_keywords(news_list, keywords):
    if not keywords:
        return news_list

    min_hits = 1 if len(keywords) <= 3 else 2
    filtered = []
    for item in news_list:
        text = " ".join(filter(None, [item.title, item.description, item.summary, item.source_name])).lower()
        hits = sum(1 for kw in keywords if kw in text)
        if hits >= min_hits:
            filtered.append(item)

    return filtered


async def verify_claim(
    db: Session,
    payload: CheckRequest,
    news_client: NewsClient | None = None,
    llm_client: LLMClient | None = None,
) -> CheckResponse:
    news_client = news_client or NewsClient()
    llm_client = llm_client or LLMClient()

    claim = payload.text.strip()
    style = payload.style

    logger.info("🔍 НАЧАЛО ПРОВЕРКИ УТВЕРЖДЕНИЯ")
    logger.info(f"   📝 Утверждение: {claim[:100]}{'...' if len(claim) > 100 else ''}")
    logger.info(f"   🎨 Стиль: {style}")

    claim_hash = _hash_claim(claim, style)
    logger.info(f"   🔐 Hash: {claim_hash}")

    cached = db.query(CachedResult).filter_by(claim_hash=claim_hash).first()
    if cached:
        logger.info("   ⚡ НАЙДЕН В КЭШЕ - возвращаем сохраненный результат")
        data = json.loads(cached.result_json)
        if _is_cache_entry_modern(data):
            data['cached'] = True
            return CheckResponse(**data)

        logger.info("   ♻️ Legacy кэш без структурированных ссылок — обновляем результат")
        db.delete(cached)
        db.commit()

    logger.info("   🆕 НЕТ В КЭШЕ - выполняем полную проверку")

    # Извлечение сущностей и n-грамм
    logger.info("   🔎 Извлекаем сущности и n-граммы...")
    entities = extract_entities(claim)
    ngrams = extract_ngrams(claim, n=2)
    logger.info(f"      📌 Сущности: {entities}")
    logger.info(f"      🔗 N-граммы: {ngrams[:5]}{'...' if len(ngrams) > 5 else ''}")

    # Поиск новостей
    logger.info("   📰 Ищем новости...")
    keyword_candidates = _collect_keywords(claim, entities)
    logger.info(f"      🔍 Ключевые слова: {keyword_candidates}")
    news_results = await news_client.search(claim, from_days=7, limit=8)
    filtered_news = _filter_news_by_keywords(news_results, keyword_candidates)

    if filtered_news:
        logger.info(f"   ✅ Найдено {len(filtered_news)} релевантных статей из {len(news_results)}")
        for idx, nr in enumerate(filtered_news, 1):
            logger.info(
                "      %s. [%s] %s",
                idx,
                (nr.source_name or "Unknown"),
                (nr.title or "Без названия")[:100],
            )
        status = "confirmed"
        probability = 0.9
        explanation_base = (
            f"По найденным новостным публикациям утверждении вероятно подтверждается. "
            f"Найдено совпадений: {len(filtered_news)}."
        )
        news_payload = filtered_news[:5]
    else:
        logger.info("   ❌ Релевантных новостей не найдено или совпадений недостаточно")
        context_summary = "No matching news articles found for this claim within the last 7 days."

        try:
            timeout_seconds = 10
            logger.info(f"   ⏱️ Запускаем LLM анализ (таймаут: {timeout_seconds}s)...")
            result = await asyncio.wait_for(
                llm_client.analyze(f"Claim: {claim}\nContext: {context_summary}"),
                timeout=timeout_seconds,
            )

            if not result or not isinstance(result, (list, tuple)):
                logger.warning("   ⚠️ LLM вернул пустой ответ, используем эвристику")
                probability, llm_explanation = 0.3, "AI-анализ не предоставил результатов."
            else:
                probability, llm_explanation = result

            if not llm_explanation:
                llm_explanation = "AI-анализ вернул пустое объяснение."

            logger.info(f"   🎯 LLM вероятность: {probability}")
            logger.info(f"   💬 LLM объяснение (первые 150 символов): {llm_explanation[:150]}...")

        except asyncio.TimeoutError:
            logger.warning(f"   ⏱️ LLM ТАЙМАУТ (>{timeout_seconds}s)")
            probability = 0.3
            llm_explanation = "AI-анализ не завершился за отведённое время (таймаут). Использована эвристическая оценка."
        except Exception as e:
            logger.error(f"   ❌ LLM ОШИБКА: {e}")
            probability = 0.3
            llm_explanation = f"AI-анализ временно недоступен: {e}"

        if not llm_explanation:
            llm_explanation = "AI-анализ недоступен или не дал результата. Использована эвристическая оценка."

        status = "uncertain" if (probability is not None and probability >= 0.4) else "not_found"
        explanation_base = llm_explanation
        news_payload = []

    logger.info(f"   📊 ИТОГОВЫЙ СТАТУС: {status} (вероятность={probability:.2f})")

    logger.info(f"   📝 Форматируем объяснение (стиль: {style})...")
    explanation = (
        _format_formal_explanation(claim, status, probability, entities, len(news_payload), explanation_base)
        if style == "formal"
        else _format_simple_explanation(status, probability)
    )

    analysis = ClaimAnalysis(
        status=status,
        probability=probability,
        explanation=explanation,
        matched_sources=news_payload,
    )

    resp_obj = CheckResponse(
        claim=claim,
        style=style,
        analysis=analysis,
        cached=False,
    )

    logger.info("   💾 Сохраняем результат в кэш БД...")
    db.add(CachedResult(claim_hash=claim_hash, result_json=resp_obj.json()))
    db.commit()
    logger.info("   ✅ ПРОВЕРКА ЗАВЕРШЕНА УСПЕШНО")

    return resp_obj
