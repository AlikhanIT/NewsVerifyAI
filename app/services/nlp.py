from __future__ import annotations
from typing import List, Dict
import re

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Простейший NER: если есть spaCy — используем, иначе грубый fallback.
    """
    entities: Dict[str, List[str]] = {"PERSON": [], "ORG": [], "GPE": [], "DATE": []}

    if _NLP:
        doc = _NLP(text)
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
    else:
        # грубый fallback: ищем капс/похожие штуки
        capitalized = re.findall(r"\b[A-Z][a-zA-Z]+\b", text)
        entities["ORG"] = capitalized

    # убираем дубликаты
    for k in entities:
        entities[k] = list(dict.fromkeys(entities[k]))
    return entities


def extract_ngrams(text: str, n: int = 2) -> List[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
