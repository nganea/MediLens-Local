import re
from difflib import SequenceMatcher

import pandas as pd
try:
    from rapidfuzz import fuzz
except ModuleNotFoundError:
    class _FallbackFuzz:
        @staticmethod
        def ratio(left: str, right: str) -> int:
            return int(round(SequenceMatcher(None, left, right).ratio() * 100))

    fuzz = _FallbackFuzz()

from .config import HIGH_CONFIDENCE_MIN_SCORE, MANUAL_MATCH_MIN_SCORE, MEDIUM_CONFIDENCE_MIN_SCORE


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s;/-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def confidence_label(score: int) -> str:
    if score >= HIGH_CONFIDENCE_MIN_SCORE:
        return "high"
    if score >= MEDIUM_CONFIDENCE_MIN_SCORE:
        return "medium"
    return "low"


def text_chunks(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", clean_text(text))
    chunks = set(tokens)

    for size in range(2, 5):
        for index in range(len(tokens) - size + 1):
            chunks.add(" ".join(tokens[index : index + size]))

    return list(chunks)


def score_name_against_ocr(name: str, chunks: list[str]) -> int:
    cleaned_name = clean_text(name)
    if not cleaned_name or not chunks:
        return 0

    if cleaned_name in chunks:
        return 100

    scores = [fuzz.ratio(cleaned_name, chunk) for chunk in chunks]
    return int(round(max(scores, default=0)))


def find_best_match(ocr_text: str, medicines: pd.DataFrame) -> dict:
    chunks = text_chunks(ocr_text)
    best = {
        "score": 0,
        "matched_name": "",
        "row": None,
    }

    for _, row in medicines.iterrows():
        brand_names = [name.strip() for name in row["brand_names"].split(";") if name.strip()]
        names = brand_names + [row["generic_name"]]

        for name in names:
            score = score_name_against_ocr(name, chunks)
            if score > best["score"]:
                best = {
                    "score": score,
                    "matched_name": name,
                    "row": row,
                }

    best["confidence"] = confidence_label(best["score"])
    return best


def apply_manual_match_safety(match: dict) -> dict:
    if match["score"] < MANUAL_MATCH_MIN_SCORE:
        match["confidence"] = "low"
    return match


def capitalize_name(name: str) -> str:
    name = (name or "").strip()
    return name[:1].upper() + name[1:] if name else name
