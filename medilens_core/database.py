import re

import pandas as pd

from .config import DATA_FILE, GLOSSARY_FILE, GLOSSARY_SUGGESTIONS_FILE


def load_medicines() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE).fillna("")


def load_glossary() -> pd.DataFrame:
    if not GLOSSARY_FILE.exists():
        return pd.DataFrame(columns=["language", "bad_phrase", "preferred_phrase"])
    return pd.read_csv(GLOSSARY_FILE).fillna("")


def load_glossary_suggestions() -> pd.DataFrame:
    columns = ["language", "bad_phrase", "preferred_phrase"]
    if not GLOSSARY_SUGGESTIONS_FILE.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(GLOSSARY_SUGGESTIONS_FILE).fillna("")


def submit_glossary_suggestion(language: str, bad_phrase: str, preferred_phrase: str):
    bad_phrase = (bad_phrase or "").strip()
    preferred_phrase = (preferred_phrase or "").strip()

    if not bad_phrase or not preferred_phrase:
        return load_glossary_suggestions(), bad_phrase, preferred_phrase

    suggestions = load_glossary_suggestions()
    new_row = pd.DataFrame(
        [{"language": language, "bad_phrase": bad_phrase, "preferred_phrase": preferred_phrase}]
    )
    suggestions = pd.concat([new_row, suggestions], ignore_index=True)
    suggestions.to_csv(GLOSSARY_SUGGESTIONS_FILE, index=False, encoding="utf-8")
    return suggestions, "", ""


def glossary_rules(language: str) -> str:
    glossary = load_glossary()
    if glossary.empty:
        return ""

    rows = glossary[glossary["language"].str.lower() == language.lower()]
    rules = []
    for _, row in rows.iterrows():
        bad_phrase = str(row["bad_phrase"]).strip()
        preferred_phrase = str(row["preferred_phrase"]).strip()
        if bad_phrase and preferred_phrase:
            rules.append(f'- Do not say "{bad_phrase}". Say "{preferred_phrase}" instead.')

    if not rules:
        return ""

    return "Wording rules:\n" + "\n".join(rules)


def apply_glossary(text: str, language: str) -> str:
    glossary = load_glossary()
    if glossary.empty or not text:
        return text

    rows = glossary[glossary["language"].str.lower() == language.lower()]
    cleaned_text = text
    for _, row in rows.iterrows():
        bad_phrase = str(row["bad_phrase"]).strip()
        preferred_phrase = str(row["preferred_phrase"]).strip()
        if bad_phrase and preferred_phrase:
            cleaned_text = re.sub(re.escape(bad_phrase), preferred_phrase, cleaned_text, flags=re.IGNORECASE)

    return cleaned_text

