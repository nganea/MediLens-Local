import re

import requests

from .config import TINY_AYA_MODEL_REF
from .database import apply_glossary, glossary_rules
from .matching import clean_text
from .messages import CONFIRM_MESSAGE, LOW_CONFIDENCE_MESSAGE
from .models import start_llama_server


class TinyAyaUnavailableError(RuntimeError):
    """Raised when Tiny Aya is requested but cannot provide an answer."""


def make_template_explanation(match: dict, language: str) -> str:
    if match["row"] is None or match["confidence"] == "low":
        return LOW_CONFIDENCE_MESSAGE[language]

    row = match["row"]
    matched_name = match["matched_name"]
    generic_name = row["generic_name"]
    common_uses = row["common_uses"]
    confirm = CONFIRM_MESSAGE[language]

    if language == "French":
        return (
            f"L'etiquette semble indiquer {matched_name}, qui contient {generic_name}. "
            f"Ce medicament est couramment utilise pour {common_uses}. {confirm}"
        )
    if language == "German":
        return (
            f"Das Etikett scheint {matched_name} anzugeben, das {generic_name} enthaelt. "
            f"Dieses Arzneimittel wird haeufig verwendet fuer {common_uses}. {confirm}"
        )
    if language == "Romanian":
        return (
            f"Eticheta pare sa indice {matched_name}, care contine {generic_name}. "
            f"Acest medicament este folosit in mod obisnuit pentru {common_uses}. {confirm}"
        )
    if language == "Italian":
        return (
            f"L'etichetta sembra indicare {matched_name}, che contiene {generic_name}. "
            f"Questo medicinale e comunemente usato per {common_uses}. {confirm}"
        )
    if language == "Spanish":
        return (
            f"La etiqueta parece indicar {matched_name}, que contiene {generic_name}. "
            f"Este medicamento se usa comunmente para {common_uses}. {confirm}"
        )

    return (
        f"This looks like {matched_name}, which contains {generic_name}. "
        f"It is commonly used for {common_uses}. {confirm}"
    )


def build_model_prompt(match: dict, language: str, ocr_text: str) -> str:
    row = match["row"]
    rules = glossary_rules(language)
    return f"""
You are a cautious medicine-label helper.

Write in {language}.
Use simple, plain language for a 14 to 15 year old.
Translate the medicine explanation into {language}.
Do not give medical advice.
Do not say the medicine is definitely correct.
Do not say the medicine is safe for the user.
Do not give dosage instructions.
Do not include precautions, warnings, contraindications, side effects, or leaflet advice.
Do not add a "Precautions", "Note", "Safety", or "Disclaimer" section.
Never mention tablets, capsules, creams, drops, liquids, inhalers, patches, sprays, or whether the medicine is easy to swallow.
Only explain what the likely medicine is and what it is commonly used for.
Do not repeat the same idea in different words.
Write no more than three short sentences.
The app displays safety information in a separate panel.
{rules}

Detected label text:
{ocr_text}

Likely medicine:
{match["matched_name"]}

Generic name:
{row["generic_name"]}

Common uses:
{row["common_uses"]}

Confidence:
{match["confidence"]} ({match["score"]}/100)

Write only the user-facing explanation.
""".strip()


def call_local_chat_model(model_url: str, prompt: str, max_tokens: int = 180) -> str:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    response = requests.post(model_url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def make_ai_explanation(match: dict, language: str, ocr_text: str, model_url: str) -> str:
    if match["row"] is None or match["confidence"] == "low":
        return LOW_CONFIDENCE_MESSAGE[language]

    prompt = build_model_prompt(match, language, ocr_text)

    try:
        answer = call_local_chat_model(model_url, prompt)
    except Exception as error:
        server_status = start_llama_server(TINY_AYA_MODEL_REF, model_url)
        raise TinyAyaUnavailableError(
            f"Local Tiny Aya server was not reached. Background start attempt: {server_status}. "
            f"Check the URL: {model_url}. Model error: {error}"
        ) from error

    cleaned_answer = clean_plain_language_explanation(apply_glossary(answer, language))
    return cleaned_answer or make_template_explanation(match, language)


def make_explanation(match: dict, language: str, ocr_text: str, use_ai_model: bool, model_url: str) -> str:
    if use_ai_model:
        return make_ai_explanation(match, language, ocr_text, model_url)
    return make_template_explanation(match, language)


def clean_plain_language_explanation(text: str) -> str:
    form_words = [
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "cream",
        "creams",
        "drop",
        "drops",
        "liquid",
        "liquids",
        "inhaler",
        "inhalers",
        "patch",
        "patches",
        "spray",
        "sprays",
        "easy to swallow",
    ]
    sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
    kept_sentences = []
    seen_sentences = set()
    for sentence in sentences:
        cleaned_sentence = sentence.strip()
        if not cleaned_sentence:
            continue
        sentence_lower = cleaned_sentence.lower()
        if any(word in sentence_lower for word in form_words):
            continue
        sentence_key = re.sub(r"[^a-z0-9]+", " ", sentence_lower).strip()
        if sentence_key in seen_sentences:
            continue
        seen_sentences.add(sentence_key)
        kept_sentences.append(cleaned_sentence)

    return " ".join(kept_sentences[:3]).strip()


def translate_medicine_warning(warning: str, language: str, use_ai_model: bool, model_url: str) -> str:
    if not warning:
        return ""
    if not use_ai_model:
        if language == "English":
            return polish_warning_text(warning)
        return f"Medicine-specific note shown in English: {polish_warning_text(warning)}"

    prompt = f"""
Rewrite the medicine-specific warning below into {language}.
Use simple, plain language.
Write complete sentences with normal punctuation.
Keep the meaning cautious and medically neutral.
Do not add new safety advice.
Do not add a heading or label.
Do not add bullet points.
Do not copy the source text word-for-word if it is only a fragment.
Do not mention prescriptions unless the source warning mentions prescriptions.
Do not repeat general advice about asking a pharmacist or doctor unless that exact idea is in the source warning.
Do not mention the patient information leaflet unless the source warning mentions the patient information leaflet.
Return one short paragraph only.
{glossary_rules(language)}

Source warning:
{warning}
""".strip()

    try:
        translated = call_local_chat_model(model_url, prompt, max_tokens=140)
    except Exception as error:
        server_status = start_llama_server(TINY_AYA_MODEL_REF, model_url)
        raise TinyAyaUnavailableError(
            f"Local Tiny Aya server was not reached. Background start attempt: {server_status}. "
            f"Model error: {error}"
        ) from error

    return clean_translated_warning(apply_glossary(translated, language), language) or polish_warning_text(warning)


def clean_translated_warning(text: str, language: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*\*{0,2}[^:\n]{1,40}:\*{0,2}\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return polish_warning_text(cleaned)


def polish_warning_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def english_robot_explanation(match: dict) -> str:
    if match["row"] is None or match["confidence"] == "low":
        return LOW_CONFIDENCE_MESSAGE["English"]
    row = match["row"]
    return f"It is used for {row['common_uses']}."
