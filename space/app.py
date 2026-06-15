"""MediLens Local - lightweight Hugging Face Spaces demo.

A trimmed, fully self-contained version of the MediLens desktop app, styled to
match it (light theme, branded header, fixed-size panels, confidence badge,
Improve-translation section).

It keeps the parts that run reliably on a free CPU Space:
- Type a medicine name (generic or brand), or upload a label image.
- Optional Tesseract OCR reads text from the image.
- Fuzzy-match the text against a local 200-medicine CSV database.
- Show the likely medicine, common use, a safety warning, the source, and the
  extracted text, in 6 languages using offline template wording.

It does NOT call any cloud API and does NOT need a GPU.

The full desktop app adds local vision OCR (MiniCPM-V 4.6) and local
explanation/translation (Tiny Aya Global) through llama.cpp, plus a Reachy Mini
hands-free voice assistant. Those need local model servers and are shown in the
demo video rather than hosted here.
"""

import re
from difflib import SequenceMatcher
from pathlib import Path

import gradio as gr
import pandas as pd

from _assets import BRAND_HEADER_HTML, FORCE_LIGHT_THEME_JS, CUSTOM_CSS

try:
    from rapidfuzz import fuzz
except ModuleNotFoundError:  # pragma: no cover
    class _FallbackFuzz:
        @staticmethod
        def ratio(left: str, right: str) -> int:
            return int(round(SequenceMatcher(None, left, right).ratio() * 100))

    fuzz = _FallbackFuzz()

try:
    import pytesseract
    from PIL import Image

    _HAS_TESSERACT = True
except Exception:  # pragma: no cover
    _HAS_TESSERACT = False


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "medicines.csv"
GLOSSARY_SUGGESTIONS_FILE = PROJECT_ROOT / "translation_glossary_suggestions.csv"

LANGUAGES = ["English", "French", "German", "Italian", "Romanian", "Spanish"]

MEDIUM_CONFIDENCE_MIN_SCORE = 80
HIGH_CONFIDENCE_MIN_SCORE = 90
MANUAL_MATCH_MIN_SCORE = 90


GENERAL_SAFETY_WARNING = {
    "English": (
        "Do not take this medicine unless it was prescribed for you. "
        "If you are unsure what this medicine is or whether it is safe for you, "
        "ask a pharmacist, doctor, or other qualified healthcare professional."
    ),
    "French": (
        "Ne prenez pas ce medicament sauf s'il vous a ete prescrit. "
        "Si vous n'etes pas sur de ce qu'est ce medicament ou s'il est sans danger pour vous, "
        "demandez conseil a un pharmacien, un medecin ou un autre professionnel de sante qualifie."
    ),
    "German": (
        "Nehmen Sie dieses Arzneimittel nicht ein, es sei denn, es wurde Ihnen verschrieben. "
        "Wenn Sie unsicher sind, was dieses Arzneimittel ist oder ob es fuer Sie sicher ist, "
        "fragen Sie einen Apotheker, Arzt oder eine andere qualifizierte medizinische Fachperson."
    ),
    "Italian": (
        "Non prendere questo medicinale a meno che non sia stato prescritto per te. "
        "Se non sei sicuro di che medicinale sia o se sia sicuro per te, "
        "chiedi a un farmacista, medico o altro professionista sanitario qualificato."
    ),
    "Spanish": (
        "No tome este medicamento a menos que se lo hayan recetado. "
        "Si no esta seguro de que medicamento es o si es seguro para usted, "
        "consulte a un farmaceutico, medico u otro profesional sanitario cualificado."
    ),
    "Romanian": (
        "Nu lua acest medicament decat daca ti-a fost prescris. "
        "Daca nu esti sigur ce este acest medicament sau daca este sigur pentru tine, "
        "intreaba un farmacist, un medic sau un alt profesionist medical calificat."
    ),
}

LOW_CONFIDENCE_MESSAGE = {
    "English": "I could not confidently identify the medicine. Please type the name or try a clearer photo, and ask a pharmacist if unsure.",
    "French": "Je n'ai pas pu identifier ce medicament avec confiance. Saisissez le nom ou essayez une photo plus claire, et demandez a un pharmacien en cas de doute.",
    "German": "Ich konnte das Arzneimittel nicht sicher identifizieren. Bitte geben Sie den Namen ein oder versuchen Sie ein klareres Foto und fragen Sie im Zweifel einen Apotheker.",
    "Italian": "Non sono riuscito a identificare il medicinale con sicurezza. Digita il nome o prova una foto piu chiara e chiedi a un farmacista in caso di dubbio.",
    "Spanish": "No pude identificar el medicamento con seguridad. Escriba el nombre o pruebe con una foto mas clara y consulte a un farmaceutico si tiene dudas.",
    "Romanian": "Nu am putut identifica medicamentul cu incredere. Scrie numele sau incearca o fotografie mai clara si intreaba un farmacist daca nu esti sigur.",
}

CONFIRM_MESSAGE = {
    "English": "Always confirm with a pharmacist, doctor, or the medicine leaflet.",
    "French": "Confirmez toujours avec un pharmacien, un medecin ou la notice du medicament.",
    "German": "Bestaetigen Sie dies immer mit einem Apotheker, Arzt oder der Packungsbeilage.",
    "Italian": "Conferma sempre con un farmacista, un medico o il foglio illustrativo.",
    "Spanish": "Confirme siempre con un farmaceutico, medico o el prospecto del medicamento.",
    "Romanian": "Confirma intotdeauna cu un farmacist, un medic sau prospectul medicamentului.",
}


def load_medicines() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE).fillna("")


MEDICINES = load_medicines()


def load_glossary_suggestions() -> pd.DataFrame:
    columns = ["language", "bad_phrase", "preferred_phrase"]
    if not GLOSSARY_SUGGESTIONS_FILE.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(GLOSSARY_SUGGESTIONS_FILE).fillna("")


def submit_glossary_suggestion(language, bad_phrase, preferred_phrase):
    bad_phrase = (bad_phrase or "").strip()
    preferred_phrase = (preferred_phrase or "").strip()
    if not bad_phrase or not preferred_phrase:
        return load_glossary_suggestions(), bad_phrase, preferred_phrase
    suggestions = load_glossary_suggestions()
    new_row = pd.DataFrame([{
        "language": language,
        "bad_phrase": bad_phrase,
        "preferred_phrase": preferred_phrase,
    }])
    suggestions = pd.concat([new_row, suggestions], ignore_index=True)
    try:
        suggestions.to_csv(GLOSSARY_SUGGESTIONS_FILE, index=False, encoding="utf-8")
    except Exception:
        pass  # Space storage is ephemeral; keep the in-memory table either way.
    return suggestions, "", ""


def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s;/-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def confidence_label(score: int) -> str:
    if score >= HIGH_CONFIDENCE_MIN_SCORE:
        return "high"
    if score >= MEDIUM_CONFIDENCE_MIN_SCORE:
        return "medium"
    return "low"


def capitalize_name(name: str) -> str:
    name = (name or "").strip()
    return name[:1].upper() + name[1:] if name else name


def text_chunks(text: str) -> list:
    tokens = re.findall(r"[a-z0-9]+", clean_text(text))
    chunks = set(tokens)
    for size in range(2, 5):
        for index in range(len(tokens) - size + 1):
            chunks.add(" ".join(tokens[index : index + size]))
    return list(chunks)


def score_name_against_text(name: str, chunks: list) -> int:
    cleaned_name = clean_text(name)
    if not cleaned_name or not chunks:
        return 0
    if cleaned_name in chunks:
        return 100
    scores = [fuzz.ratio(cleaned_name, chunk) for chunk in chunks]
    return int(round(max(scores, default=0)))


def find_best_match(query_text: str) -> dict:
    chunks = text_chunks(query_text)
    best = {"score": 0, "matched_name": "", "row": None}
    for _, row in MEDICINES.iterrows():
        brand_names = [n.strip() for n in str(row["brand_names"]).split(";") if n.strip()]
        names = brand_names + [row["generic_name"]]
        for name in names:
            score = score_name_against_text(name, chunks)
            if score > best["score"]:
                best = {"score": score, "matched_name": name, "row": row}
    best["confidence"] = confidence_label(best["score"])
    return best


MATCH_NA_HTML = '<p class="match-na">N/A</p>'


def likely_match_html(match: dict) -> str:
    row = match["row"]
    if row is None or match["confidence"] == "low":
        return MATCH_NA_HTML
    generic_name = row["generic_name"]
    matched_name = match["matched_name"]
    confidence = match["confidence"]
    score = match["score"]
    if clean_text(matched_name) == clean_text(generic_name):
        display_name = capitalize_name(generic_name)
    else:
        display_name = f"{capitalize_name(matched_name)} / {capitalize_name(generic_name)}"
    badge_class = f"confidence-{confidence}"
    badge_label = f"{confidence.capitalize()} confidence &middot; {score}/100"
    return (
        f'<div class="match-card">'
        f'<p class="match-medicine-name">{display_name}</p>'
        f'<span class="confidence-badge {badge_class}">{badge_label}</span>'
        f"</div>"
    )


def source_url_html(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    return (
        f'<div class="source-url-box">'
        f'<span class="source-url-label">Source</span>'
        f'<a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>'
        f"</div>"
    )


def template_explanation(match: dict, language: str) -> str:
    if match["row"] is None or match["confidence"] == "low":
        return LOW_CONFIDENCE_MESSAGE[language]
    row = match["row"]
    matched_name = match["matched_name"]
    generic_name = row["generic_name"]
    common_uses = row["common_uses"]
    confirm = CONFIRM_MESSAGE[language]
    if language == "French":
        return (f"L'etiquette semble indiquer {matched_name}, qui contient {generic_name}. "
                f"Ce medicament est couramment utilise pour {common_uses}. {confirm}")
    if language == "German":
        return (f"Das Etikett scheint {matched_name} anzugeben, das {generic_name} enthaelt. "
                f"Dieses Arzneimittel wird haeufig verwendet fuer {common_uses}. {confirm}")
    if language == "Romanian":
        return (f"Eticheta pare sa indice {matched_name}, care contine {generic_name}. "
                f"Acest medicament este folosit in mod obisnuit pentru {common_uses}. {confirm}")
    if language == "Italian":
        return (f"L'etichetta sembra indicare {matched_name}, che contiene {generic_name}. "
                f"Questo medicinale e comunemente usato per {common_uses}. {confirm}")
    if language == "Spanish":
        return (f"La etiqueta parece indicar {matched_name}, que contiene {generic_name}. "
                f"Este medicamento se usa comunmente para {common_uses}. {confirm}")
    return (f"This looks like {matched_name}, which contains {generic_name}. "
            f"It is commonly used for {common_uses}. {confirm}")


def ocr_image(image) -> str:
    if image is None or not _HAS_TESSERACT:
        return ""
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return pytesseract.image_to_string(image)
    except Exception:
        return ""


def identify(image, typed_name, language):
    language = language or "English"
    typed_name = (typed_name or "").strip()

    ocr_text = ocr_image(image)
    query_text = typed_name if typed_name else ocr_text

    if not query_text:
        return (
            MATCH_NA_HTML,
            LOW_CONFIDENCE_MESSAGE[language],
            GENERAL_SAFETY_WARNING[language],
            "",
            ocr_text or "(no text found - type the medicine name above)",
        )

    match = find_best_match(query_text)
    if typed_name and match["score"] < MANUAL_MATCH_MIN_SCORE:
        match["confidence"] = "low"

    if match["row"] is None or match["confidence"] == "low":
        return (
            MATCH_NA_HTML,
            LOW_CONFIDENCE_MESSAGE[language],
            GENERAL_SAFETY_WARNING[language],
            "",
            ocr_text or typed_name,
        )

    row = match["row"]
    explanation = template_explanation(match, language)
    specific = str(row.get("safety_warning", "")).strip()
    safety = GENERAL_SAFETY_WARNING[language]
    if specific and language == "English":
        safety = f"{specific}\n\n{safety}"
    source = source_url_html(str(row.get("source_url", "")))
    return likely_match_html(match), explanation, safety, source, ocr_text or typed_name


INTRO_HTML = (
    '<p>Type a medicine name or upload a label photo, choose a language, and MediLens '
    'explains what the medicine is commonly used for - fully offline, against a local '
    '200-medicine database. <strong>Informational only:</strong> it gives no dosage advice '
    'and does not replace a pharmacist or doctor.</p>'
)

MANUAL_HELPER_HTML = '<p>Tip: typing the name (generic or brand) gives the most reliable match.</p>'

MODEL_NOTE = (
    "This hosted Space is the **lightweight CPU demo**: name lookup, Tesseract OCR, and "
    "offline multilingual explanations from a local database. The full desktop app adds "
    "**MiniCPM-V 4.6** (OpenBMB) vision OCR and **Tiny Aya Global** (Cohere) translation via "
    "llama.cpp - shown in the demo video."
)

REACHY_HTML = (
    '<div class="source-url-box">'
    '<p><strong>Reachy Mini voice assistant (in the demo video).</strong> '
    'The full project also runs hands-free on a Reachy Mini robot: say its name plus a '
    'medicine word, it captures the label with its camera, identifies the medicine, and '
    'speaks the answer in your language. Speech is fully offline - faster-whisper (OpenAI '
    'Whisper) for listening, Kokoro and Piper voices for speaking. This is shown in the demo '
    'video; the hosted Space runs the lightweight web version above.</p>'
    '</div>'
)

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    radius_size=gr.themes.sizes.radius_lg,
)

with gr.Blocks(title="MediLens Local") as demo:
    gr.HTML(BRAND_HEADER_HTML, elem_id="brand-header-block")
    gr.HTML(INTRO_HTML, elem_id="intro-text")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_id="top-left-panel"):
            image_in = gr.Image(
                label="Medicine label image (optional)",
                sources=["upload", "webcam"],
                type="pil",
                elem_id="medicine-image",
            )
            with gr.Group(elem_id="manual-label-section"):
                name_in = gr.Textbox(
                    label="Medicine name (generic or brand)",
                    placeholder="e.g. paracetamol or Panadol",
                    elem_id="manual-label-input",
                )
                gr.HTML(MANUAL_HELPER_HTML, elem_id="manual-label-helper")
            lang_in = gr.Dropdown(LANGUAGES, value="English", label="Language", elem_id="language-selector")
            with gr.Column(elem_id="read-action-section"):
                search_btn = gr.Button("Search", variant="primary", elem_id="read-btn")
                if not _HAS_TESSERACT:
                    gr.HTML(
                        '<p class="match-na">Image OCR is unavailable - please type the medicine name.</p>',
                        elem_id="processing-progress",
                    )

        with gr.Column(scale=1, elem_id="top-right-panel"):
            match_output = gr.HTML(MATCH_NA_HTML, elem_id="match-output")
            explanation_output = gr.Textbox(label="Medicine use", lines=5, elem_id="explanation-output")
            warning_output = gr.Textbox(label="Safety warning", lines=5, elem_id="warning-output")
            source_url_output = gr.HTML("", elem_id="source-url-output")

    gr.Markdown(MODEL_NOTE, elem_id="model-note")

    gr.Examples(examples=[["paracetamol"], ["ibuprofen"], ["amoxicillin"], ["omeprazole"]], inputs=[name_in])

    with gr.Accordion("Improve translation", open=False):
        gr.HTML(
            '<p>Suggest better wording for a phrase. Suggestions are collected for human '
            'review and are not applied automatically.</p>',
            elem_id="glossary-note",
        )
        with gr.Row(equal_height=True):
            glossary_language_input = gr.Dropdown(choices=LANGUAGES, value="Romanian", label="Language")
            bad_phrase_input = gr.Textbox(label="Phrase to improve")
            preferred_phrase_input = gr.Textbox(label="Preferred phrase")
        submit_glossary_button = gr.Button("Submit suggestion")
        glossary_suggestions_table = gr.Dataframe(
            value=load_glossary_suggestions(),
            headers=["language", "bad_phrase", "preferred_phrase"],
            datatype=["str", "str", "str"],
            label="Submitted suggestions (this session)",
            interactive=False,
        )

    with gr.Accordion("Technical details", open=False):
        ocr_output = gr.Textbox(label="Extracted OCR text", lines=4, elem_id="ocr-output")
        gr.HTML(REACHY_HTML, elem_id="reachy-status-output")

    search_btn.click(
        identify,
        inputs=[image_in, name_in, lang_in],
        outputs=[match_output, explanation_output, warning_output, source_url_output, ocr_output],
    )
    submit_glossary_button.click(
        submit_glossary_suggestion,
        inputs=[glossary_language_input, bad_phrase_input, preferred_phrase_input],
        outputs=[glossary_suggestions_table, bad_phrase_input, preferred_phrase_input],
    )

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS, theme=THEME, js=FORCE_LIGHT_THEME_JS)
