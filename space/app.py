"""MediLens Local - lightweight Hugging Face Spaces demo.

This is a trimmed, fully self-contained version of the MediLens desktop app,
built for the Build Small Hackathon Space.

It keeps the parts that run reliably on a free CPU Space:
- Type a medicine name (generic or brand), or upload a label image.
- Optional Tesseract OCR reads text from the image.
- Fuzzy-match the text against a local 200-medicine CSV database.
- Show the likely medicine, what it is commonly used for, a safety warning,
  and the source, in 6 languages using offline template wording.

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

try:
    from rapidfuzz import fuzz
except ModuleNotFoundError:  # pragma: no cover - fallback if rapidfuzz missing
    class _FallbackFuzz:
        @staticmethod
        def ratio(left: str, right: str) -> int:
            return int(round(SequenceMatcher(None, left, right).ratio() * 100))

    fuzz = _FallbackFuzz()

# Tesseract is optional. If it is not installed, the image box still works but we
# fall back to asking the user to type the medicine name.
try:
    import pytesseract
    from PIL import Image

    _HAS_TESSERACT = True
except Exception:  # pragma: no cover
    _HAS_TESSERACT = False


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "medicines.csv"

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


def capitalize_name(name: str) -> str:
    name = (name or "").strip()
    return name[:1].upper() + name[1:] if name else name


def identify(image, typed_name, language):
    language = language or "English"
    typed_name = (typed_name or "").strip()

    ocr_text = ocr_image(image)
    query_text = typed_name if typed_name else ocr_text

    if not query_text:
        return (
            "N/A",
            LOW_CONFIDENCE_MESSAGE[language],
            GENERAL_SAFETY_WARNING[language],
            "",
            ocr_text or "(no text found - type the medicine name above)",
        )

    match = find_best_match(query_text)
    # Typed names must clear a higher bar before we trust them.
    if typed_name and match["score"] < MANUAL_MATCH_MIN_SCORE:
        match["confidence"] = "low"

    if match["row"] is None or match["confidence"] == "low":
        return (
            "N/A",
            LOW_CONFIDENCE_MESSAGE[language],
            GENERAL_SAFETY_WARNING[language],
            "",
            ocr_text or typed_name,
        )

    row = match["row"]
    medicine_match = f"{capitalize_name(match['matched_name'])} ({match['confidence']} confidence)"
    explanation = template_explanation(match, language)

    specific = str(row.get("safety_warning", "")).strip()
    safety = GENERAL_SAFETY_WARNING[language]
    if specific and language == "English":
        safety = f"{specific}\n\n{safety}"

    source = ""
    if str(row.get("source_url", "")).strip():
        source = f"{row.get('source_name', 'Source')}: {row['source_url']}"

    return medicine_match, explanation, safety, source, ocr_text or typed_name


DISCLAIMER = (
    "**Informational only.** MediLens does not give dosage instructions, does not tell you to take a "
    "medicine, and does not confirm a medicine is safe for you. It runs fully offline against a local "
    "200-medicine database. Always check with a pharmacist or doctor."
)

with gr.Blocks(title="MediLens Local") as demo:
    gr.Markdown("# 💊 MediLens Local: Medicine Label Helper")
    gr.Markdown(
        "Upload a medicine label photo **or** type a medicine name, pick a language, and MediLens "
        "explains what it is commonly used for - in plain language, fully offline."
    )
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(label="Medicine label image (optional)", sources=["upload", "webcam"], type="pil")
            name_in = gr.Textbox(label="Medicine name (generic or brand)", placeholder="e.g. paracetamol or Panadol")
            lang_in = gr.Dropdown(LANGUAGES, value="English", label="Language")
            search_btn = gr.Button("Search", variant="primary")
            if not _HAS_TESSERACT:
                gr.Markdown("_Image OCR is unavailable on this server - please type the medicine name._")
        with gr.Column():
            match_out = gr.Textbox(label="Medicine match")
            use_out = gr.Textbox(label="Medicine use", lines=4)
            safety_out = gr.Textbox(label="Safety warning", lines=4)
            source_out = gr.Textbox(label="Source")
            ocr_out = gr.Textbox(label="Extracted text (technical detail)", lines=3)

    gr.Examples(
        examples=[["paracetamol"], ["ibuprofen"], ["amoxicillin"], ["omeprazole"]],
        inputs=[name_in],
    )

    search_btn.click(
        identify,
        inputs=[image_in, name_in, lang_in],
        outputs=[match_out, use_out, safety_out, source_out, ocr_out],
    )

if __name__ == "__main__":
    demo.launch()
