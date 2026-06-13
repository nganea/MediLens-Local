from pathlib import Path
import base64
import concurrent.futures
from io import BytesIO
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from urllib.parse import urlparse

import gradio as gr
import pandas as pd
import pytesseract
import requests
from pytesseract import TesseractNotFoundError
from PIL import Image, ImageFilter, ImageOps
from rapidfuzz import fuzz


DATA_FILE = Path(__file__).with_name("medicines.csv")
GLOSSARY_FILE = Path(__file__).with_name("translation_glossary.csv")
GLOSSARY_SUGGESTIONS_FILE = Path(__file__).with_name("translation_glossary_suggestions.csv")
WINDOWS_TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
DEFAULT_MODEL_URL = os.getenv("TINY_AYA_MODEL_URL", "http://127.0.0.1:8080/v1/chat/completions")
DEFAULT_VISION_OCR_URL = os.getenv("MINICPM_V_OCR_URL", "http://127.0.0.1:8081/v1/chat/completions")
TINY_AYA_MODEL_REF = os.getenv("TINY_AYA_MODEL_REF", "CohereLabs/tiny-aya-global-GGUF:Q4_K_M")
MINICPM_V_MODEL_REF = os.getenv("MINICPM_V_MODEL_REF", "openbmb/MiniCPM-V-4.6-gguf:Q4_K_M")
APP_SERVER_NAME = os.getenv("MEDILENS_SERVER_NAME")
APP_SERVER_PORT = int(os.getenv("MEDILENS_SERVER_PORT", "7860"))
APP_SHARE = os.getenv("MEDILENS_SHARE", "").strip().lower() in {"1", "true", "yes", "on"}
MEDIUM_CONFIDENCE_MIN_SCORE = 80
HIGH_CONFIDENCE_MIN_SCORE = 90
MANUAL_MATCH_MIN_SCORE = 90
OCR_MATCH_MIN_SCORE = 90
DEFAULT_IMAGE_IDENTIFICATION_SECONDS = 60
MIN_IMAGE_IDENTIFICATION_SECONDS = 10
MAX_IMAGE_IDENTIFICATION_SECONDS = 99
RESPONSE_VERSION = 0
RESPONSE_VERSION_LOCK = threading.Lock()

if shutil.which("tesseract") is None and WINDOWS_TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(WINDOWS_TESSERACT_PATH)


def next_response_version() -> int:
    global RESPONSE_VERSION
    with RESPONSE_VERSION_LOCK:
        RESPONSE_VERSION += 1
        return RESPONSE_VERSION


def current_response_version() -> int:
    with RESPONSE_VERSION_LOCK:
        return RESPONSE_VERSION


def stale_read_response():
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def hide_progress_bar():
    return gr.update(value="", visible=True)


def progress_bar_html(elapsed_seconds: float, total_seconds: int, status: str):
    total_seconds = max(1, int(total_seconds))
    elapsed_seconds = max(0, min(total_seconds, int(round(elapsed_seconds))))
    return gr.update(
        value=f'<div class="single-progress-status">{status} {elapsed_seconds}/{total_seconds}s</div>',
        visible=True,
    )


def normalize_local_url(url: str, default_url: str) -> str:
    cleaned_url = (url or "").strip() or default_url
    if "://" not in cleaned_url:
        cleaned_url = f"http://{cleaned_url}"
    return cleaned_url


def parse_host_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def is_port_reachable(url: str, timeout: float = 1.0) -> bool:
    try:
        host, port = parse_host_port(url)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def start_llama_server(model_ref: str, url: str) -> str:
    if is_port_reachable(url):
        host, port = parse_host_port(url)
        return f"Already running on {host}:{port}."

    llama_server = shutil.which("llama-server")
    if not llama_server:
        return "Could not find llama-server on PATH. Open a new terminal after installing llama.cpp, or start llama-server manually."

    _, port = parse_host_port(url)
    command = [llama_server, "-hf", model_ref, "--port", str(port)]
    kwargs = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    try:
        subprocess.Popen(command, **kwargs)
    except OSError as error:
        return f"Could not start llama-server: {error}"

    return f"Starting in the background on port {port}. Wait a minute for the model to load, then try again."


def start_local_model_servers(vision_ocr_url: str, model_url: str) -> str:
    vision_ocr_url = normalize_local_url(vision_ocr_url, DEFAULT_VISION_OCR_URL)
    model_url = normalize_local_url(model_url, DEFAULT_MODEL_URL)
    minicpm_status = start_llama_server(MINICPM_V_MODEL_REF, vision_ocr_url)
    tiny_aya_status = start_llama_server(TINY_AYA_MODEL_REF, model_url)
    return f"MiniCPM-V OCR: {minicpm_status}\nTiny Aya: {tiny_aya_status}"


def check_local_model_servers(vision_ocr_url: str, model_url: str) -> str:
    vision_ocr_url = normalize_local_url(vision_ocr_url, DEFAULT_VISION_OCR_URL)
    model_url = normalize_local_url(model_url, DEFAULT_MODEL_URL)
    minicpm_status = "reachable" if is_port_reachable(vision_ocr_url) else "not reachable"
    tiny_aya_status = "reachable" if is_port_reachable(model_url) else "not reachable"
    return f"MiniCPM-V OCR: {minicpm_status}\nTiny Aya: {tiny_aya_status}"


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
    "English": "I could not confidently identify the medicine. Please try a clearer photo or ask a pharmacist.",
    "French": "Je n'ai pas pu identifier ce medicament avec suffisamment de confiance. Essayez une photo plus claire ou demandez conseil a un pharmacien.",
    "German": "Ich konnte das Arzneimittel nicht sicher identifizieren. Bitte versuchen Sie ein klareres Foto oder fragen Sie einen Apotheker.",
    "Italian": "Non sono riuscito a identificare il medicinale con sicurezza. Prova una foto piu chiara o chiedi a un farmacista.",
    "Spanish": "No pude identificar el medicamento con suficiente confianza. Pruebe con una foto mas clara o consulte a un farmaceutico.",
    "Romanian": "Nu am putut identifica medicamentul cu incredere. Incearca o fotografie mai clara sau intreaba un farmacist.",
}

CONFIRM_MESSAGE = {
    "English": "Please confirm this with a pharmacist if unsure.",
    "French": "Veuillez confirmer cela avec un pharmacien si vous avez un doute.",
    "German": "Bitte bestaetigen Sie dies bei einem Apotheker, wenn Sie unsicher sind.",
    "Italian": "Conferma con un farmacista se non sei sicuro.",
    "Spanish": "Confirme esto con un farmaceutico si no esta seguro.",
    "Romanian": "Te rugam sa confirmi cu un farmacist daca nu esti sigur.",
}

UI_LABELS = {
    "English": {
        "intro": "Upload a photo of a medicine label. This tool tries to read the label and explain what the medicine is commonly used for. It does not replace a pharmacist or doctor.",
        "image_label": "Medicine label photo",
        "image_help": "Upload an image, or use the camera option to capture a new photo.",
        "manual_label": "Type medicine label",
        "manual_placeholder": "Optional: type the medicine name or label text",
        "manual_help": "Type the medicine name or label text here, then click Read medicine label again.",
        "read_button": "Read medicine label",
        "match": "Medicine match",
        "explanation": "Medicine use",
        "warning": "Safety warning",
        "source_url": "Source URL",
        "model_note": "This app uses two small local AI models: MiniCPM to read text from images and Tiny Aya Global to generate text. It uses a curated database of 200 commonly used medicines in the UK and USA, with information taken from NHS Medicines A to Z and British National Formulary Drugs A to Z. These are publicly available medicine databases where you can find more information. This app was developed using OpenAI Codex.",
        "glossary_title": "Improve translation",
        "glossary_note": "Tiny Aya is useful for translation, but it is not perfect. You can suggest better wording for the translation glossary. Suggestions are public and must be reviewed before they are added. Thank you for helping improve the app.",
        "glossary_language": "Language",
        "bad_phrase": "Bad phrase",
        "preferred_phrase": "Preferred phrase",
        "glossary_submit": "Add glossary suggestion",
        "glossary_table": "Public glossary suggestions awaiting review",
    },
    "French": {
        "intro": "Importez une photo d'une etiquette de medicament. Cet outil essaie de lire l'etiquette et d'expliquer a quoi le medicament sert habituellement. Il ne remplace pas un pharmacien ou un medecin.",
        "image_label": "Photo de l'etiquette du medicament",
        "image_help": "Importez une image ou utilisez l'option appareil photo pour prendre une nouvelle photo.",
        "manual_label": "Saisir le medicament",
        "manual_placeholder": "Facultatif : saisissez le nom du medicament ou le texte de l'etiquette",
        "manual_help": "Saisissez ici le nom du medicament ou le texte de l'etiquette, puis relancez la lecture.",
        "read_button": "Lire l'etiquette du medicament",
        "match": "Medicament trouve",
        "explanation": "Utilisation du medicament",
        "warning": "Avertissement de securite",
        "source_url": "URL source",
        "model_note": "Cette application utilise deux petits modeles d'IA locaux : MiniCPM pour lire le texte des images et Tiny Aya Global pour generer le texte. Elle utilise une base de donnees selectionnee de 200 medicaments couramment utilises au Royaume-Uni et aux Etats-Unis, avec des informations issues de NHS Medicines A to Z et de British National Formulary Drugs A to Z. Ce sont des bases de donnees publiques ou vous pouvez trouver plus d'informations sur les medicaments. Cette application a ete developpee avec OpenAI Codex.",
        "glossary_title": "Ameliorer la traduction",
        "glossary_note": "Tiny Aya est utile pour la traduction, mais il n'est pas parfait. Vous pouvez suggerer de meilleures formulations pour le glossaire de traduction. Les suggestions sont publiques et doivent etre verifiees avant d'etre ajoutees. Merci d'aider a ameliorer l'application.",
        "glossary_language": "Langue",
        "bad_phrase": "Phrase a ameliorer",
        "preferred_phrase": "Phrase preferee",
        "glossary_submit": "Ajouter une suggestion",
        "glossary_table": "Suggestions publiques en attente de verification",
    },
    "German": {
        "intro": "Laden Sie ein Foto eines Arzneimitteletiketts hoch. Dieses Tool versucht, das Etikett zu lesen und zu erklaeren, wofuer das Arzneimittel ueblicherweise verwendet wird. Es ersetzt keinen Apotheker oder Arzt.",
        "image_label": "Foto des Arzneimitteletiketts",
        "image_help": "Laden Sie ein Bild hoch oder verwenden Sie die Kameraoption, um ein neues Foto aufzunehmen.",
        "manual_label": "Arzneimittel eingeben",
        "manual_placeholder": "Optional: Arzneimittelname oder Etikettentext eingeben",
        "manual_help": "Geben Sie hier den Arzneimittelnamen oder den Etikettentext ein und starten Sie die Suche erneut.",
        "read_button": "Arzneimitteletikett lesen",
        "match": "Arzneimittel-Treffer",
        "explanation": "Verwendung des Arzneimittels",
        "warning": "Sicherheitshinweis",
        "source_url": "Quellen-URL",
        "model_note": "Diese App verwendet zwei kleine lokale KI-Modelle: MiniCPM zum Lesen von Text aus Bildern und Tiny Aya Global zum Erzeugen von Text. Sie nutzt eine kuratierte Datenbank mit 200 haeufig verwendeten Arzneimitteln in Grossbritannien und den USA, mit Informationen aus NHS Medicines A to Z und British National Formulary Drugs A to Z. Das sind oeffentlich verfuegbare Datenbanken, in denen Sie mehr Informationen zu Arzneimitteln finden koennen. Diese App wurde mit OpenAI Codex entwickelt.",
        "glossary_title": "Uebersetzung verbessern",
        "glossary_note": "Tiny Aya ist fuer Uebersetzungen nuetzlich, aber nicht perfekt. Sie koennen bessere Formulierungen fuer das Uebersetzungsglossar vorschlagen. Vorschlaege sind oeffentlich und muessen geprueft werden, bevor sie hinzugefuegt werden. Vielen Dank fuer Ihre Hilfe.",
        "glossary_language": "Sprache",
        "bad_phrase": "Unpassende Formulierung",
        "preferred_phrase": "Bevorzugte Formulierung",
        "glossary_submit": "Glossarvorschlag hinzufuegen",
        "glossary_table": "Oeffentliche Vorschlaege zur Pruefung",
    },
    "Italian": {
        "intro": "Carica una foto dell'etichetta di un medicinale. Questo strumento prova a leggere l'etichetta e a spiegare per cosa viene usato di solito il medicinale. Non sostituisce un farmacista o un medico.",
        "image_label": "Foto dell'etichetta del medicinale",
        "image_help": "Carica un'immagine oppure usa l'opzione fotocamera per scattare una nuova foto.",
        "manual_label": "Inserisci medicinale",
        "manual_placeholder": "Facoltativo: inserisci il nome del medicinale o il testo dell'etichetta",
        "manual_help": "Inserisci qui il nome del medicinale o il testo dell'etichetta, poi avvia di nuovo la lettura.",
        "read_button": "Leggi etichetta medicinale",
        "match": "Medicinale trovato",
        "explanation": "Uso del medicinale",
        "warning": "Avvertenza di sicurezza",
        "source_url": "URL fonte",
        "model_note": "Questa app usa due piccoli modelli IA locali: MiniCPM per leggere il testo dalle immagini e Tiny Aya Global per generare il testo. Usa un database curato di 200 medicinali comunemente usati nel Regno Unito e negli Stati Uniti, con informazioni prese da NHS Medicines A to Z e British National Formulary Drugs A to Z. Sono database pubblici dove puoi trovare piu informazioni sui medicinali. Questa app e stata sviluppata usando OpenAI Codex.",
        "glossary_title": "Migliora traduzione",
        "glossary_note": "Tiny Aya e utile per la traduzione, ma non e perfetto. Puoi suggerire frasi migliori per il glossario di traduzione. I suggerimenti sono pubblici e devono essere controllati prima di essere aggiunti. Grazie per aiutare a migliorare l'app.",
        "glossary_language": "Lingua",
        "bad_phrase": "Frase da migliorare",
        "preferred_phrase": "Frase preferita",
        "glossary_submit": "Aggiungi suggerimento",
        "glossary_table": "Suggerimenti pubblici in attesa di revisione",
    },
    "Romanian": {
        "intro": "Incarca o fotografie cu eticheta unui medicament. Acest instrument incearca sa citeasca eticheta si sa explice pentru ce este folosit de obicei medicamentul. Nu inlocuieste un farmacist sau un medic.",
        "image_label": "Fotografie cu eticheta medicamentului",
        "image_help": "Incarca o imagine sau foloseste optiunea camera pentru a face o fotografie noua.",
        "manual_label": "Introdu medicamentul",
        "manual_placeholder": "Optional: introdu numele medicamentului sau textul de pe eticheta",
        "manual_help": "Introdu aici numele medicamentului sau textul de pe eticheta, apoi apasa din nou pe citire.",
        "read_button": "Citeste eticheta medicamentului",
        "match": "Medicament gasit",
        "explanation": "Utilizarea medicamentului",
        "warning": "Avertisment de siguranta",
        "source_url": "URL sursa",
        "model_note": "Aceasta aplicatie foloseste doua modele AI locale mici: MiniCPM pentru citirea textului din imagini si Tiny Aya Global pentru generarea textului. Foloseste o baza de date curata manual cu 200 de medicamente folosite frecvent in UK si SUA, cu informatii preluate din NHS Medicines A to Z si British National Formulary Drugs A to Z. Acestea sunt baze de date publice unde poti gasi mai multe informatii despre medicamente. Aplicatia a fost dezvoltata folosind OpenAI Codex.",
        "glossary_title": "Imbunatateste traducerea",
        "glossary_note": "Tiny Aya este util pentru traducere, dar nu este perfect. Poti sugera formulari mai bune pentru glosarul de traducere. Sugestiile sunt publice si trebuie verificate inainte de a fi adaugate. Iti multumim ca ajuti la imbunatatirea aplicatiei.",
        "glossary_language": "Limba",
        "bad_phrase": "Expresie de imbunatatit",
        "preferred_phrase": "Expresie preferata",
        "glossary_submit": "Adauga sugestia",
        "glossary_table": "Sugestii publice in asteptarea verificarii",
    },
    "Spanish": {
        "intro": "Suba una foto de la etiqueta de un medicamento. Esta herramienta intenta leer la etiqueta y explicar para que se usa normalmente el medicamento. No sustituye a un farmaceutico ni a un medico.",
        "image_label": "Foto de la etiqueta del medicamento",
        "image_help": "Suba una imagen o use la opcion de camara para tomar una foto nueva.",
        "manual_label": "Escribir medicamento",
        "manual_placeholder": "Opcional: escriba el nombre del medicamento o el texto de la etiqueta",
        "manual_help": "Escriba aqui el nombre del medicamento o el texto de la etiqueta y vuelva a buscar.",
        "read_button": "Leer etiqueta del medicamento",
        "match": "Medicamento encontrado",
        "explanation": "Uso del medicamento",
        "warning": "Advertencia de seguridad",
        "source_url": "URL de origen",
        "model_note": "Esta aplicacion usa dos pequenos modelos locales de IA: MiniCPM para leer texto de imagenes y Tiny Aya Global para generar texto. Usa una base de datos curada de 200 medicamentos de uso comun en el Reino Unido y Estados Unidos, con informacion tomada de NHS Medicines A to Z y British National Formulary Drugs A to Z. Son bases de datos publicas donde puede encontrar mas informacion sobre los medicamentos. Esta aplicacion se desarrollo con OpenAI Codex.",
        "glossary_title": "Mejorar traduccion",
        "glossary_note": "Tiny Aya es util para la traduccion, pero no es perfecto. Puede sugerir mejores frases para el glosario de traduccion. Las sugerencias son publicas y deben revisarse antes de anadirse. Gracias por ayudar a mejorar la aplicacion.",
        "glossary_language": "Idioma",
        "bad_phrase": "Frase a mejorar",
        "preferred_phrase": "Frase preferida",
        "glossary_submit": "Anadir sugerencia",
        "glossary_table": "Sugerencias publicas pendientes de revision",
    },
}


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


def prepare_image_for_ocr(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)

    width, height = gray.size
    scale = max(1, 1600 // max(width, height))
    if scale > 1:
        gray = gray.resize((width * scale, height * scale), Image.Resampling.LANCZOS)

    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.SHARPEN)
    return gray.point(lambda pixel: 255 if pixel > 160 else 0)


def ocr_quality_score(text: str) -> int:
    tokens = re.findall(r"[a-z0-9]+", clean_text(text))
    useful_tokens = [token for token in tokens if len(token) > 1]
    alpha_chars = sum(1 for char in text if char.isalpha())
    digit_chars = sum(1 for char in text if char.isdigit())
    symbol_chars = sum(1 for char in text if not char.isalnum() and not char.isspace())
    one_letter_tokens = sum(1 for token in tokens if len(token) == 1)

    return (
        len(useful_tokens) * 20
        + min(alpha_chars, 500)
        + min(digit_chars, 80)
        - symbol_chars * 4
        - one_letter_tokens * 8
    )


def run_ocr(image: Image.Image, medicines: pd.DataFrame) -> str:
    prepared_image = prepare_image_for_ocr(image)
    attempts = []

    for candidate_image in [image, prepared_image]:
        for config in ["--psm 6", "--psm 11", "--psm 12", "--psm 7", "--psm 8", "--psm 13"]:
            text = pytesseract.image_to_string(candidate_image, config=config).strip()
            if text:
                match = find_best_match(text, medicines)
                attempts.append((ocr_quality_score(text), match["score"], len(clean_text(text)), text))

    if not attempts:
        return ""

    attempts.sort(reverse=True)
    combined_texts = []
    seen = set()
    for _, _, _, text in attempts:
        cleaned = clean_text(text)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            combined_texts.append(text)

    return "\n\n".join(combined_texts[:4])


def orientation_score(ocr_text: str, medicines: pd.DataFrame) -> int:
    if not ocr_text:
        return 0
    match = find_best_match(ocr_text, medicines)
    return ocr_quality_score(ocr_text) + (match["score"] * 3)


ORIENTATION_NORMAL_FIRST = "Normal first"
ORIENTATION_MIRRORED_FIRST = "Mirrored first"
ORIENTATION_FULL_AUTO = "Full auto"
DEVICE_PHONE_OR_TABLET = "phone_or_tablet"
DEVICE_DESKTOP_OR_LAPTOP = "desktop_or_laptop"
IMAGE_SOURCE_UPLOAD = "upload"
IMAGE_SOURCE_WEBCAM = "webcam"


def readable_image_candidates(image: Image.Image, orientation_mode: str = ORIENTATION_NORMAL_FIRST):
    rotated_180 = image.rotate(180, expand=True)
    original = ("", image)
    mirrored = ("Image was horizontally unmirrored before OCR/model reading.", ImageOps.mirror(image))
    rotated_mirrored = (
        "Image was rotated 180 degrees and horizontally unmirrored before OCR/model reading.",
        ImageOps.mirror(rotated_180),
    )
    rotated = ("Image was rotated 180 degrees before OCR/model reading.", rotated_180)
    vertically_flipped = ("Image was vertically flipped before OCR/model reading.", ImageOps.flip(image))
    rotated_flipped = (
        "Image was rotated 180 degrees and vertically flipped before OCR/model reading.",
        ImageOps.flip(rotated_180),
    )

    if orientation_mode == ORIENTATION_MIRRORED_FIRST:
        return [mirrored, original]
    if orientation_mode == ORIENTATION_FULL_AUTO:
        return [
            original,
            mirrored,
            rotated_mirrored,
            rotated,
            vertically_flipped,
            rotated_flipped,
        ]

    return [
        original,
        mirrored,
    ]


def choose_readable_image_orientation(
    image: Image.Image,
    medicines: pd.DataFrame,
    orientation_mode: str = ORIENTATION_NORMAL_FIRST,
):
    scored_candidates = []
    for note, candidate_image in readable_image_candidates(image, orientation_mode):
        candidate_text = run_ocr(candidate_image, medicines)
        scored_candidates.append((orientation_score(candidate_text, medicines), note, candidate_image, candidate_text))

    original_score, _original_note, original_image, original_text = next(
        candidate for candidate in scored_candidates if not candidate[1]
    )
    best_score, best_note, best_image, best_text = max(scored_candidates, key=lambda item: item[0])
    if best_score > original_score + 25:
        return best_image, best_text, best_note

    return original_image, original_text, ""


def run_staged_vision_ocr(
    image: Image.Image,
    vision_ocr_url: str,
    medicines: pd.DataFrame,
    orientation_mode: str,
    deadline: float,
):
    attempts = []
    for note, candidate_image in readable_image_candidates(image, orientation_mode):
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            break
        vision_text = run_vision_ocr(candidate_image, vision_ocr_url, timeout=max(1.0, remaining_seconds))
        vision_match = find_best_match(vision_text, medicines)
        attempts.append((vision_match["score"], note, candidate_image, vision_text, vision_match))
        if vision_match["score"] >= OCR_MATCH_MIN_SCORE:
            return candidate_image, note, vision_text, vision_match, attempts, False

    if not attempts:
        empty_match = {"score": 0, "matched_name": "", "row": None, "confidence": "low"}
        return image, "", "", empty_match, attempts, True

    _score, note, candidate_image, vision_text, vision_match = max(attempts, key=lambda attempt: attempt[0])
    return candidate_image, note, vision_text, vision_match, attempts, time.monotonic() >= deadline


def pending_read_response(progress_update):
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        progress_update,
        gr.update(),
    )


def complete_read_response(
    ocr_text,
    likely,
    common_uses,
    safety_warning,
    source_url,
    manual_help,
    match_text,
    server_status,
    manual_label_update=None,
):
    if manual_label_update is None:
        manual_label_update = gr.update()
    return (
        ocr_text,
        likely,
        common_uses,
        safety_warning,
        source_url,
        manual_help,
        match_text,
        server_status,
        hide_progress_bar(),
        manual_label_update,
    )


def run_staged_vision_ocr_with_progress(
    image: Image.Image,
    vision_ocr_url: str,
    medicines: pd.DataFrame,
    orientation_mode: str,
    deadline: float,
    total_seconds: int,
):
    attempts = []
    empty_match = {"score": 0, "matched_name": "", "row": None, "confidence": "low"}

    for note, candidate_image in readable_image_candidates(image, orientation_mode):
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            break

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            run_vision_ocr,
            candidate_image,
            vision_ocr_url,
            max(1.0, remaining_seconds),
        )

        try:
            while True:
                try:
                    vision_text = future.result(timeout=0.5)
                    break
                except concurrent.futures.TimeoutError:
                    elapsed_seconds = total_seconds - max(0, deadline - time.monotonic())
                    if elapsed_seconds >= 5:
                        yield pending_read_response(
                            progress_bar_html(
                                elapsed_seconds,
                                total_seconds,
                                "Reading image...",
                            )
                        )
                    if time.monotonic() >= deadline:
                        future.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        if attempts:
                            _score, best_note, best_image, best_text, best_match = max(
                                attempts,
                                key=lambda attempt: attempt[0],
                            )
                            return best_image, best_note, best_text, best_match, attempts, True
                        return image, "", "", empty_match, attempts, True
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        vision_match = find_best_match(vision_text, medicines)
        attempts.append((vision_match["score"], note, candidate_image, vision_text, vision_match))
        if vision_match["score"] >= OCR_MATCH_MIN_SCORE:
            return candidate_image, note, vision_text, vision_match, attempts, False

    if not attempts:
        return image, "", "", empty_match, attempts, True

    _score, note, candidate_image, vision_text, vision_match = max(attempts, key=lambda attempt: attempt[0])
    return candidate_image, note, vision_text, vision_match, attempts, time.monotonic() >= deadline


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{image_base64}"


def run_vision_ocr(image: Image.Image, vision_ocr_url: str, timeout: float = 180) -> str:
    prompt = (
        "Read the medication package label in this image. "
        "Return only the text that appears on the label. "
        "Focus on the brand name, generic or active ingredient, strength, and medicine type. "
        "Do not explain the medicine. Do not guess words that are not visible."
    )
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                ],
            }
        ],
        "max_tokens": 220,
        "temperature": 0,
    }

    response = requests.post(vision_ocr_url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def likely_match_text(match: dict) -> str:
    row = match["row"]
    if row is None:
        return "No likely match found"

    generic_name = row["generic_name"]
    matched_name = match["matched_name"]
    confidence = match["confidence"]
    score = match["score"]

    if clean_text(matched_name) == clean_text(generic_name):
        display_name = generic_name
    else:
        display_name = f"{matched_name} / {generic_name}"

    return f"{display_name}\nConfidence: {confidence} ({score}/100)"


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
        fallback = make_template_explanation(match, "English")
        return (
            f"Local Tiny Aya server was not reached, so this explanation is shown in English.\n\n"
            f"{fallback}\n\n"
            f"Background start attempt: {server_status}\n"
            f"Check the URL: {model_url}\n"
            f"Model error: {error}"
        )

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
        return (
            "Local Tiny Aya server was not reached, so this medicine-specific note is shown in English.\n\n"
            f"{warning}\n\n"
            f"Background start attempt: {server_status}\n"
            f"Model error: {error}"
        )

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


def no_database_info_message(language: str) -> str:
    messages = {
        "English": "No information is available for this medicine in the local database.",
        "French": "Aucune information n'est disponible pour ce medicament dans la base de donnees locale.",
        "German": "Zu diesem Arzneimittel sind in der lokalen Datenbank keine Informationen verfuegbar.",
        "Italian": "Non sono disponibili informazioni su questo medicinale nel database locale.",
        "Romanian": "Nu exista informatii despre acest medicament in baza de date locala.",
        "Spanish": "No hay informacion disponible sobre este medicamento en la base de datos local.",
    }
    return messages[language]


def manual_entry_prompt(language: str) -> str:
    messages = {
        "English": "N/A\nCould not process image. Type the medicine name or label text, then search again.",
        "French": "N/A\nImpossible de traiter l'image. Saisissez le nom du medicament ou le texte de l'etiquette, puis relancez la recherche.",
        "German": "N/A\nDas Bild konnte nicht verarbeitet werden. Geben Sie den Arzneimittelnamen oder den Etikettentext ein und suchen Sie erneut.",
        "Italian": "N/A\nImpossibile elaborare l'immagine. Inserisci il nome del medicinale o il testo dell'etichetta, poi cerca di nuovo.",
        "Romanian": "N/A\nImaginea nu a putut fi procesata. Introdu numele medicamentului sau textul de pe eticheta, apoi cauta din nou.",
        "Spanish": "N/A\nNo se pudo procesar la imagen. Escriba el nombre del medicamento o el texto de la etiqueta y busque de nuevo.",
    }
    return messages[language]


def manual_label_help_update(language: str, visible: bool = True):
    return gr.update(visible=False)


def hide_manual_label_help():
    return gr.update(visible=False)


def manual_label_optional_placeholder(language: str):
    return gr.update(placeholder=UI_LABELS[language]["manual_placeholder"])


def manual_label_required_placeholder(language: str):
    prompt = UI_LABELS[language]["manual_help"]
    return gr.update(placeholder=prompt)


def clear_manual_prompt_after_typing(text):
    next_response_version()
    if (text or "").strip():
        return hide_manual_label_help(), "", None
    return hide_manual_label_help(), "", gr.update()


def begin_read_attempt(image_timeout_seconds):
    total_seconds = normalize_image_timeout_seconds(image_timeout_seconds)
    return (
        gr.update(interactive=False),
        progress_bar_html(0, total_seconds, "Processing image..."),
    )


def enable_read_button():
    return gr.update(interactive=True)


def update_ui_language(language: str):
    labels = UI_LABELS[language]
    return (
        gr.update(value=f"# MediLens Local: Medicine Label Helper\n{labels['intro']}"),
        gr.update(label=labels["image_label"], show_label=True),
        gr.update(value=labels["image_help"]),
        gr.update(label=labels["manual_label"], placeholder=labels["manual_placeholder"]),
        gr.update(value=labels["read_button"]),
        gr.update(label=labels["match"]),
        gr.update(label=labels["explanation"]),
        gr.update(label=labels["warning"]),
        gr.update(label=labels["source_url"]),
        manual_label_help_update(language, visible=False),
        gr.update(value=labels["model_note"]),
        gr.update(label=labels["glossary_title"]),
        gr.update(value=f'<div id="glossary-note">{labels["glossary_note"]}</div>'),
        gr.update(value=language, label=labels["glossary_language"]),
        gr.update(label=labels["bad_phrase"]),
        gr.update(label=labels["preferred_phrase"]),
        gr.update(value=labels["glossary_submit"]),
        gr.update(label=labels["glossary_table"], value=load_glossary_suggestions()),
    )


def browser_language_to_supported(browser_language: str) -> str:
    language_code = (browser_language or "").lower().split("-")[0].split("_")[0]
    browser_language_map = {
        "en": "English",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "ro": "Romanian",
        "es": "Spanish",
    }
    return browser_language_map.get(language_code, "English")


def normalize_browser_device(browser_device: str) -> str:
    if browser_device == DEVICE_PHONE_OR_TABLET:
        return DEVICE_PHONE_OR_TABLET
    return DEVICE_DESKTOP_OR_LAPTOP


def default_orientation_for_device(browser_device: str) -> str:
    return ORIENTATION_NORMAL_FIRST


def normalize_image_timeout_seconds(timeout_seconds) -> int:
    try:
        seconds = int(timeout_seconds)
    except (TypeError, ValueError):
        seconds = DEFAULT_IMAGE_IDENTIFICATION_SECONDS
    return max(MIN_IMAGE_IDENTIFICATION_SECONDS, min(MAX_IMAGE_IDENTIFICATION_SECONDS, seconds))


def apply_browser_language(browser_language: str, browser_device: str = DEVICE_DESKTOP_OR_LAPTOP):
    language = browser_language_to_supported(browser_language)
    device = normalize_browser_device(browser_device)
    return (
        gr.update(value=language),
        *update_ui_language(language),
        gr.update(value=device),
        gr.update(value=default_orientation_for_device(device)),
    )


def update_language_and_refresh_result(
    language: str,
    use_ai_model: bool,
    model_url: str,
    current_ocr: str,
    match_text_state: str,
):
    next_response_version()
    ui_updates = list(update_ui_language(language))
    manual_helper_update = ui_updates.pop(9)
    labels = UI_LABELS[language]
    ui_updates[5] = gr.update(label=labels["match"], value="")
    ui_updates[6] = gr.update(label=labels["explanation"], value="")
    ui_updates[7] = gr.update(label=labels["warning"], value="")
    ui_updates[8] = gr.update(label=labels["source_url"], value="")
    return (*ui_updates, "", "", manual_helper_update)


def clear_results_for_new_image(
    _image,
    language: str,
    vision_ocr_url: str,
    model_url: str,
    browser_device: str,
    image_source: str,
):
    next_response_version()
    server_status = check_local_model_servers(vision_ocr_url, model_url)
    manual_label_update = "" if _image is not None else gr.update()
    if _image is not None:
        manual_label_update = gr.update(value="", placeholder=UI_LABELS[language]["manual_placeholder"])
    if _image is None:
        source_update = ""
        orientation_update = gr.update(value=ORIENTATION_NORMAL_FIRST)
    elif normalize_browser_device(browser_device) == DEVICE_PHONE_OR_TABLET:
        source_update = image_source or IMAGE_SOURCE_UPLOAD
        orientation_update = gr.update(value=ORIENTATION_NORMAL_FIRST)
    else:
        source_update = image_source or IMAGE_SOURCE_WEBCAM
        orientation_update = gr.update(
            value=ORIENTATION_FULL_AUTO if source_update == IMAGE_SOURCE_WEBCAM else ORIENTATION_NORMAL_FIRST
        )

    return (
        manual_label_update,
        "",
        "",
        "",
        "",
        "",
        server_status,
        hide_manual_label_help(),
        "",
        source_update,
        orientation_update,
    )


def mark_uploaded_image_source(browser_device: str):
    return IMAGE_SOURCE_UPLOAD, gr.update(value=default_orientation_for_device(browser_device))


def read_label(
    image,
    language: str,
    manual_label: str,
    use_vision_ocr: bool,
    vision_ocr_url: str,
    orientation_mode: str,
    image_timeout_seconds,
    use_ai_model: bool,
    model_url: str,
):
    read_version = next_response_version()
    manual_label = (manual_label or "").strip()
    vision_ocr_url = normalize_local_url(vision_ocr_url, DEFAULT_VISION_OCR_URL)
    model_url = normalize_local_url(model_url, DEFAULT_MODEL_URL)
    image_timeout_seconds = normalize_image_timeout_seconds(image_timeout_seconds)
    server_status_output = gr.update()
    if image is None:
        if manual_label:
            ocr_text = f"Entered medicine label:\n{manual_label}"
        else:
            empty = "Please upload a medicine label image first, or type the medicine label in the search box."
            yield complete_read_response(
                empty,
                "N/A",
                empty,
                "N/A",
                "",
                hide_manual_label_help(),
                "",
                server_status_output,
                manual_label_required_placeholder(language),
            )
            return
    else:
        pil_image = Image.fromarray(image).convert("RGB")
        ocr_text = ""
        server_status_output = check_local_model_servers(vision_ocr_url, model_url)

    medicines = load_medicines()
    match_text = manual_label
    can_match_ocr = False

    if image is not None and use_vision_ocr:
        tesseract_text = ""
        orientation_note = ""
        try:
            image_deadline = time.monotonic() + image_timeout_seconds
            pil_image, orientation_note, vision_text, vision_match, _vision_attempts, vision_timed_out = yield from run_staged_vision_ocr_with_progress(
                pil_image,
                vision_ocr_url,
                medicines,
                orientation_mode,
                image_deadline,
                image_timeout_seconds,
            )
            ocr_parts = []
            if orientation_note:
                ocr_parts.append(orientation_note)
            if vision_text:
                ocr_parts.append(f"MiniCPM-V OCR:\n{vision_text}")
            if vision_match["score"] >= OCR_MATCH_MIN_SCORE:
                match_text = vision_text
            elif vision_timed_out and not manual_label:
                ocr_text = (
                    f"{chr(10).join(ocr_parts)}\n\n"
                    f"N/A. MiniCPM-V could not read the image within {image_timeout_seconds} seconds. "
                    "Type the medicine name or label text, then search again."
                ).strip()
                if read_version != current_response_version():
                    yield stale_read_response()
                    return
                yield complete_read_response(
                    ocr_text,
                    "N/A",
                    manual_entry_prompt(language),
                    "N/A",
                    "",
                    hide_manual_label_help(),
                    "",
                    server_status_output,
                    manual_label_required_placeholder(language),
                )
                return
            elif not manual_label:
                try:
                    tesseract_text = run_ocr(pil_image, medicines)
                except TesseractNotFoundError:
                    tesseract_text = ""
                if tesseract_text:
                    ocr_parts.append(f"Tesseract OCR:\n{tesseract_text}")
                ocr_text = (
                    f"{chr(10).join(ocr_parts)}\n\n"
                    "MiniCPM-V did not produce a high-confidence medicine match, "
                    "so the app will not identify this medicine from OCR alone."
                ).strip()
            ocr_text = ocr_text or "\n\n".join(ocr_parts)
        except Exception:
            server_status = start_llama_server(MINICPM_V_MODEL_REF, vision_ocr_url)
            server_status_output = f"MiniCPM-V OCR: {server_status}\nTiny Aya: {'reachable' if is_port_reachable(model_url) else 'not reachable'}"
            try:
                pil_image, tesseract_text, orientation_note = choose_readable_image_orientation(
                    pil_image,
                    medicines,
                    orientation_mode,
                )
            except TesseractNotFoundError:
                tesseract_text = ""
            if tesseract_text:
                ocr_text = (
                    "MiniCPM-V OCR server was not reached, so Tesseract OCR was used instead.\n\n"
                    f"Background start attempt: {server_status}\n\n"
                    f"{orientation_note + chr(10) + chr(10) if orientation_note else ''}"
                    f"{tesseract_text}"
                )
                can_match_ocr = False
            else:
                message = (
                    "MiniCPM-V OCR server was not reached, and Tesseract OCR is not installed "
                    "or is not on your PATH.\n\n"
                    f"Background start attempt: {server_status}"
                )
                if manual_label:
                    ocr_text = message
                else:
                    if read_version != current_response_version():
                        yield stale_read_response()
                        return
                    yield complete_read_response(
                        message,
                        "N/A",
                        manual_entry_prompt(language),
                        "N/A",
                        "",
                        hide_manual_label_help(),
                        "",
                        server_status_output,
                        manual_label_required_placeholder(language),
                    )
                    return
    elif image is not None:
        try:
            _selected_image, tesseract_text, orientation_note = choose_readable_image_orientation(
                pil_image,
                medicines,
                orientation_mode,
            )
            ocr_text = "\n\n".join(
                part
                for part in [
                    orientation_note,
                    "Tesseract-only OCR is shown for technical details, but it is not used to identify a medicine.",
                    tesseract_text,
                ]
                if part
            )
        except TesseractNotFoundError:
            message = (
                "Tesseract OCR is not installed or is not on your PATH. "
                "Install Tesseract OCR for Windows, then restart this app."
            )
            if read_version != current_response_version():
                yield stale_read_response()
                return
            yield complete_read_response(
                message,
                "OCR setup needed",
                message,
                "N/A",
                "",
                hide_manual_label_help(),
                "",
                server_status_output,
                manual_label_required_placeholder(language),
            )
            return

    if read_version != current_response_version():
        yield stale_read_response()
        return

    if not ocr_text:
        ocr_text = "No text was detected. Please try a clearer, well-lit photo."

    if not match_text and can_match_ocr:
        match_text = ocr_text

    if not match_text:
        yield complete_read_response(
            ocr_text,
            "N/A",
            manual_entry_prompt(language),
            "N/A",
            "",
            hide_manual_label_help(),
            "",
            server_status_output,
            manual_label_required_placeholder(language),
        )
        return

    match = find_best_match(match_text, medicines)
    if manual_label:
        match = apply_manual_match_safety(match)

    if match["confidence"] == "low":
        likely = "N/A"
        if manual_label:
            common_uses = no_database_info_message(language)
            manual_help = hide_manual_label_help()
            manual_label_update = manual_label_optional_placeholder(language)
        else:
            common_uses = manual_entry_prompt(language)
            manual_help = hide_manual_label_help()
            manual_label_update = manual_label_required_placeholder(language)
        medicine_warning = ""
        source_url = ""
    else:
        likely = likely_match_text(match)
        common_uses = make_explanation(match, language, ocr_text, use_ai_model, model_url)
        medicine_warning = match["row"]["safety_warning"]
        source_url = match["row"].get("source_url", "")
        manual_help = hide_manual_label_help()
        manual_label_update = manual_label_optional_placeholder(language)

    if medicine_warning:
        translated_warning = translate_medicine_warning(medicine_warning, language, use_ai_model, model_url)
        safety_warning = translated_warning
    else:
        safety_warning = GENERAL_SAFETY_WARNING[language]

    if read_version != current_response_version():
        yield stale_read_response()
        return

    yield complete_read_response(
        ocr_text,
        likely,
        common_uses,
        safety_warning,
        source_url,
        manual_help,
        match_text,
        server_status_output,
        manual_label_update,
    )
    return


CUSTOM_CSS = """
#medicine-image {
    height: 280px;
}

#medicine-image .image-container,
#medicine-image .image-preview,
#medicine-image img {
    height: 280px !important;
    max-height: 280px !important;
    object-fit: contain !important;
}

#ocr-output textarea,
#match-output textarea {
    height: 96px !important;
    min-height: 96px !important;
    max-height: 96px !important;
    overflow-y: auto !important;
}

#top-left-panel,
#top-right-panel {
    min-height: 540px;
}

#technical-left-panel,
#technical-right-panel {
    min-height: 220px;
}

#language-selector {
    border: 2px solid #f97316;
    border-radius: 8px;
    padding: 8px;
    background: #fff7ed;
}

#manual-label-helper {
    border: 2px solid #f97316;
    border-radius: 8px;
    padding: 10px 12px;
    background: #fff7ed;
    color: #7c2d12;
}

#manual-label-section:has(#manual-label-helper:not([style*="display: none"])) {
    border: 2px solid #f97316;
    border-radius: 8px;
    padding: 8px;
    background: #fff7ed;
}

#manual-label-section:has(#manual-label-helper:not([style*="display: none"])) #manual-label-input textarea {
    border-color: #f97316 !important;
    box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.2) !important;
}

#glossary-note {
    border-left: 4px solid #f97316;
    padding: 8px 12px;
    background: #fff7ed;
}

#processing-progress {
    min-height: 18px;
    margin-top: -3px !important;
    margin-bottom: 0;
}

#read-action-section {
    gap: 2px !important;
}

#read-action-section > .form,
#read-action-section .block {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

.single-progress-status {
    font-size: 12px;
    line-height: 18px;
    color: #4b5563;
}
"""


with gr.Blocks(title="MediLens Local: Medicine Label Helper", css=CUSTOM_CSS) as demo:
    intro_markdown = gr.Markdown(f"# MediLens Local: Medicine Label Helper\n{UI_LABELS['English']['intro']}")
    browser_language_input = gr.Textbox(visible=False)
    browser_device_input = gr.Textbox(value=DEVICE_DESKTOP_OR_LAPTOP, visible=False)
    image_source_input = gr.Textbox(value="", visible=False)
    match_text_state = gr.State("")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_id="top-left-panel"):
            image_input = gr.Image(
                label=UI_LABELS["English"]["image_label"],
                sources=["upload", "webcam"],
                type="numpy",
                height=280,
                elem_id="medicine-image",
            )
            image_help_markdown = gr.Markdown(UI_LABELS["English"]["image_help"], visible=False)
            with gr.Group(elem_id="manual-label-section"):
                manual_label_input = gr.Textbox(
                    label=UI_LABELS["English"]["manual_label"],
                    placeholder=UI_LABELS["English"]["manual_placeholder"],
                    elem_id="manual-label-input",
                )
                manual_label_helper = gr.Markdown(
                    UI_LABELS["English"]["manual_help"],
                    visible=False,
                    elem_id="manual-label-helper",
            )
            language_input = gr.Dropdown(
                choices=["English", "French", "German", "Italian", "Romanian", "Spanish"],
                value="English",
                label="Language",
                elem_id="language-selector",
            )
            with gr.Column(elem_id="read-action-section"):
                read_button = gr.Button("Read medicine label", variant="primary")
                processing_progress = gr.HTML("", visible=True, elem_id="processing-progress")

        with gr.Column(scale=1, elem_id="top-right-panel"):
            match_output = gr.Textbox(label=UI_LABELS["English"]["match"], lines=3, elem_id="match-output")
            explanation_output = gr.Textbox(label=UI_LABELS["English"]["explanation"], lines=5)
            warning_output = gr.Textbox(label=UI_LABELS["English"]["warning"], lines=5)
            source_url_output = gr.Textbox(label=UI_LABELS["English"]["source_url"], lines=1)

    model_note_markdown = gr.Markdown(UI_LABELS["English"]["model_note"])

    with gr.Accordion(UI_LABELS["English"]["glossary_title"], open=False) as glossary_accordion:
        glossary_note = gr.Markdown(f'<div id="glossary-note">{UI_LABELS["English"]["glossary_note"]}</div>')
        with gr.Row(equal_height=True):
            glossary_language_input = gr.Dropdown(
                choices=["English", "French", "German", "Italian", "Romanian", "Spanish"],
                value="Romanian",
                label=UI_LABELS["English"]["glossary_language"],
            )
            bad_phrase_input = gr.Textbox(label=UI_LABELS["English"]["bad_phrase"])
            preferred_phrase_input = gr.Textbox(label=UI_LABELS["English"]["preferred_phrase"])
        submit_glossary_button = gr.Button(UI_LABELS["English"]["glossary_submit"])
        glossary_suggestions_table = gr.Dataframe(
            value=load_glossary_suggestions(),
            headers=["language", "bad_phrase", "preferred_phrase"],
            datatype=["str", "str", "str"],
            label=UI_LABELS["English"]["glossary_table"],
            interactive=False,
        )

    with gr.Accordion("Technical details", open=False):
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_id="technical-left-panel"):
                use_vision_ocr_input = gr.Checkbox(
                    value=True,
                    label="Use local MiniCPM-V OCR server with Tesseract",
                )
                orientation_mode_input = gr.Dropdown(
                    choices=[
                        ORIENTATION_NORMAL_FIRST,
                        ORIENTATION_MIRRORED_FIRST,
                        ORIENTATION_FULL_AUTO,
                    ],
                    value=ORIENTATION_NORMAL_FIRST,
                    label="Image orientation mode",
                )
                image_timeout_input = gr.Number(
                    value=DEFAULT_IMAGE_IDENTIFICATION_SECONDS,
                    minimum=MIN_IMAGE_IDENTIFICATION_SECONDS,
                    maximum=MAX_IMAGE_IDENTIFICATION_SECONDS,
                    precision=0,
                    label="Max image wait time, seconds",
                )
                use_ai_model_input = gr.Checkbox(
                    value=True,
                    label="Use local Tiny Aya Global server for explanation",
                )
                server_status_output = gr.Textbox(label="Local model server status", lines=2)
                start_servers_button = gr.Button("Start/check local model servers")
            with gr.Column(scale=1, elem_id="technical-right-panel"):
                vision_ocr_url_input = gr.Textbox(
                    value=DEFAULT_VISION_OCR_URL,
                    label="Local MiniCPM-V OCR server URL",
                    placeholder="http://127.0.0.1:8081/v1/chat/completions",
                )
                model_url_input = gr.Textbox(
                    value=DEFAULT_MODEL_URL,
                    label="Local Tiny Aya server URL",
                    placeholder="http://127.0.0.1:8080/v1/chat/completions",
                )
                ocr_output = gr.Textbox(label="Extracted OCR text", lines=3, elem_id="ocr-output")

    demo.load(
        fn=apply_browser_language,
        inputs=[browser_language_input, browser_device_input],
        outputs=[
            language_input,
            intro_markdown,
            image_input,
            image_help_markdown,
            manual_label_input,
            read_button,
            match_output,
            explanation_output,
            warning_output,
            source_url_output,
            manual_label_helper,
            model_note_markdown,
            glossary_accordion,
            glossary_note,
            glossary_language_input,
            bad_phrase_input,
            preferred_phrase_input,
            submit_glossary_button,
            glossary_suggestions_table,
            browser_device_input,
            orientation_mode_input,
        ],
        js="""() => {
            const language = navigator.language || navigator.userLanguage || "";
            const userAgent = navigator.userAgent || "";
            const isPhoneOrTablet = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(userAgent)
                || (navigator.maxTouchPoints > 1 && Math.min(screen.width, screen.height) <= 1024);
            return [language, isPhoneOrTablet ? "phone_or_tablet" : "desktop_or_laptop"];
        }""",
    )

    image_input.change(
        fn=clear_results_for_new_image,
        inputs=[
            image_input,
            language_input,
            vision_ocr_url_input,
            model_url_input,
            browser_device_input,
            image_source_input,
        ],
        outputs=[
            manual_label_input,
            ocr_output,
            match_output,
            explanation_output,
            warning_output,
            source_url_output,
            server_status_output,
            manual_label_helper,
            match_text_state,
            image_source_input,
            orientation_mode_input,
        ],
    )

    image_input.upload(
        fn=mark_uploaded_image_source,
        inputs=[browser_device_input],
        outputs=[image_source_input, orientation_mode_input],
    )

    language_input.change(
        fn=update_language_and_refresh_result,
        inputs=[
            language_input,
            use_ai_model_input,
            model_url_input,
            ocr_output,
            match_text_state,
        ],
        outputs=[
            intro_markdown,
            image_input,
            image_help_markdown,
            manual_label_input,
            read_button,
            match_output,
            explanation_output,
            warning_output,
            source_url_output,
            model_note_markdown,
            glossary_accordion,
            glossary_note,
            glossary_language_input,
            bad_phrase_input,
            preferred_phrase_input,
            submit_glossary_button,
            glossary_suggestions_table,
            ocr_output,
            match_text_state,
            manual_label_helper,
        ],
    )

    manual_label_input.change(
        fn=clear_manual_prompt_after_typing,
        inputs=[manual_label_input],
        outputs=[manual_label_helper, match_text_state, image_input],
    )

    start_servers_button.click(
        fn=start_local_model_servers,
        inputs=[vision_ocr_url_input, model_url_input],
        outputs=[server_status_output],
    )

    submit_glossary_button.click(
        fn=submit_glossary_suggestion,
        inputs=[glossary_language_input, bad_phrase_input, preferred_phrase_input],
        outputs=[glossary_suggestions_table, bad_phrase_input, preferred_phrase_input],
    )

    read_event = read_button.click(
        fn=begin_read_attempt,
        inputs=[image_timeout_input],
        outputs=[read_button, processing_progress],
    )

    read_event.then(
        fn=read_label,
        inputs=[
            image_input,
            language_input,
            manual_label_input,
            use_vision_ocr_input,
            vision_ocr_url_input,
            orientation_mode_input,
            image_timeout_input,
            use_ai_model_input,
            model_url_input,
        ],
        outputs=[
            ocr_output,
            match_output,
            explanation_output,
            warning_output,
            source_url_output,
            manual_label_helper,
            match_text_state,
            server_status_output,
            processing_progress,
            manual_label_input,
        ],
        show_progress="hidden",
    ).then(
        fn=enable_read_button,
        inputs=[],
        outputs=[read_button],
    )


if __name__ == "__main__":
    launch_kwargs = {"server_port": APP_SERVER_PORT, "share": APP_SHARE}
    if APP_SERVER_NAME:
        launch_kwargs["server_name"] = APP_SERVER_NAME
    demo.launch(**launch_kwargs)
