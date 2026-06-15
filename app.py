from pathlib import Path
import concurrent.futures
import html
import json
import os
import subprocess
import sys
import threading
import time

import gradio as gr
import pandas as pd
from pytesseract import TesseractNotFoundError
from PIL import Image

from medilens_core.config import (
    DEFAULT_IMAGE_IDENTIFICATION_SECONDS,
    DEFAULT_MODEL_URL,
    DEFAULT_VISION_OCR_URL,
    DEVICE_DESKTOP_OR_LAPTOP,
    DEVICE_PHONE_OR_TABLET,
    IMAGE_SOURCE_UPLOAD,
    IMAGE_SOURCE_WEBCAM,
    MAX_IMAGE_IDENTIFICATION_SECONDS,
    MIN_IMAGE_IDENTIFICATION_SECONDS,
    MINICPM_V_MODEL_REF,
    OCR_MATCH_MIN_SCORE,
    ORIENTATION_FULL_AUTO,
    ORIENTATION_MIRRORED_FIRST,
    ORIENTATION_NORMAL_FIRST,
    TINY_AYA_MODEL_REF,
    normalize_image_timeout_seconds,
)
from medilens_core.database import load_glossary_suggestions, load_medicines, submit_glossary_suggestion
from medilens_core.explanations import TinyAyaUnavailableError, make_explanation, translate_medicine_warning
from medilens_core.matching import apply_manual_match_safety, capitalize_name, clean_text, find_best_match
from medilens_core.messages import GENERAL_SAFETY_WARNING, manual_entry_prompt, no_database_info_message
from medilens_core.models import (
    check_local_model_servers,
    is_port_reachable,
    normalize_local_url,
    start_llama_server,
    start_local_model_servers,
)
from medilens_core.ocr import choose_readable_image_orientation, readable_image_candidates, run_ocr, run_vision_ocr


APP_SERVER_NAME = os.getenv("MEDILENS_SERVER_NAME")
APP_SERVER_PORT = int(os.getenv("MEDILENS_SERVER_PORT", "7860"))
APP_SHARE = os.getenv("MEDILENS_SHARE", "").strip().lower() in {"1", "true", "yes", "on"}

# True when running on a Hugging Face Space (HF sets SPACE_ID). On the hosted
# CPU demo there is no GPU, no llama.cpp, and no Reachy Mini, so the local-model
# features cannot run. The UI stays identical; we just show a note and start with
# the offline template path so the app is usable without errors.
IS_HOSTED = bool(os.getenv("SPACE_ID"))
HOSTED_BANNER_HTML = """
<div id="hosted-note">
  <strong>Hosted demo (no GPU).</strong> This is the live MediLens app, but a free
  Hugging Face Space has no GPU, so the local AI models (MiniCPM-V 4.6 vision OCR and
  Tiny Aya Global translation) and the Reachy Mini robot cannot run here. Medicine
  lookup and offline multilingual explanations work fully. To use the AI models and
  the robot, run MediLens on your own computer from the GitHub repository - see the
  demo video for the complete experience.
</div>
"""
PROJECT_ROOT = Path(__file__).resolve().parent
REACHY_SERVICE_HOST = os.getenv("MEDILENS_REACHY_SERVICE_HOST", "0.0.0.0")
REACHY_SERVICE_PORT = int(os.getenv("MEDILENS_REACHY_SERVICE_PORT", "8765"))
REACHY_SERVICE_URL = os.getenv("MEDILENS_REACHY_SERVICE_URL", f"http://127.0.0.1:{REACHY_SERVICE_PORT}")
REACHY_HOST = os.getenv("MEDILENS_REACHY_HOST", "reachy-mini.local")
REACHY_PORT = int(os.getenv("MEDILENS_REACHY_PORT", "8000"))
REACHY_MIC = os.getenv("MEDILENS_REACHY_MIC", "reachy")
REACHY_LOG_DIR = PROJECT_ROOT / "robot" / "logs"
REACHY_STATUS_FILE = REACHY_LOG_DIR / "reachy_status.txt"
REACHY_RESULT_FILE = REACHY_LOG_DIR / "reachy_result.json"
REACHY_STOP_FILE = REACHY_LOG_DIR / "reachy_stop_requested.txt"
MODEL_START_WAIT_SECONDS = int(os.getenv("MEDILENS_MODEL_START_WAIT_SECONDS", "75"))
REACHY_PROCESS_LOCK = threading.Lock()
REACHY_SERVICE_PROCESS = None
REACHY_APP_PROCESS = None
RESPONSE_VERSION = 0
RESPONSE_VERSION_LOCK = threading.Lock()


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


UI_LABELS = {
    "English": {
        "intro": "Upload a photo of a medicine label. This tool tries to read the label and explain what the medicine is commonly used for. It does not replace a pharmacist or doctor.",
        "image_label": "Photo",
        "image_help": "Upload an image, or use the camera option to capture a new photo.",
        "manual_label": "Medicine name",
        "manual_placeholder": "Optional: type medicine name",
        "manual_help": "Type medicine name, then press Search button",
        "read_button": "Search",
        "match": "Medicine match",
        "explanation": "Medicine use",
        "warning": "Safety warning",
        "source_url": "Source URL",
        "model_note": "This app uses two small local AI models: MiniCPM-V 4.6 and Tiny Aya Global. Medicine information comes from a database of 200 medicines, with medicine uses and safety warnings based on NHS Medicines A to Z and the British National Formulary.",
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
        "image_label": "Photo",
        "image_help": "Importez une image ou utilisez l'option appareil photo pour prendre une nouvelle photo.",
        "manual_label": "Nom du medicament",
        "manual_placeholder": "Facultatif : saisissez le nom du medicament",
        "manual_help": "Saisissez le nom du medicament, puis appuyez sur Rechercher",
        "read_button": "Rechercher",
        "match": "Medicament trouve",
        "explanation": "Utilisation du medicament",
        "warning": "Avertissement de securite",
        "source_url": "URL source",
        "model_note": "Cette application utilise deux petits modeles d'IA locaux : MiniCPM-V 4.6 et Tiny Aya Global. Les informations proviennent d'une base de donnees de 200 medicaments ; les usages et les avertissements de securite reposent sur NHS Medicines A to Z et le British National Formulary.",
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
        "image_label": "Foto",
        "image_help": "Laden Sie ein Bild hoch oder verwenden Sie die Kameraoption, um ein neues Foto aufzunehmen.",
        "manual_label": "Arzneimittelname",
        "manual_placeholder": "Optional: Arzneimittelname eingeben",
        "manual_help": "Arzneimittelname eingeben, dann auf Suchen klicken",
        "read_button": "Suchen",
        "match": "Arzneimittel-Treffer",
        "explanation": "Verwendung des Arzneimittels",
        "warning": "Sicherheitshinweis",
        "source_url": "Quellen-URL",
        "model_note": "Diese App verwendet zwei kleine lokale KI-Modelle: MiniCPM-V 4.6 und Tiny Aya Global. Die Arzneimittelinformationen stammen aus einer Datenbank mit 200 Arzneimitteln; Anwendung und Sicherheitshinweise basieren auf NHS Medicines A to Z und dem British National Formulary.",
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
        "image_label": "Foto",
        "image_help": "Carica un'immagine oppure usa l'opzione fotocamera per scattare una nuova foto.",
        "manual_label": "Nome del medicinale",
        "manual_placeholder": "Facoltativo: inserisci il nome del medicinale",
        "manual_help": "Inserisci il nome del medicinale, poi premi Cerca",
        "read_button": "Cerca",
        "match": "Medicinale trovato",
        "explanation": "Uso del medicinale",
        "warning": "Avvertenza di sicurezza",
        "source_url": "URL fonte",
        "model_note": "Questa app usa due piccoli modelli IA locali: MiniCPM-V 4.6 e Tiny Aya Global. Le informazioni provengono da un database di 200 medicinali; usi e avvertenze di sicurezza si basano su NHS Medicines A to Z e British National Formulary.",
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
        "image_label": "Fotografie",
        "image_help": "Incarca o imagine sau foloseste optiunea camera pentru a face o fotografie noua.",
        "manual_label": "Numele medicamentului",
        "manual_placeholder": "Optional: scrie numele medicamentului",
        "manual_help": "Scrie numele medicamentului, apoi apasa Cauta",
        "read_button": "Cauta",
        "match": "Medicament gasit",
        "explanation": "Utilizarea medicamentului",
        "warning": "Avertisment de siguranta",
        "source_url": "URL sursa",
        "model_note": "Aceasta aplicatie foloseste doua modele AI locale mici: MiniCPM-V 4.6 si Tiny Aya Global. Informatiile provin dintr-o baza de date cu 200 de medicamente; utilizarea medicamentelor si avertismentele de siguranta se bazeaza pe NHS Medicines A to Z si British National Formulary.",
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
        "image_label": "Foto",
        "image_help": "Suba una imagen o use la opcion de camara para tomar una foto nueva.",
        "manual_label": "Nombre del medicamento",
        "manual_placeholder": "Opcional: escriba el nombre del medicamento",
        "manual_help": "Escriba el nombre del medicamento y pulse Buscar",
        "read_button": "Buscar",
        "match": "Medicamento encontrado",
        "explanation": "Uso del medicamento",
        "warning": "Advertencia de seguridad",
        "source_url": "URL de origen",
        "model_note": "Esta aplicacion usa dos pequenos modelos locales de IA: MiniCPM-V 4.6 y Tiny Aya Global. La informacion proviene de una base de datos de 200 medicamentos; los usos y las advertencias de seguridad se basan en NHS Medicines A to Z y British National Formulary.",
        "glossary_title": "Mejorar traduccion",
        "glossary_note": "Tiny Aya es util para la traduccion, pero no es perfecto. Puede sugerir mejores frases para el glosario de traduccion. Las sugerencias son publicas y deben revisarse antes de anadirse. Gracias por ayudar a mejorar la aplicacion.",
        "glossary_language": "Idioma",
        "bad_phrase": "Frase a mejorar",
        "preferred_phrase": "Frase preferida",
        "glossary_submit": "Anadir sugerencia",
        "glossary_table": "Sugerencias publicas pendientes de revision",
    },
}


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
        source_url_html(source_url),
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
    max_attempts_per_orientation: int = 2,
):
    attempts = []
    empty_match = {"score": 0, "matched_name": "", "row": None, "confidence": "low"}

    for note, candidate_image in readable_image_candidates(image, orientation_mode):
        for _attempt_index in range(max(1, int(max_attempts_per_orientation))):
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

        if time.monotonic() >= deadline:
            break

    if not attempts:
        return image, "", "", empty_match, attempts, True

    _score, note, candidate_image, vision_text, vision_match = max(attempts, key=lambda attempt: attempt[0])
    return candidate_image, note, vision_text, vision_match, attempts, time.monotonic() >= deadline


MATCH_NA_HTML = '<p class="match-na">N/A</p>'
MATCH_OCR_SETUP_HTML = '<p class="match-na">OCR setup needed</p>'


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


def reachy_result_match_html(result: dict) -> str:
    if not result.get("found"):
        return MATCH_NA_HTML

    medicine_name = html.escape(str(result.get("medicine_name") or "Medicine found"))
    confidence = html.escape(str(result.get("confidence") or "low"))
    score = html.escape(str(result.get("score") or "0"))
    badge_class = f"confidence-{confidence}"
    return (
        f'<div class="match-card">'
        f'<p class="match-medicine-name">{medicine_name}</p>'
        f'<span class="confidence-badge {badge_class}">{confidence.capitalize()} confidence &middot; {score}/100</span>'
        f"</div>"
    )


def reachy_result_updates():
    try:
        result_payload = json.loads(REACHY_RESULT_FILE.read_text(encoding="utf-8"))
        result = result_payload.get("result") or {}
    except (OSError, json.JSONDecodeError):
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    ocr_text = (result.get("ocr_text") or "").strip()
    if not ocr_text:
        ocr_text = "Reachy Mini has not returned MiniCPM OCR text yet."

    if result.get("found"):
        match_text = result.get("matched_name") or result.get("medicine_name") or result.get("generic_name") or ""
        explanation = result.get("common_uses") or ""
        warning = result.get("safety_warning") or ""
    else:
        match_text = ""
        explanation = result.get("message") or "N/A"
        warning = "N/A"

    return (
        gr.update(value=ocr_text),
        gr.update(value=reachy_result_match_html(result)),
        gr.update(value=explanation),
        gr.update(value=warning),
        gr.update(value=source_url_html(result.get("source_url") or "")),
        match_text,
    )


def likely_match_text(match: dict) -> str:
    row = match["row"]
    if row is None:
        return '<p class="match-na">No likely match found.</p>'

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


def match_not_found_html(language: str) -> str:
    message = manual_entry_prompt(language).split("\n", 1)[-1]
    return f'<p class="match-na">{message}</p>'


def tiny_aya_unavailable_message(language: str) -> str:
    messages = {
        "English": "MediLens is not working. Check AI model status.",
        "French": "MediLens ne fonctionne pas. Verifiez l'etat du modele d'IA.",
        "German": "MediLens funktioniert nicht. Pruefen Sie den Status des KI-Modells.",
        "Italian": "MediLens non funziona. Controlla lo stato del modello IA.",
        "Romanian": "MediLens nu functioneaza. Verifica starea modelului AI.",
        "Spanish": "MediLens no funciona. Compruebe el estado del modelo de IA.",
    }
    return messages.get(language, messages["English"])


def tiny_aya_unavailable_html(language: str) -> str:
    return f'<p class="match-na">{tiny_aya_unavailable_message(language)}</p>'


def manual_label_help_update(language: str, visible: bool = True):
    return gr.update(visible=False)


def hide_manual_label_help():
    return gr.update(visible=False)


def show_manual_label_help(language: str):
    # Visible acts only as a highlight flag (CSS); no extra text is shown.
    return gr.update(value="", visible=True)


def manual_label_optional_placeholder(language: str):
    return gr.update(placeholder=UI_LABELS[language]["manual_placeholder"])


def manual_label_required_placeholder(language: str):
    prompt = UI_LABELS[language]["manual_help"]
    return gr.update(placeholder=prompt)


def clear_manual_prompt_after_typing(text):
    next_response_version()
    if (text or "").strip():
        # User is typing a label: drop the old image and clear any shown answer.
        return hide_manual_label_help(), "", None, "", "", "", ""
    return hide_manual_label_help(), "", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()


def begin_read_attempt(image_timeout_seconds):
    total_seconds = normalize_image_timeout_seconds(image_timeout_seconds)
    return (
        gr.update(interactive=False),
        progress_bar_html(0, total_seconds, "Processing image..."),
    )


def enable_read_button():
    return gr.update(interactive=True)


def enable_read_button_and_clear_progress():
    return gr.update(interactive=True), hide_progress_bar()


def begin_model_start_for_image(image, vision_ocr_url: str, model_url: str):
    if image is None:
        return gr.update(interactive=True), gr.update(), hide_progress_bar()
    if is_port_reachable(vision_ocr_url) and is_port_reachable(model_url):
        return gr.update(interactive=True), check_local_model_servers(vision_ocr_url, model_url), hide_progress_bar()
    return (
        gr.update(interactive=False),
        "Starting/checking local AI models...",
        gr.update(value='<div class="single-progress-status">Loading AI models...</div>', visible=True),
    )


def begin_model_start_for_manual_text(text, vision_ocr_url: str, model_url: str):
    if not (text or "").strip():
        return gr.update(interactive=True), gr.update(), hide_progress_bar()
    if is_port_reachable(vision_ocr_url) and is_port_reachable(model_url):
        return gr.update(interactive=True), check_local_model_servers(vision_ocr_url, model_url), hide_progress_bar()
    return (
        gr.update(interactive=False),
        "Starting/checking local AI models...",
        gr.update(value='<div class="single-progress-status">Loading AI models...</div>', visible=True),
    )


def _wait_for_model_servers_if_starting(status: str, vision_ocr_url: str, model_url: str) -> str:
    hard_failure_markers = (
        "Could not find llama-server",
        "Model not downloaded",
        "Could not start llama-server",
    )
    if any(marker in status for marker in hard_failure_markers):
        return status
    if "Starting in the background" not in status:
        return status

    vision_ocr_url = normalize_local_url(vision_ocr_url, DEFAULT_VISION_OCR_URL)
    model_url = normalize_local_url(model_url, DEFAULT_MODEL_URL)
    deadline = time.monotonic() + max(0, MODEL_START_WAIT_SECONDS)
    while time.monotonic() < deadline:
        if is_port_reachable(vision_ocr_url) and is_port_reachable(model_url):
            return f"{status}\n\nReady: both local AI model servers are reachable."
        time.sleep(2)

    return (
        f"{status}\n\nStill loading: the local AI model servers are not reachable yet. "
        "Wait a little longer, then press Search again or use Check/retry local model servers."
    )


def start_local_model_servers_for_image(image, vision_ocr_url: str, model_url: str):
    if image is None:
        return gr.update()
    status = start_local_model_servers(vision_ocr_url, model_url)
    return _wait_for_model_servers_if_starting(status, vision_ocr_url, model_url)


def start_local_model_servers_for_manual_text(text, vision_ocr_url: str, model_url: str):
    if not (text or "").strip():
        return gr.update()
    status = start_local_model_servers(vision_ocr_url, model_url)
    return _wait_for_model_servers_if_starting(status, vision_ocr_url, model_url)


def update_ui_language(language: str):
    labels = UI_LABELS[language]
    return (
        gr.update(value=f"# MediLens\n{labels['intro']}"),
        gr.update(label=labels["image_label"], show_label=True),
        gr.update(value=labels["image_help"]),
        gr.update(label=labels["manual_label"], placeholder=labels["manual_placeholder"]),
        gr.update(value=labels["read_button"]),
        gr.update(),
        gr.update(label=labels["explanation"]),
        gr.update(label=labels["warning"]),
        gr.update(),
        manual_label_help_update(language, visible=False),
        gr.update(value=labels["model_note"]),
        gr.update(label=labels["glossary_title"]),
        gr.update(value=f'<div id="glossary-note">{labels["glossary_note"]}</div>'),
        gr.update(value=language, label=labels["glossary_language"]),
        gr.update(label=labels["bad_phrase"]),
        gr.update(label=labels["preferred_phrase"]),
        gr.update(value=labels["glossary_submit"]),
        gr.update(label=labels["glossary_table"]),
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


def apply_browser_language(browser_language: str, browser_device: str = DEVICE_DESKTOP_OR_LAPTOP):
    language = browser_language_to_supported(browser_language)
    device = normalize_browser_device(browser_device)
    return (
        gr.update(value=language),
        *update_ui_language(language),
        gr.update(value=device),
        gr.update(value=default_orientation_for_device(device)),
    )


def _process_is_running(process) -> bool:
    return process is not None and process.poll() is None


def _background_process(command: list[str], log_name=None):
    kwargs = {
        "cwd": str(PROJECT_ROOT),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }
    if log_name:
        REACHY_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = REACHY_LOG_DIR / log_name
        log_file = log_path.open("a", encoding="utf-8")
        kwargs["stdout"] = log_file
        kwargs["stderr"] = subprocess.STDOUT
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(command, **kwargs)


def _stop_background_process(process, name: str) -> str:
    if not _process_is_running(process):
        return f"{name}: already stopped."
    try:
        process.terminate()
        process.wait(timeout=8)
        return f"{name}: stopped."
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
        return f"{name}: forced to stop."
    except OSError as error:
        return f"{name}: could not stop ({error})."


def reachy_runtime_status() -> tuple:
    with REACHY_PROCESS_LOCK:
        running = _process_is_running(REACHY_APP_PROCESS)
    if running:
        status = (
            "MediLens on Reachy Mini is running.\n"
            f"Robot target: {REACHY_HOST}:{REACHY_PORT}\n"
            'Say "Reachy stop", or press Stop Reachy Mini MediLens.'
        )
        return gr.update(value="Stop Reachy Mini MediLens", variant="stop"), status
    status = (
        "Reachy Mini is not connected yet.\n"
        f"{reachy_start_reminder()}"
    )
    return gr.update(value="Start Reachy Mini MediLens", variant="secondary"), status


def reachy_start_reminder(prefix: str = "Before starting") -> str:
    return (
        f"{prefix}, turn on Reachy Mini at {REACHY_HOST}:{REACHY_PORT} "
        "and make sure all robot applications are off. "
        "If Windows Firewall asks for permission, allow the connection so the laptop can communicate with Reachy Mini."
    )


def begin_reachy_toggle(reachy_is_running: bool) -> tuple:
    if reachy_is_running:
        return (
            gr.update(value="Stopping Reachy Mini MediLens...", interactive=False),
            "Stopping Reachy Mini MediLens...",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    return (
        gr.update(value="Starting Reachy Mini MediLens...", interactive=False),
        (
            "Starting Reachy Mini MediLens...\n"
            f"{reachy_start_reminder()}\n\n"
            f"This can take a moment while the local service, robot connection, {REACHY_MIC} microphone, "
            "and voice models initialize."
        ),
        None,
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    )


def enable_reachy_button():
    return gr.update(interactive=True)


def start_reachy_mini_medilens(vision_ocr_url: str, model_url: str) -> tuple:
    global REACHY_SERVICE_PROCESS, REACHY_APP_PROCESS
    with REACHY_PROCESS_LOCK:
        if _process_is_running(REACHY_APP_PROCESS):
            status = (
                "MediLens on Reachy Mini is already running.\n"
                f"Robot target: {REACHY_HOST}:{REACHY_PORT}\n"
                'Say "Reachy stop", or press Stop Reachy Mini MediLens.'
            )
            return gr.update(value="Stop Reachy Mini MediLens", variant="stop"), status, True

        try:
            REACHY_LOG_DIR.mkdir(parents=True, exist_ok=True)
            REACHY_STATUS_FILE.write_text("Starting Reachy Mini MediLens...", encoding="utf-8")
            REACHY_STOP_FILE.unlink(missing_ok=True)
            REACHY_RESULT_FILE.unlink(missing_ok=True)
        except OSError:
            pass

        model_status = start_local_model_servers(vision_ocr_url, model_url)
        messages = [
            reachy_start_reminder(),
            f"Logs: {REACHY_LOG_DIR / 'medilens_robot_service.log'} and {REACHY_LOG_DIR / 'reachy_mini_app.log'}",
            model_status,
        ]

        if is_port_reachable(REACHY_SERVICE_URL):
            messages.append(f"MediLens robot service: already reachable at {REACHY_SERVICE_URL}.")
        else:
            service_command = [
                sys.executable,
                str(PROJECT_ROOT / "robot" / "medilens_robot_service.py"),
                "--host",
                REACHY_SERVICE_HOST,
                "--port",
                str(REACHY_SERVICE_PORT),
            ]
            try:
                REACHY_SERVICE_PROCESS = _background_process(service_command, "medilens_robot_service.log")
                messages.append(f"MediLens robot service: starting at {REACHY_SERVICE_URL}.")
            except OSError as error:
                messages.append(f"MediLens robot service: could not start ({error}).")
                return gr.update(value="Start Reachy Mini MediLens", variant="secondary"), "\n\n".join(messages), False

        reachy_command = [
            sys.executable,
            str(PROJECT_ROOT / "robot" / "reachy_mini_app.py"),
            "--use-reachy",
            "--listen",
            "--language",
            "auto",
            "--orientation-mode",
            ORIENTATION_NORMAL_FIRST,
            "--max-vision-attempts",
            "1",
            "--service-url",
            REACHY_SERVICE_URL,
            "--reachy-host",
            REACHY_HOST,
            "--reachy-port",
            str(REACHY_PORT),
            "--timeout",
            "90",
            "--mic",
            REACHY_MIC,
            "--show-result",
            "--status-file",
            str(REACHY_STATUS_FILE),
            "--result-file",
            str(REACHY_RESULT_FILE),
            "--stop-file",
            str(REACHY_STOP_FILE),
        ]
        try:
            REACHY_APP_PROCESS = _background_process(reachy_command, "reachy_mini_app.log")
            time.sleep(1)
            if not _process_is_running(REACHY_APP_PROCESS):
                exit_code = REACHY_APP_PROCESS.poll()
                REACHY_APP_PROCESS = None
                messages.append(
                    "Reachy Mini app stopped immediately. Check that Reachy Mini is turned on, "
                    f"reachable at {REACHY_HOST}:{REACHY_PORT}, and has all other applications off. "
                    f"Exit code: {exit_code}."
                )
                return gr.update(value="Start Reachy Mini MediLens", variant="secondary"), "\n\n".join(messages), False
            messages.append("Reachy Mini app: starting hands-free listening mode.")
            messages.append(f"Microphone: {REACHY_MIC}.")
            messages.append('When Reachy says "I am listening", ask about a medicine label.')
            return gr.update(value="Stop Reachy Mini MediLens", variant="stop"), "\n\n".join(messages), True
        except OSError as error:
            messages.append(f"Reachy Mini app: could not start ({error}).")
            return gr.update(value="Start Reachy Mini MediLens", variant="secondary"), "\n\n".join(messages), False


def stop_reachy_mini_medilens() -> tuple:
    global REACHY_SERVICE_PROCESS, REACHY_APP_PROCESS
    with REACHY_PROCESS_LOCK:
        if _process_is_running(REACHY_APP_PROCESS):
            try:
                REACHY_LOG_DIR.mkdir(parents=True, exist_ok=True)
                REACHY_STATUS_FILE.write_text(
                    "Stop requested from the desktop app. Waiting for Reachy Mini to say goodbye...",
                    encoding="utf-8",
                )
                REACHY_STOP_FILE.write_text("stop", encoding="utf-8")
            except OSError:
                pass

            try:
                REACHY_APP_PROCESS.wait(timeout=12)
                app_status = "Reachy Mini app: stopped after saying goodbye."
            except subprocess.TimeoutExpired:
                app_status = _stop_background_process(REACHY_APP_PROCESS, "Reachy Mini app")
                app_status = f"{app_status} Graceful stop timed out."
        else:
            app_status = _stop_background_process(REACHY_APP_PROCESS, "Reachy Mini app")

        service_status = _stop_background_process(REACHY_SERVICE_PROCESS, "MediLens robot service")
        REACHY_APP_PROCESS = None
        REACHY_SERVICE_PROCESS = None
    status = (
        f"{app_status}\n{service_status}\n\n"
        f"{reachy_start_reminder('Before starting again')}"
    )
    try:
        REACHY_STATUS_FILE.write_text("Reachy Mini MediLens stopped.", encoding="utf-8")
        REACHY_STOP_FILE.unlink(missing_ok=True)
    except OSError:
        pass
    return gr.update(value="Start Reachy Mini MediLens", variant="secondary"), status, False


def toggle_reachy_mini_medilens(reachy_is_running: bool, vision_ocr_url: str, model_url: str) -> tuple:
    if reachy_is_running:
        return stop_reachy_mini_medilens()
    return start_reachy_mini_medilens(vision_ocr_url, model_url)


def refresh_reachy_status(reachy_is_running: bool, current_status: str) -> tuple:
    if not reachy_is_running:
        return gr.update(), gr.update(), False, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    with REACHY_PROCESS_LOCK:
        running = _process_is_running(REACHY_APP_PROCESS)
    if not running:
        status = (
            "Reachy Mini MediLens is not running.\n"
            f"{reachy_start_reminder('Before starting again')}"
        )
        return (
            gr.update(value=status),
            gr.update(value="Start Reachy Mini MediLens", variant="secondary"),
            False,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    try:
        latest_status = REACHY_STATUS_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        latest_status = ""

    result_updates = reachy_result_updates()

    if not latest_status or latest_status == (current_status or "").strip():
        return gr.update(), gr.update(), reachy_is_running, *result_updates

    status = (
        "Reachy Mini MediLens is running.\n"
        f"{latest_status}\n\n"
        'Say "Reachy stop", or press Stop Reachy Mini MediLens.'
    )
    return gr.update(value=status), gr.update(), reachy_is_running, *result_updates


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
    ui_updates[5] = gr.update(value="")
    ui_updates[6] = gr.update(label=labels["explanation"], value="")
    ui_updates[7] = gr.update(label=labels["warning"], value="")
    ui_updates[8] = gr.update(value="")
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
    # Clear instantly; the (slow) server check runs in a chained .then() afterwards
    # so old answers don't linger in the boxes during upload.
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
        gr.update(),
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
    max_attempts_per_orientation,
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
                MATCH_NA_HTML,
                empty,
                "N/A",
                "",
                show_manual_label_help(language),
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
                max_attempts_per_orientation,
            )
            ocr_parts = []
            if orientation_note:
                ocr_parts.append(orientation_note)
            if vision_text:
                ocr_parts.append(f"MiniCPM-V 4.6:\n{vision_text}")
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
                    match_not_found_html(language),
                    "",
                    "",
                    "",
                    show_manual_label_help(language),
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
            server_status_output = f"MiniCPM-V 4.6: {server_status}\nTiny Aya: {'reachable' if is_port_reachable(model_url) else 'not reachable'}"
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
                    "MiniCPM-V 4.6 server was not reached, so Tesseract OCR was used instead.\n\n"
                    f"Background start attempt: {server_status}\n\n"
                    f"{orientation_note + chr(10) + chr(10) if orientation_note else ''}"
                    f"{tesseract_text}"
                )
                can_match_ocr = False
            else:
                message = (
                    "MiniCPM-V 4.6 server was not reached, and Tesseract OCR is not installed "
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
                        match_not_found_html(language),
                        "",
                        "",
                        "",
                        show_manual_label_help(language),
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
                MATCH_OCR_SETUP_HTML,
                message,
                "N/A",
                "",
                show_manual_label_help(language),
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
            match_not_found_html(language),
            "",
            "",
            "",
            show_manual_label_help(language),
            "",
            server_status_output,
            manual_label_required_placeholder(language),
        )
        return

    match = find_best_match(match_text, medicines)
    if manual_label:
        match = apply_manual_match_safety(match)

    if match["confidence"] == "low":
        likely = MATCH_NA_HTML
        common_uses = ""
        if manual_label:
            likely = f'<p class="match-na">{no_database_info_message(language)}</p>'
            manual_help = hide_manual_label_help()
            manual_label_update = manual_label_optional_placeholder(language)
        else:
            likely = match_not_found_html(language)
            manual_help = hide_manual_label_help()
            manual_label_update = manual_label_required_placeholder(language)
        medicine_warning = ""
        source_url = ""
    else:
        likely = likely_match_text(match)
        if use_ai_model and not is_port_reachable(model_url):
            yield complete_read_response(
                ocr_text,
                tiny_aya_unavailable_html(language),
                "",
                "",
                "",
                hide_manual_label_help(),
                match_text,
                server_status_output,
                manual_label_optional_placeholder(language),
            )
            return
        try:
            common_uses = make_explanation(match, language, ocr_text, use_ai_model, model_url)
        except TinyAyaUnavailableError:
            yield complete_read_response(
                ocr_text,
                tiny_aya_unavailable_html(language),
                "",
                "",
                "",
                hide_manual_label_help(),
                match_text,
                server_status_output,
                manual_label_optional_placeholder(language),
            )
            return
        medicine_warning = match["row"]["safety_warning"]
        source_url = match["row"].get("source_url", "")
        manual_help = hide_manual_label_help()
        manual_label_update = manual_label_optional_placeholder(language)

    if medicine_warning and IS_HOSTED and not use_ai_model and language != "English":
        # On the no-GPU hosted demo Tiny Aya cannot translate the free-text
        # medicine-specific warning, so show the fully translated general safety
        # advice instead of a mixed English/translated box. (Desktop is unaffected.)
        safety_warning = GENERAL_SAFETY_WARNING[language]
    elif medicine_warning:
        try:
            translated_warning = translate_medicine_warning(medicine_warning, language, use_ai_model, model_url)
        except TinyAyaUnavailableError:
            yield complete_read_response(
                ocr_text,
                tiny_aya_unavailable_html(language),
                "",
                "",
                "",
                hide_manual_label_help(),
                match_text,
                server_status_output,
                manual_label_optional_placeholder(language),
            )
            return
        safety_warning = translated_warning
    else:
        safety_warning = "" if match["confidence"] == "low" else GENERAL_SAFETY_WARNING[language]

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


CUSTOM_CSS = Path(__file__).with_name("static").joinpath("medilens.css").read_text(encoding="utf-8")

BRAND_HEADER_HTML = """
<div id="medilens-header">
  <img class="brand-logo" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJkAAACgCAYAAAAfKc7fAACOlUlEQVR4nO29Bbxc1bU//j0ybtddIjfuLkACJLgTaLFSChQrNShWnNLi7hqKFoJTXJNAjCTEPbnJdZ07c8fnyP5/1t7n3HsT4L3+3++9QEp3O8zNzBk7e52l3/VdEv79lgLAoD8YY1LesNlTo80t+5nx8CQYyeGAWQywQiDDAEgAsw6ne3tJ1s06pOc5tsfzdMN3PNf33/ay34uWad3LezwmWY/Z30nucwz9LHrCKQFSDFBb4PBskT3+ZXn5RZ+feOX8JY+fL2l7noMfw+p7lv4N1kwVmK/XHP7bYN3KxWdn2+pPB1ITJKdbKinKQ2V5MQoL81BYmA+P2wPV4YQkSSSMYMwkqeT/tk+LJJMMMr7liizDZEJQZFkIgmnawsKPBn8pg3Ucg0wPSOK9mCnem/7Jn7dk0P6bfxY/kD6Pvg89zp+AJMn837qeRSaVRGe4C+2d3WhobEdzSzOMTBxAYJ0zp/AfRWXljzRs+CgMnKwA8340gvbvsvil7i0ZfTiUsq2Qi9jU/Q9ld999P1uxfIUeDnfojDGDMWayf49lRrq6jJUrV+p33nk3m3bAoQxKEd3qPEUjT+h7Tn7o9e+iyRRAMpwlEy/Jtmy7a/DQAbjtluv0448/1rY3fOm6wbWPYZj8nrSI0GLiea44vvOM9DF1dHyPBRV/9BhG9l0mU/rW+9ifwe/6fKD4U7Ie3/0bkBYkzcjvZQmKokBVd5Mh8513/mn+8dLr1O1bd8Jd1P/qdNvKv/0YTOe/g5Dxk+gunXRaurXuhdmzJpkv/+MZ5OXly8LE6DBN1itY/EYekDBTtoj0yIe1oT0C18ct6xUZ6zW06cL3Ey/c06371j+YMI3cjPY5ZDeh6z1UmG7xmEzCJ/cKKC1ZkqGoMn9OVhQoiozW1lbz+BNOYUuWrlfcxQNPTzcvefGHFrR9Xci4c5Qz7Miq6Lav14wYNsC36MuPpUAgIGezWTgcDrGvll9lMhOmYULnAsfEv/n9HlK2x7I3n0TKEq++z/RZ4nP+y5PKet+wR6j6vN23Xmt9Lv+xstTnJnM/UZEV8XIGGIYOh9OBzvZWc8r0Waita42HKvcb2bX9rQbrk/o6kXtt7etCxq9QJTTs72Zs55kfffiOPnv2LDWb1aA6VEh77Hjf/aRFAkYCKATN/tsEjwHIsbcFtK8AWgJr/eNb0WTvob2SI+2p5PqoLDvOoKP4n5ZJFDGDEChxT/qKBI0CAfEYfS963v68rGYglsjC5/di3sv/0M884zTVERryNy266WoAKnkM+AHWvixkfO9GHXVh7tr3Xtk+a/b0nE8+ehvprC5ta4yjPM8Nn0dEZmRGhOmzVYWIGncTwD6mcTelZplU23ySQPY+1Rsl7i5b31ZN0m7f3Bao3md6/DErqUIXSG+g+93bRN8lmTHQFc8iltCwYWc3NNNARaEPBX7FPPHYI6RNG2q3nvfI+yMfP38iCdj3q+v/w0XSvS+bSmPnqhXTwdK5Jx53JJkCeUdjFN1xHf0KXTB0esiEplH6gHwY8muEJiB/hmsEnqaw0w4S31xbV9m+E+kQ7uzTCVPEY7v7cLaA9qY8bPUl7fGlrU/5lgNHrxLmmFIfvQEBfS8y8YbJkDVMJFIGInENkXgW8ZSGaCKLRMbA2AH5KMlzY9HaZkRjaRwydYh80EEz2aZ1Gwe9ceelIwCs+qF8s31ZyPgupNLpUXTdjxg+hAtZNJ6B3+2Ew6Fw08e3jTbepLNL/7HPse1IyzxKI8eZvyEXDGuDyXRS7qxv5Gk7+73v0iOAu0ned2k4SfxJvmDfw4U5tqNchqwOpLImMpqBVMZAVmcwTBO6biKrm2jtTPF78V50VZgwmImcgAsORUIqoyFjMAwbOph+rBqLZ4YLIZspAfOxt9e+LGR8mamIQ3K4pLz8POFDmYDbKZKltpnDHqaJ64yeDTfATBmSlQ0QWkhoFZJJTdN2S3PsFnD2WEPpv0xa4L/03frUEywfyzQlRBMaFyr7s5l1wXicCjxuBUgLASMBNA3GtbUiMSiy8O4pCCgpKeFbbETrHOIT9r6A7fNCxsMlXWdOhxMetxsZHTxy7MmWf2f8J/7LWG+Gnms0y1T1Hsesx/t8mCUr9mMisNg93uxx03aTKdbHdvbxwfoILj+Km1uJayNVYsiSYNnpEx6kiOMoBUO/07Lm3ITbgQL9rqymIZ3R4XK5xLub2g/ii/1bCFmv7ekt35DvQjcrd2H5O33Wd2RcKYemKMKs2mUlYcHsyJLe6zuceO7OWUJjK6s9pJr1SUH0PNbzn95H7PexpdSh9JacRPVKCD0FMfbvpOqW8DXFvW3O7XPAS2VW8gY/4OrJhu+zy+EP2RvTR4FZQsL/Epu8R8TIRc/yt0xTh2EYvclPS9MYptGTY7M1mG0ev/VXXzn6vr+txb+P7QDutnpVm0rCZP8mO7XB66biJtIv4nmRfqG/haRJssIDHPFeJIUeUmk/2Nr3hUxPp2z/i5Z90vnfPQeJ6LKnjNNH5uw8rK5r0DUNpmHwm6Zl+b/5S/r4cOKBHgn+7tU38mS7l476fIU9BK1Xl9EiQdrN4FvvwbVW3yrDd3ysLZAU1NAyA1VD8AOufd9cuny5MLt220fyX3r+/p5K4p5H0Gt0s2+usrcE1OM+2Rts/21rTdu3s4W35837OnDWH3smd/fMZliLFJEdbYqX9X773mqTJYhWctZ+ki448tnotruk/jBrX9Zk4tJ351RY17h4cA/LiP/GivX1l3bXJX1gOvZ/enOv1mdZ5sny1/pqEvsdYZnk/zINamme3b5hH9Pf6woISSZTuPvb2aa7x0OEwQDNyhP+UEnYfwchE4uZ2m7q4DsSoN96yR7/6hW4Pf/4nvC052alSfqoC1to7UQudnvz3qCj5xU9fuN3mT8rx9f3e1uOWA8kzUoM9z1O+JuATH4d/x7yD2qx9n1z2Sfx0JOe2P3p3mTp7g9ba480RY+Q9Pp4u+fmdzeZva/7rm9lCdRuybpegewpEvSkL4TA765dd0/W8iNs+eUvsWquVmKWIkoOZ+IhKR2oQtKizfgB1z6tySx/RIVVeunzqJWG6JN23y313ru+lQfriVRFGuP77ExfIe01pX1zbd+XnpX2+Ms2lbYQ7p6v4+9gJ4Ot7ydwZeJxkiX+POUHCcIkHuDPa5rO8x9qoqkRP+Dap4WMWx8t3mgwyg0ZfKsMoze31WOy+jpUljn9Vkqrj4D16sQ+nlLvLu+Wxf+ODJhVOO+Tf8C38ipi9XlxX7/N1lLibXoFrK9RppvZR6h4jZNRxl/mkSlVBkCeBDKATFm3H27tw0JGdThAYUaboRksEU9wE0cRVTpLWf/veRmPvvZQbvzxPSSPSkt9zadVFeh7+LciRf7Y7m+J7zXLu7mRvY9bn9ljFu1C+R6JPg5Fsh6TFcqdCUdfVWU4FCFoiURMHO0L/aD7vA8LWRE/w66cwl0wItL7H3woeRwSFMlEMq1BMxgcqmjAELiwb5u+Xt/KdsD3FAtSe5a5+5bQ7p7+7wkodw8Q0XtYnzf4jgvANqp7Fud3T4cIE27j4GwVR+YzkdLR1JHgyX2nqsDnkvH+e+/JgNP05JavtN7kP6DF/5+L70Zo/wtzYl+/vjnkkwu++vJTVtpvmPzN+l3IC7pQnOuGz63wk94T1fUInL2tFlas5x2t7FNfefvvEgB982e75TJ2V1W8trinj7YbCFK8nrRQOJZFU2caCiVlrQiB7j1uB2qbYuiIJHlVgMpHZDFTWR3xZIajeaZPrEHrpi/0n885TjWVvAVmtn0mRwH8R8j+R4vjo5yF407MdjW+1q+qhM198j6zZtQUuSWclSLdSSiSgVyfglDAAY9DgUOV4HKofIMoIU5I074IC1smegTRzk/ZB/R1s3YTrt0OsSBAVkL3W878nvGjLYDi3w5V4ULW0JHmR6qqxIWNYD+RmIa6lhiH9ihUGJcVyLICAwqHXnuVLGtc/5nxl+tvVjsiCaaEKmfo7Su+/CFx/j9oJvh/afGTp+aN/K0ej94NM6MeevB47D9ztt5/yCipuLRcKSwsgCy7EE8bHMhI/ovHpcDDIUEmXA4JLlWgZ2ULLsM7gqhOaHUJ2WbTLkjbf++5+gqZncNC31yX9bftzJNf1WsCya8ykNVMJDIM4bjGoTyEjdMMIBLTEU0aUFQVigK4XTIUaAh3drLWxnpz15ZVbPlXn6hfL1kEOPIzcqjfL82OpS//0D2Y/w5C1iNo3qoDxqejbX81o7tmAYbD5StESXExCguCCAbcKCotR1FJBXzBfBQVlyAUzEEwNx+q0wO32w23xy0AjDJlRUhLAE5ypB0qlwoSPK4BuXSYPNnJa4k2zp7QD33SCuQLmlYqhfcRkJ/FD6RGYeubSzJ0QwgWz3OZDPEkyYOBkI/en6ErGkO4K4pYrBumlkRt7XbU7dqJro52dIebUV+7BZ3hBAxqipddScXvfU0OltyuNSxeZ/ndP4gv9u8mZOgxB5KMvOEHD0t0ds3QDMcRZlfTsU63CZ/fKSWi7cimkgCcfHNlpOH3OiE7XPD6QgiGChDKCcLvc0NWnfAF8hAKheDxeeB2+eB0OnmDisvphtfjgtPpEA0rksKL0aLdjvoyBbxbN0RHFAmlaSM6mIFsVoeW1ZBOp5FKJaDrdJ9CR0cnEvEoYGiIx1NIJJPQyESGW5CItSOlKTAMGTC6IclOeIIlLBlL0c+OSt78dz3B4vmO/NJPomvm1e52Tn7g9e8kZOhDJMH1RM5Bf6mOfn77lqmHHus49NQrpHhXM5iegoQstHQSne0tiITbwfQk9EwKse44ksk4UokwTD0LQ9eRSqaRzaa5uaJ/G3qyJ4dlEGrDNoeM9pJUGMHbBAKi15dTRFKC/CdF5khWBgWy6rAElISSweNxwuXxwOH0wu3Nh9sbhOLwwOPzIq8gF8HcUkhOH2SnF95gMdwqjKdvvUiJ6573WWzzkTB75ImEi/3QGuzfqKy027JO6nAnsMHUGzckJKdDa6nb7mzc+CUrLa+WSsoreCmPMPRVA0dDUR3cRNKNHGnSOGT9HA4ZEtNAG0dAwe5YHMlUBtRHIrx5EzIzkc4QCjULMEO0r3GkKkWzAlio8GK2zD+PrgH6HNJwZGJlRYVDdXDIhQQFLpeTP0bmlnNvGAwZjVIZEpxOlWfwY/EUT7KSxtuy8nOWSVNEKa8TAlbsAy5IATf+4Nrr31mTSXYXE2QVnsqpx6fa615AMuwBEph15AnSIcefi1BuMcfuB/1u5AXdPb2Lfq8LbpfCGzHSWYP3nHAcfVZDdyLDk7wejwsetwsmqNWOBMaErhlQHQpcTocVFFAPpI5kKsuxafS3SJQq3OeidANFuFRnJG3H/T5F5k4/tbSRZqQeSvLTeHSpCNMbT2bR2rwTzVuXYuWiT9DSsMWAlCe78yvecg099ezol1d19Unp/ii02L+bkPU4uP7KyQfEo4m/oXvH/v6AH8cefwKmzzoS/vyBCEeTSKYzXCDcLge8bgdyg26EfC4EvCqCPifH2JPg+TxuuN0uDpshk0aNGtlsCulkgmu2lrZ2pFMpLkBcWJwOOFQV9JkBvx8qmTp/kJtGKq06FQnxRArhSIILHrWyUVrChoCnuWCJG3UqkaDbcHE6LpXS4PU4MKAyFy4pjY0rv8BrL/+DrV+7kySxwZ1XdG+6dfXdVnX/B2vm/XcVMu7ghkYdlZuo336HHmk7x+VV8IvT55i/vvAiuXrAUGQyOmLdMUTiab6JHqcKnTEk0zpvLyOzlyXtxWSoDheCfhdkZBHpbMWu7ZtQu20zWpoaULtzF+rrdqE7Qn5bwvp4h3XTrJsMb6AQxcUFqBk4EDU1A9CvZjAqqgZB8RXD68vh5rKtsxuGlhFZNAt4aAsZ+XriInDyY1VZQsDnQllBAEGfgzeJeLwe6OluvD7vZXbXHbdLO3fWQQ5ULHAHS3+XbPxqtU2l9QPvzb+DkIkT6amcNDHd0fQ8S0WGHH7Eoea1N1zDJk4ar2QyQDIR52aPOnhIg1Dei0pOTis5qygOuL0e3ryxY2cTPvz0Cyz+ahGWL1uCbRtXWwrBCThCqKgoRUVpPirKC5GfV4C8gjx4PH7IkgJNS6OzswPRSBSt7WHU7qxDY1MLkt2tlh/uhC+nGOMnTMDEqfujsmYkRgwdjFAogEwmDZmR/6dAM0WSmMwklcfIR3OqEtz0fR0K3E4FAa+Dm1unwwGX14Wmxlb26IP3Gfc98JRqMDXlza/8fbJ56RN7BkM/xNrXhYybBE/p2ONSHS0vepy69847btYvuPB8lc5oIi5MGfkzZI503rtIiVhyuGV4PG7uDxGef/4X8/HyK6/j3fc/RXPjVr431f2HYPq0SZg6ZTJGjR6JmgH9UFxcBKfL8699O6ajszOMuvoGrF6zDt98sxaLFi/DN6vWwsh2QVL8mLH/ATjllONw3HHHorS0lGtVPZuFqqrcfIajacSSOpJ0gegiKKH0CC0y9YV26czp5MnlTz/7wjj3179Vdu5ohqew4qZUx7rrwYz/sPr8z5bIYnvKphybatvxeklhUHn5H3ONGTMOUDKUh9J0q+ta+DyUUHU5VStyFL2ubW2teP75l/D4k89gM9dYToyfOAVzTjgaRxxxCEaOGAaH093zieQiUX0wnc4KrrOeJKtARfDqAE/YkqZUuHPvIb+OVxZ6165du/D551/gn+9+hHff/xzpRDMCwWKcftpJOO+8X2PcuDH8OEqRKLzuSkGBie5EFm1dKbSFk1zgRLlK4sJWVuBBca6L5+9qd+5iPzv5dGP5ii2qO7/kmnTH2r/+h3Dlf+jke6tnjk21bl1SmONyfvzh22z06JFyIpnqadzl6SzLISd/xyaNa25qwiOPPo4HHp6LSGcdSssG4Be/OAVnnnEKRowc1fMh3QkN0ViCSFy4NuwpXNqYfKsAvieEh4e4Vi5NksEFzi5R+bxuHtXaq6mpCW+99Q4ee+xxrF69EpBzcO65Z+GSP5yPocOGIhbP8E4qSosQjId+A5GsNLQlEYml+eeksgaSacYj5ZpyL4rzfejo7GSHH36ctmLlJqe7eMAZ6eavX/ihkrP7ornkexgadVowvv2rrx0sPvDTj982pu83XUkkUoK3ixe/e9D2PXxe0WgEjz72NG697T5EwnUYPXYKLrv0dzjhuKPhCwT50R3RNLoTaa6p7Lqj3US7B6SrF2XRAzTspX2y0auyLHGhIq3GedEs+A4lX0mj8pIV2XxNw8effIL7H5qLD959F5LqxKV/vAhXXP5HFBQUoDPcLVrcyJ90yFxYw90ZbKmLIEWt8wwCRwcJQ6uDqKkIoba2lu23/yGstSORDJQMGh2tW7jzh0hvSPtwQfwJPbz93Pvuv0v/3W8vVJPJFM+oU0KVHyQT4YrRYxpff/0N/Omy61C7Yx2Gj5yCq674HU495SQoqhNpnYQriQxFl1STZJQFEyiMHmxEHz4Mi/pnNyQGLZtuk4SLay6LGdHvccHntfprewrukvATNdKSOk+VkNNP68MPP8R11/8Ny5YuQL8B43Dv3Tdwny2WoOpDlgubEF4PUhkTKze1cYYfChQyuolkhmFoVQCja/Lx5pvv6HNOOlOVPQXv6vHtRwNsr2uzfU3I+N7nDzlscOeWZesOPGiq/Pmn78qZrMa1BM+0E77d0mIkYE3NzfjTZVfjpRfmIpBTgZuuvxwXXXAunG4PklkTXYks97U4oQmXIo5p5uUhu3+zB05trR4SFLvdzaYJ6EFpCDoBYSLBBY9MJkWGNpkdb/bgr5fgdKj8OUGTIDQxEaY8Pfd5XHb5dYh21eOii36Pm2++Af5gCF1dUSiKyj/U53FyqM+KTW2IxjI8FULIjUTawITBuRhSnYuTfn6m+dor78iB8iFTY41Ll+5tsyntiwTESs7QuSxWf9aCLz7Q99t/PzUaS4qykEWJTsJFG/Xmm2/hvAv+iPbWnfjlWefh5r9cjYqKSp4i6Epo/N6mSadGFBI0vriACew8iRgHBlqmzvbzbFNIQmMLEvqwNtIizaIqAjLEsfeE/7J+iC2IZC7dLuduP5JMNV0wdLHU1e3CJZdcgddeexnDhk/FI4/ciQNm7IfOziinu6KeBjLHpAW/XN2CRIrqqVQxENH08TMHYt3qFfq0/Q5XTWfOO2Zix7Fg7AdHZvxYF9+fobN+kw8pLzJz1tFElW7GkhnW0ZVgsUSKRWMJzj2u6zq76qrruIQUFJSzefNe7eElDyc01powWEvMYC3dOmuIZFhdOM12dSbZrvYk29UWZztbY2x7U5RtbehiW+rDbHtjF2tq72aRWJJlsxozTYOZhiHuTYMZRu+t9zGdaZrGkqkM64rGWVtHlLV2RFlbZ5R1dsVYNBpn8XiS6ZrOTNNkpmGKe+tvwzD5Z9nriSfmMkXNY5IcYk899Sx/rD0cYy1tEdbQEmFZTWetnUn2+uc72Jtf7GCvfraNzX13E/tydTOnY99vxiEm5IJk5fSzy/qez72x9qEC+UyFkq51m1YcBaaHfnn6SZQhVbu6U3A7VV4nzM8NorOjA2edczH++fbLOPSwY/HkEw+isrKSR2ApQlJICmQydbLg9eIkJpLwvkyJtJYgMyHIdtDrQsAj6pFpzURXXENtWzc6o1mezyJkBgEhkxmTH0MNQj6XBK9LhkuRkOOnPJaHR305QXGqU+ksTwqTqeTu4278Cr2VR15oVxUYlIKRwCPO0aOH4xdnXoBzzrkIW7fvwpVXXcrfi7RxWziB8qIgqop9qG3q5iae3n9bfRemjSqRfnbSccZXC5Z6undt3A/APPt87o2d24eEbD4nFUsGBh3u8Tkx44D9oPP8FDnQBoJBP3Zs34oT5/wM69auwhVXXoe//fV6EVWmdWhMMClyp67HkSI0LJlBgNrqyPzleGSEPCKf1hnLYunWKDa2ZNCdpsK2iWRKg89pIuA0oBE8KB7nn0/ZEZdMSVE3taBBZyoMU4ZTjcIhmygMKBhQFkBVSQBFuV4uUek0Ed3pPJHaE0zQ6tMGwLvAqVilaZg8eTIWLngfv/jlxbj1b9di184duO+B+/k5oNxdPJHB4OpcDs+mSJOStinNRGuXhlkHzWCy04fuWKclZHuPEG8fEjIuC5KUCI8YMHIgqvtVy9ShQxucU5CL2m1bcfTRx2LHtk14+JHHcOEF5/HcViRlwCTtRQ0Z/B1EWoL4KzjRML2zJCPfJ8PvUhBPa1i+I4q1jVmEU6IUxbQYUl2N2LlpLdp2bUGkpQGRtjp0dzYhldS4s84jPuhwe51w+4LILemHqpqhqB40CgUl1YilitESk/HVxhhyPQyjBuZgaFUOr0/2/jjrl+7ZxMKppFQeiRYVFePtN5/DOef68MLzcznD4uNPPAhCydW3dKOmXwEv+O9sjsHpVLjwN3XEMLCiQios8KG1PTHEYs7eaz7ZviJkXDSGzb44D8ysrBlYA1V1oLG+HYbkxrr1G/Hzk07Bju2NeO75l3DG6afw3FGCUKQkANyXFwImfHJJtPUDyPUo8DokrrXeW9OJjS06h+0gGcauDcuwcekn2LHmG7TWbYJhB2TuQiiBANy51XBVFUJyuXm6RE9GEU1G0NEdxq6vF2HVkk/44Q44UD5gAMZPPxQjJx+MeKgK7TET32zuxODKAMYMKkTA5+RfjnK+PazpPb9cmFXC9mu6DofDheefnYtAIIRHH7mfv+Zvt9/OXQKCFpUWCJNpj/5qD8cxsl8JCvIDaG0KF5jMlCWbh2EvrH1KyDraG/MBPae8vIA/FkmZSEQb8JtfnYwdWzfhhRf/gdNOPQmJjIaUQUMU7IkhlvbqwyhNPlPIJaMzlsGna6PYGXcgaTrR0bQaGz5/Devmf4S2FoFidg2YgMJjz0fBsDHwlw2C5CmAovrhdnuF38SADDFUGwyqTMCeLDKJCNd+3fUbEVu3GA1rv8TO5x/A688/gQFDR2P/Q4/BAbOPRV1YwvqPd2Bk/xAmjSjmtVROqPytLhUhExSlkumkyPSRh+/jgMjHHrkXAb8bd9xzD+85JR+VImLdFHk/4vZXnU4pLzcHwLaSn83j+57t23r6f7n2FSHjy53YRR2GzEnQCQA+B8NlV16CrZs34MmnnxECltaQYpRD6m3zF/krqafJN88jg97h6x1RfLQlgzSlJ1q/xtK3n8GCt+dBYzpc1SMx+OybUTDmQEiFA6E4fHCQ75ZNA1oWToXBYWjQs2lu5tz0gZTS4PrDAcVbDFegAjkD94dj9rkwU11o27oc4WUfoW7Rx3j2/mvx8bwncfzp52H8/kfhmx0pbG/cgQPGFqN/WYD/Xp4YJtNm07zbbSiyhAyhcQE88uBd6OxoxbPPPIqamkH41fm/Q2NHN+9msmcOZHTCpgm3QNB/r9+r+7ZPCZlYbsiK8GNuuvZyLFrwAe685yGc86szkcxYAkarD2kc/UXaxiEzLmCN4RReX9kBwxmAV2vBZ0/fic/efI4fWzTtRJQe/kuY/SbB5w7AJxmAloaZ6OJQoACVdPwOuCh5SqBpmbSX3T4nfCuKRAlBYRgZmGaa+kJgOrxQhs9CxdjZmH52B5qXvIsVL96HR+68GlWvPIM5v7oY4/Y7Fp9/046djd3Yf1wpR4qQoNjkKjZtgT2pJJlKw+f14O/PPImWti5cd821yC2uxOQZR8KhhLmWI22WzhjoJv+VC524QPfm2seEjEozKohO/aGHHsbrLz+Hs889H5f+4SJexE4SosU6hYKKSWThKQolvyvokrFqVwxf7dKQMBSseONBfPzk3YjGu5A/+VAUHvcHlAyfAQUq8pFANBWD1+tAUb4bXocXHlVCwCmjyCNz7yylAy6VwaeSBhNyTVFqQjOR0kVGP6OZiGcZR1EkMlnE0iYUVz4GHX0Rcg+Yg7rPX8KGuXfinht/j/0P+hS//M2V2NhcjKbIThw1rQI5fifXaDyzQUxDNmcZn6XEQA3M+bkBvPT8k5g8ZX9ce/lv8c4HI5Cbk4O2jgjHo9GLCbQpFCEl+2l2xN5b+5SQtXd3A4oDH3+6AEsWfY3xE6bg/nvv4rCXmC7cXCoNCXpNkWyikxx0SfA5gFe/bsfaiAMlrBVv3/5nLP7sTTjLqjHlqsdQNP5YJFMmHJkYOnUJlx8Qwq64Dx/W66jMVTGtWEGxR4JXlRDLMrQmSYjIRxK6QYBuLDIXJvOmXXo+oTF0pQ2kaeADc6IzbSKaMZFKRRCUvRh9/KWYOHsO1vz9Dnz58lysXbkW5112FSYdeCzeWdSI6cPzMbAi2GP67P5NW8NRHq0j3I2KinI888zjOOzQw/Hnyy/DH256EPEMTWMxODCT6qS80v8DFHn2KSHLpNOS4lQx//OlCAT9eP65Z+Dz+dCR0Hljh5hg0zt9jTRYjlviBe+5X3agxfQjufUjXHfjRehoa8GIn/8eRSddCpe7GM5UHLkuGV2qB8O8CvYrVTFaAypzVJR7ZZR7AJVgOzIQcpKjL0NnIiHLI0JZBsHG6CaqWxJ0Rn4jkO+WoVnT6LqzDE0pAynNwXNYmUwU3oISVFx1H4rHH4zP7roWd1xzKU49ewMOP+V3eG9JMw4aq2NkTZ7VZS5I7uwmFBI0SrOEu+I49JBDcONfbsH1116BoqcewAHHXYhMNilMOyOzbm/3f3yy71sSsgmNGnENswu33XILhg0bKnwNS8D6HMn9D9JgKhj+vqwTDZoXLUtexNzrfgNdUTDphmcw+OBfIttFOPsoTKcbkkfFgfkq5gxQ4VEBn8IwIU9GbRwIa4BfYejWgF0xE60p0kgGqA/E9nT8KrhZdslC4/kUQetEZtbL0yYS8tzg5pbqmj4VaEqYqI9SAJHClENPROXoSfj01kvx0tP3om7nNvz68ruwaEMXF7DRg/N5XqwHKMlrrwIpQr0LTpcD1/z5T/joow/xxvMPYuSE/ZBfMZy39vHKwA+EwN5XNBkHLvhKhvZL1G2WZh92hHn+eWfL1M2Toky+1YZmazFKwvqdEtwy8MryLujeHDR8eD+e+stl8BVX4ICbXoR3wAEwujrhdavozjqQ75Vx/GAXjigX8SGlOQh7mu9k2KgDz28x4JQFMoMmfigkMpLEUyEuGXDx0r3ES1fU051SZcRUi5CO8Pk0ZUQ2+bHlfgUe6vFkDAUeBTUhGbURFQ0ENwoVY+rVj0EKevDVO89zKPbvr38A89d08mL78IF5yGSzFg2CPRhW/O72zjj6VebjgfvuxOSpM/DWC4/ijzc9iGxaRkd3FoRWEf0Ke9cn21f4ydjMmder6fbaO31+Wbr/3ls4vCVmURL3ZT0kZ5s2nrTKK0vbEVP92Pz+E1zAcgcOw5Q730Wgejp8yQ5kJYUnbE8e4sENU704sFjitJhk1myiFdrHKQXA8dUy+vlMnjZRcwNwlITgLQsipzyAgooACsoDCJX54S3xI7/cj8oyL4r8DoQUEw4JSOom2pImIlmTaz0ysdx/MwGvImFskQNDClxwmhk4NWD6pY9h0i8vxdIvP8L9f/kd8nMULmjNnUkOWKSWOfKzxOQRAawk1EVbRzfGjRuHSy+9FKuWfYJVX30A1eVFJJayosu9D77YFzQZqSpj2c6F5xnJtgkXX3GJMWzoUCXCzaTS04/fS8AJ5LolfLSmA1viToS/egEPXPN75NWMxX63vwrFXQk11YWM7EC+R8LpIzwYV6DCITF4VWvEzG6zMElbMRQ6AU/AA5fTAPvyUzhWfcWDBK+TOr5NxMkHpOjPMOHNz4V30mRoY6bCzPXBEc/A55ChO2hekrgFHBT19n4M+WyD8lREM260JGJIxZIYf/4tyOpZLHnhAeQ+eCPOvfwOfLaiFXNmVvIGY4os7SoGs4IQakLOCeq48vLf49lnX8ALTz2InH4ToZeUWpTryt52yX70QsbNZPHp3/ja3jr1uvLqGnb5ZZdImmEiZdjoid5IkgKoYr+M2tYEVneq0BuX4LFrL4YjvxAjr54LyVkJJdONJFMxLl/FuWO8XOt1ZYRwtqYYhgdMXsO0WGd5UvPjOg0fd6nIa96IkmvOQu6KZT1lKTJA3X0wzfQ6yuIRE3Cmuho1f70Lg38+B5lIFjq5lYbE/TgKHjgmzUqxmkxCWmMYV+JEyvTjo+1JpDq7Meb8O5COdOL9t55DZb/+OOiEi/DR4l049sCBSKSynLyFv56DBUT5rKktiv4V+bj+mstxwYXnYennb+LIn18MLZvlxfv/OP67L1JUeuS9U09hsabSK/56q5GXn690JjRq3rdmWYrNokiS/DBycD/anILfCOOBm/4AoqCYcNVcKHkjkOruRExx47hBHpw50oUkDVsF4/ffdJpoTJlY3sFw+gCRD6NgYm3YwDdJFTVqCuM/mododwplf/gtCqqK4cimYCSTMF0eyBWVkMorYebmw0xk0LJ8OZb87Va0nnESBuE5BE89A9GODLyqwtMZlEsjP82+kmwiZfInxxc7savbxMqmFJypNPa/7BEk21rxxH1/QUW/IcivmoxVm1oxenAhdjaEe3oQbLNJ6A6Cav/yrDNw97334qv3XsSxJ/wcqpPQH3ufo/jH7JNxF2v0IXf4st3hqyr6lbMzzzyFa7EMzafs0ylEf9F+BZ0S3l8dRkwH5t1/DeprN2PYxbcib+ThcCS7kJJdOH6IB78a5UJGF77bpi4TH9drCKc15DpMdGUZ3m+kcg5DR1LHJ80GPC6GypsuRGjXJkx55B4MO/5wFKYTyNmyGfmr1qLw6xXIX7oEeatXo4AZKDpgJkb/8VKcumwpckcOx2e/uQhtW3ZBJtYemaJeyqNZvQjWWB3eH8AFjfHc3tB8Bwp9DkgmXVBOHHD1w3DmlODxe2+GT01i8YYOjusP+F3QCOJreaV0YRAzI+XO3G4PLrv0UkTDTdi0cj78waA9cm6vrh+zkHF3a9vm909lenLgeef8wgyFcuRI2prm1of7lfD9VC6qbUugNubAjmVv4/N35qFg5nHInXUBWKQTMThwbI0b54xy8QQplYIWN+tY3EKNGYTvB9cw+S6gJWWgJcGwuUtH1OGE9M1CxF95Do5sF/xf/BP6s8/A3L4L+qYd0BpboO/YCf2LhdAffRzG7y6Gecs10Ot3ItC/P8bedD02RWOoe+EZFAck+BUTHqswEc6YiGtWv6YlcPSbKDAIOCX0y1WR73dByiTgLR6MiRffhIZdG/HuK48hPy8HS9e2oDDPz4vlfQdKUImL/C8Ccp7y8zkoKRuMN197FakY8bFQEXPDXt3IH7OQmdczJqdat52TV1jIzjv3lzxy1JnSM99ROL1CE1BK4cutCbBUC9548Ga48kpR8+vb4Uxp6NYZJpY4cO5oD6JpxhOmXzbrWBvW4KWqd880D4GWJXjjig4DrWmgMh/IX/YpQrKMvOIcYMFCSLsawCLUAS5DdqmQfG5IQT+kwjxIHjfw4XuQf38h2JoVqBo+FP2LCpH5/EMEuWmneqKYJ06rWzPRnjaR0Bl3/vlvoRqoQYlkGZUhFfkBF1yJCAYc+guUTj0cr7/0OOo2L0d9h4bWzgQK831WvqyXQZsEt72zmzeenHv2KRwLt3VrHWSvB/Pm7V3PX/4RazHzoUGzx7BMeOrxxx6C4pJSJZoSWsxux6BoisJywoRtboyhPSNhwauPoqWxFv1OvwKenBpkkklU5rhx5WQv30TKZy1qMbExosPvsGZE8my4yHVRRj/PJXEnnd67yAm4W5uhmSacO6htUSYMNdDcSs2SlJTjHGaUi5B0DSyVAlOdYBvWwnzjTSg5ITgLC9HY2IhkMg1dAAa5JlUlCU7SOibj5aampInGhImd3QYXProYXCpQ4lMQcMnwQcLY826CITvx/BP3cnaihataEPB5uDbjjS72yEJZ5oJMv+HMM0+HL6eQE8rI8u5NKz9lIeNnqquj9UxJdeCsM0/neXWKynpGPXDojvBBXLKJbxo1ZCINmP/6s/APHIPig38JvTsCXVXx+wleeBwyD6U3dTFsimoIOHrJzsm80Gb7VBkFbhklXgn5vN4pIaASqkJHhD41RpoyzQVLMnQRzhLDIt1bg+eh6ZAiUeDQmWANtZCWLASNbOignk5Nh9qn/OSUGRdqEm4aK05/251O9LxbBb8Qgi4g16vCoaVQPmISRsw5B6tXfIG6LV+jOWKiqS2O3KDX6nK3WvisPoFwJI5BgwZjxv7joCfbuK+5l3OxP0oh4w7/4b99z2XEOo8bOmwIpk2dIhPiQUCcrWwYA2/moChwW3MMzQkJy99/AbFYBIN/eQWCvhB/pzNGejG6gJpeGepiwOJWDR5FTCzhyC/F2mQVyHVJKHTL8HNUBeMbT/6TQQJt9fRKXHNZvZkkXLSxXNjEbEmJsuozp0OeOA4sNwgW6UYhcVVQ3ZNa4KjhxBIyEiqCH9Fjdl3Ufpy4zKgq4FOociFxn9PrluHSNUw49ULIrgAWvv8PFOS4sGlXF4IBDwc08iVyIlzjd8fT/KFTTjqO6hCi0B7Lk37iQnYyl6Nl7989Dkai/xGHzmSq0ynHM0T5ZB3C8xaifudxAOubMkhEm/DF68/DO2AMcsYdjVRXFEMKXThtqBOaLuiXFrfpUGTqRxRyQZEcN5MykOOSeYGcErIUAdJOkW/EBcuGC9r9AaTFbDNpIyHtRW/cHSOuJ0hEgizLKHI64IPJa4cctSFLloBJcBBHh9XpRI9xM0rPWcJIx5DAUbqjwKdCySbhLhuGigOPxteLPkbTzvXY1pxCLCkiTQ4LshxWuqeqAEXks2fPQii3HHpWN2dOGW38xIVsHhelaGfrEVCdOO6YI/kJMc1edAUJBwkNFb+T6Sw6s05s/vozhMOtqDrhN3C7AtAYcNwgF4/YSDhXdZpIGzrfPJFbIyESG0wbSNqLNBcJHE8DyFZNkjbeLixzDWaAkYBxrk9Lg9GiQ3Sd8gfAspVgr74D6culQEMdEvE4ElQNoIlt1hwAEmASJqHBhH9GAie0GVUZxL2t3dyqhAqfjHyvClU3MfT4c3k31MKP3+MCtqMhwk0m/0o951K4FV3RJMrKK6TxE8YTqje//qP786wDpJ+okFEQySQjET2irKwQ48ePlUhgOHTYCvEpcoqndH7i6zozaAp3Y9Gbc6EEclA+7SgYqSRGlrgwqUjlZnJHlGFzVOcFc7vfkTaW7mkzqYBtaw+e2LRQFZRHc1ubzU8UCVYfcylxMKHFl8E5+EXOX3K5gO0NQCoFKdoFIg/OEqmdLVwSYVbF3/xmYdJ6TCZ9pkIazBJAGm6hAnkUbQZVSJkkBo+ZiqJR+2PBx++BZSLojGqcvJiaYPqAgrlAExENPXLgjGkE8w2019VN+ikLGTdI1TPOL0E2NmLiuFHwB4IKNdDygQw982bA6ZO8Tgn1UYZk+3Y0blmD4v1ORCC/DJKRxTH9HXxjKE+5ulOn8Qz8dUJ7CQHjEBwiw1OF49+7MaRBhCbz0yxwK7vENRgXrj1MJJlQC73K3W46jnrRnCoXTKINJVHkQsZB0UK4yO8TQtcrcMKM2lpOaDJKqXDzqUgo9ClwKSacXi8GzzoCXeFaNO3aiHBCUJMSVxnNj7ORKfSzyITSV95/v+mMmL+Tse4Zfc73T07I+PeJbFs9kujH9ps+iUsG+RQc02lh3CnRSI0c1IYWTkuo37iEpwHyphyNdMrkIMExhQKoR9ivriyVcMRrCedFTj6ZSIoe6d8kjJynTgwa4ZqJpw9ITqzok9xnRkNKCSzYw/azOz6LtzJaU0V4BEpfONaNRCrFYUOUl+kZwWn9zYXNEjRxs/w1697212jRXa5T4rkzPcNQMIr6dIEta5fwRHJLRwI+r9MiirHyZlZFIZ7KYtiwIVJBUQ6MZHw8p6HaS5CMH5uQ8fOfTHVOI/M4ZZIQMq4j+miZWEqH16mgI6ZhV1sUa7/8hAbewzFgLDojSQzOU7kAEWp1e4zxCJI0GAkamVi/UwAG3TDhI4wXfQIzwEwDJvGSGQZk4vPn/peODPVU0rbxKLKXNEW4171pA5uMxdZyvM8gEYdJxLVk6gnRauicQRE8BcInUPBAQjL1ns+UuUkXuTRi4ub5O0uRk3YdEFLBMlm4KwZDzavEtnVrOCt3c2caPreTd47bHVoCsiQjlUojv6BAqq6qBPTU0BnXfUpMfNYM6Z8WCoPxInHaGBPM8WLQoAHWWeg9D/Q8NbEWB5zY1SmhO9KB+vWr4Bu+P4IF5TCjCexXEeD73JgQcOeQs9epVmlgF9E6+YiPHyhyC41lR64Emea8Fui9JVUHbxyR3Q7iC+iZj9nX57ejT1FStabD0ZdIZ+Am9ehywB0K8ePJZev5wdZO0023zDL5oFLCoEgQDj5sQqRQqFJAF0uxV4FkJKD48lA1ZQZ2ffo2OtqbEc7xcbeCaKg4W5FNaSUB2YxGI3rkmpp+WL5sTdGGj18qB7AduF4CbmQ/JSEzJDqpmta/bGAFCouKZCKoEyrf0hVUWjLFZLe4riLZ2YBIrAsVI6bAoTpRFExiSK6CuM6wqNXkeSgvJWIpmiS/ye2G1wsYG9egZdVK3pEUUiTee8nrhgTRMYnbgiaSkJSpkNaugZsiAvKzrLSF4NMQLF92iQt7CBvjnSQZeCUZRlcXlt13L0wyU3yUtU05apXFJKv5xOWCf8Qo5I2fCHfQBTOaBSO+sp6BYODdUTluoE1XUDhyCna8/yLCbQ1IVQ0UroRDQdqq8dqjd4j1m9bAAQOpo8Spd9SXCCG78f98U39MQsatzQHn3+ef//CfiyvKx8HhdEnJNOkQO6okV0ewWdPBBHWOtWznW+utHI5MhmFYkRP5bglLOxhWtBo4oFw48JIJOPxupBu2Y/WfL0bsgw9Ao7yIiZ8AMPaiHmvaljhpMGLKBkCXvC/oAyO2xJ70hWAG6gE/70ZkYUsc4yUoohdItnbggz/8sUfrceff7gm1/u2x7uk7FYwcgRkPPoyimTPQ3qFxDUZHk6CRGS3xqVjXZsBT1l9wqCVa4fY4OT8ZabJkWmyu7XTZwlZVVcm/eTbeUW6xJcnAfPOnImR81a340gdkcspK8vm/KZkoXGaxd5Tlt+Gg1KSR6Gji2yPnVEHP6Bhd5OT+zOYuE9G0yQvqZLYcXgVyRz3WzTkUWu0OzLzwHPSbNh7q1q2Qd9ZBJf+LNEx+HtThQ6G3t8HYWQ+FSkm6AUdRLlwNLTDDMS4J3LnmX8XOsvexOLaN4sO6dBx65IHIer1QcnMhFRUD+QWA18c1FGluU9dgZDKQFCdM1YW6ZSuw9oGH8dnsg3Hgux/Cd9AsZLuyPCggYaFT4KNowNDhLyJZcSHc2sKbgSnCDHiEuWSSfd7sQRSMT8GjpWeTok19L7D7/IiETPgG8YxZAridOTm5PNvTw3ZjqQDqzhazxWlfdbQ11gGKC5InwMl9y3xObI0CWyMmPBbCgmqeXreCHVf9DonaHfjFub/AsENmAM31QDQCNLYA8YSoMbV2AjsbeY4LlF+iAVn0+S3tMNNpId8UAFDZhqc9LKHndzxH0WM+JUrMZrIo2FUnEnA0rMvnFTe3G/B4gGAAKMwHcnOBAg8wajQqjjsZo88+G8/NmIlXf3UWTlq5Fi53EEaW0iH0iwQWToWOQGERZH8hGhqa0RXPIpV1Ii9I5C32CGw7aiaGIgP+QIB/Dx3uGvHkzP9zQfsRCdkGbmP0ZHeAtqeoqFiUCy2acqEoGOdDpekcZKSIpLertQ7ICcGTkwOfavLa41t1BmIZMT7ZrZhw+13oXL0Wde+8hRnDhmJYfQv0P1wJKTcEjq9OZcEyGiSrjiRpBhif/d07utka3daDs+aCZKUKeBTJk8V9ogCeO5a4FjS6iJTOyqdZWXgurBbTkMQLqGK8DauuhnnKL+GvKMLE44/EnQ8/jY3vvINxv/oF1LDGJ9xRPZXqrCoMMMkF1QNEu9p4foxG5lAN0+au5bMGGJEVa0ikswgGglAdEoxM1PwJarJ54i5eawIZuOhKt/fMitT4+GSDcmRifzOZDOLdUajBXPiDAQSYidVhE+vDBlTJRMIUcOaCELBt7dfoNhkGhoJgu+ohOdz0ZlzAaEk0Hpo3MpKzRCbWqgH2UDdZWVl7vHNfT58LldV+xCcOCinrEVLQFGDadSuMFYSzVkgrCeHmNwVSXR3kKy+DOXYshgwcgFLChS1aCPXcX/B+TkMCcpwSwhnGUx3RNGCqKoxMHD4n/V77ewuht0dPRxMaPD7C+KvcR9Rt1MheWD8iIev7lXoHO9AS3A8Mhk6ajHBWYryyaWjQMjoc/kIoVE7Jmnh3lwGnYnFHSAQKFPsXMLPcwXe2hyHRtGfSTDGiZRebbLNZ960qiLHP9tSHPkx1XMuJ72bXU8VuWi6a2Nk+YSfjWqsH0d/D00HpCbsmyjkVRBjh80Nat56XiTxUgop0oJj8VYIMpQjkyLA1avCUDA1idXpz+URfgvGQ5rJnCNhzzmnCHPljnNqdR7V8yDp+guYSu01cI9AdLdGKT86uEDDaLMrA09UacNOMJJdwmLmQmBxloZlU72OcIoAy4aL1jLq4AYW0FXGqZwk7b6VGBHkG/zzBqmmB/2x8T5/OTnGoXRzs9Xl6cnm2gFnEdbDNKH/OEiaLr5+bT0EcCySyMDUdJv02qkHSlyZfUVVg6gShFDk/AjcmeWaYKOUlhDw0K5NGVFPyWZwbWhyXZvHfdid1/jzx4NJcKf7bdKHB94a5/LFl/DnmiXaIZiP1hUXzaR4cxmKVYQg54XZx0EMmHuEzvSmlQTrsxAEqgk5ChgrhDJKwWQ0UhCAVBBZinDP3iyxHvqfwbWfteYbeKhPxblyqUdLjRAdq8Z71qQDAlj+qPtEFQa8xrcSs3S9nH2QV0/lndKdhJDXopHGSGlKRFFIkGBnSvib/HW06kOYzza0YgspeNDEOBrR4GG6vi+fHOFqYEYuQIGYhASMEC1UOaJKJ0Ko6IP+UzaW7UCLDEI/HejQZlwFrs/jsb7vs4nDCSYO4o1HAJBpOBw6vduDQCgXbu03sShB3BUOGN2eIrDmViQRnrCUA2INv0NZclrWz6szWNCfbae/zgh6Tajv24jnu9IsHwf9v1RBtv06i4IVYsLtpvjmlWkSgIegHKCtNpr0LapbBdIiR0eRfUr6MfDOqYJDZo+Ank4jBRayPihhTSGkfcgcoxZhI6zzVYxrkgqjcj9VpxqYjtNf4PH9Emuxk/l8pUKnRLsQotUCLKxrh9NNJFqw2wr+hqW++nFIaKgQHy8KnKhgSlJAyCbIsOCjIxLSkiApA5jrSzFqs4iQvtqbpc7OHkkg99R6x8bbPRRqNEBe2hiItxxl2bM1kH9vThC5ZeSpLSLmAA3o0Da0likwizbnDtLSGLN3z0YkCm09d5NQ4bCoyL+q7CC1C8zllwblGFAvN4U4Y3WHk5xfwoRNel4O/lm40laSv+aSzFo5EudMvqxQF/OSEbB7X3z450QCwdCTazXU7qXnB+SAmbNCiTSdXhorBxWVlQHcr5GwS4QzQmhCc+gUuqlUSrIYhTs4/E0XuXkyYFSPaJrCPIPXIiRUE8G2yBM5iurOg19aN/CcSNtvs9gke0AdvRvcmTXRriyPdkUCKekgpxUFT6Gj+uIVi5ebOZJxfjPQ5o1E4ci80aSAVyBkQ0YFMtIPP1aysrub+FplMEtpkWpTJODrW8mfJhHa0dXJwpUM2qVTyUxMysSsHH/vHMOBqbyCn18I80JXHh43qQrjIwRX9lkBxRT/ATCHS2ojOjIraiA4XhfkOiYcQxEGRzAIhv5fXDbrSGY6E6NFO1k34Yb2oCjs066l395Dr9QoM12iUMqCbTvfWY7YptqE/BvmSJrKpLCIt3UglMkgTCyO1wVEMwlv9iCRPaDASMnpduymhVZLgzy/saXohmHjIKXOfk5K9WcoT0tz1vBLQnE+PS+FIWd0myeMzP00+z5yErKm5hTsBTq+7te95/6kIGV/P3Tw7DcnT1doW5oVkUhrEpMjnG5EGo1wR/5v8DBOhskH8dZ11G0EEeW2EXiDyYTcVkWW4HTKaowzG8Il8jtG6VBKSpECxNp+LqxX60/8E34p4THTLCdwaj3JtqiY71cEdekugrKYSRh9uazRujk2eN0vHs+juTIraK6f5JPZFmpLCkKGAl7NVMy5wSSq6mzIaMil+bPnEyfw3klai38/J9IgWVJXRsmUNF75AXimntHK7VMSSGVHPtL4nkTXT5F/yzerr6zgMUnaX/GSFTOG+jera2dDcSYNBTTppxLtqZw04iIEGVBkMHsVAXvkAeJx+JDYvg98DNMQlRLOMCxll/ykVYibSiA0ajuqjj0VTJo2liSRiqSwXNH4zGY+AaKY0v/Hpu/Luj9nC1FMBsP04yy+zo1MubIRJI81GplmH3pFCtiMN1SDCARmKAf6ZdCPBFagc3pLLo1B62TeajqXdYYysrMD0449BhtwAStZKQJwagjMyAg4Tbeu/QU6gEPmlFQh4KNMv8RkG3KU0SFOa6IppcKgqzThn9XUNdBYTvrIBndY5/8lRrEvc3/J61nS0Nh63efMWTN+/SIzs2y0QlECNvgEH4AoVo2zAUOxc8Smgd6Nec6MrzVDko84jGeGsgUKvhGTSxMH33IcPv1mJN3buxNeKA9VMhx8yfLKYhSSCSIlP8uCt/hpNM5G5U+9mJvq7aRpbL46sR53ZFpb3PYpEq0xaKcuQiGuoS2fRznNiCvwOlTMBpZnJo17Ci/HZTbyzVPQJbDA0LEjF4QNw3SOPIq8wH7Vhjfds0toRM9GUdsKRbEZ4/TqMHzkWXn8uCoJO3sCbohIZb5xiaOtK8+9TmBdANh1j27fvoOxz07k33N9y40EP2D/iJyVkfHm8/pXxaAzLli2V99t/f7gcstXC1ts5TuNshhQ6UJIfQM3kGdj+7D1Q2zaAVU7B1rCBXI+MEp+EiEYUmjLMlA5HVQUGv/I+Gn53ATYumY8VhgaCIpDvRlEcGRn7rBO8kJ5rBrjzPd7lwGWFuVBiFKNatcc994fnYAkqLiGR0dGR0LhJf8hIoZYZKNCA3DQQtbKBvGHFemnWgviQ30gFtfKBA3DibXeh6qij0BzTeUWDNCCpsp0RHZrqRuualTDijRgx4VdwOT0ozfegPZLiBCwEIiABI1+WarjFeT6EO+vNurp6GS7XtpsOlvW9NffyxyZkXEcEyoavjDdvTs9fuMT9x0uAkNfJNRevX/LuauL5YpBVBf3znKgZPxMfPXs3Iss+RMGQqVjQpGNUiYqBARntSQNOCTwz3hzOwls9GP0ffgtdKxZCWrcUgVgXbzJJQ0KK6APsqrxJqFii23Sj5fMPkLNjG0dpCCBgLz6rT9JfBAPUO5vWEUlr/MLIUIKfmSjPL8DE406GSeaM3DUwBCQLCUuRI2X66aaqqJk4CROOPQZqThCRbr1ndI/d1dQeN0Fzwuq+fJu8K4wYvx8CbjFgdWtdF8/JtUUyoNlTFG0S+7Xf78HCT1chlYhDzc1foKd7OknxUxQyqWn5s/WSa8DGFd+sHxftjhkBn1/pTiUtvgdxsumPeIahX46MvPLhKKkYiK3vPY/yE/6ITVkF4ZSJIq+MMq/CIyyCWJOOcTIdNBVQnjIbJYcdLfosSXtaEB0bHZvSGZKZLIorXdh88YXo3rKZ4/9p9QqYyNj2ScFyRut4RiBfqbTlkBQUaQzp/gMw/m8PgxHluSLDSQTGTsDhFE1NVPR3UyXDElhiQ0haAibgPYL2sytjohtOuDJtqP/kdQwaNgZFFYNRnkdzlwyEo2kkUzpnXKRaJZlOj8vF/cwF8+fzKr+7sP/CeNcm/gv2xqb+2Bx/WgoNl1Kcjk8bG9qxbu1aFnCJbDg3LzS7iNrVVBnRpIHikAMDS/Mx/rCfI9mwDZGN85FWPVjZlOHwsIFB0ZVkoxGoOW5QqQ+jcnQEUl2Qu8MwusJIU1Iz0gUzEkays5PfO+MR6BEd7mwGfhuYxstLvbQEXM4szn7i909RE4oldBZuEXywhGkiFe4Ci3bCnY7Al+mGKx2HK5mEEk/DiGaQ7M4iEcsiQ4k9eh8OuxYfySsCErCm3YA7qCK9bj60SBgHH3syHG4vqkv8aO6Io6k9hkgszYWKf2WC/qgq2jvC5rKvl8uQPO3lQw9cY53rn2S3EmX++dXlyy35iBka3nnnfZk32nIMmTWS2RokT1qDCnnjK52YePAJyAkVYMvLj8OQDXywC0hpDIUeiUNjqK5pjT/i5rM418NnTxbmepET8iAn6EFeyIXCXDeKc92oKPCgf7EfZQEVWWagjbfE2Y29vdtDJ5BoOLszBjTqdrJq4FabL1JMAmX8ooaBQSU+jO6XgwElPpTne1ASciPP5+LuQMCj8g4sQp+YssLNKg9mqXzGSWWA1iTDh7uI8UfH4heeQG5uOfY/+EgUBWT4vS5s2NHFy0gUfVNujJLY1D+anxvCmrWrsXXLVkhuX2JzY44gyNhL60coZK9wk+k+9LY1cAU73njrTTmRTDOa4WgD8foAaNAZ1zG0xImRwwZh0mFz0LH0XXRumY9WeLC8UeeFYQFytJt2BR8F7xOgpK1fRVmuAxV5LpSEnCjLdaGq0I2qAhcqC5wo8wBpWeYBACMGH1q9YzORNEh7WZNCKFvf08InaqVJwsBRv6TCUBmSOXLE7VD4NDjRHS660+1qqEi5iSQtZf1JfVECOsclYXunhojqxsYln6Fuycc45OhjEMqvwLBKL7q609jRFOFFc7u6lckavCheXhzC8sULYWhZAnXISDX91AlXeLJCaXt6eqsjkPOPLZt3Yv6ChUbQq0DhUIbeng06+RwBqyo4ZFgQv/rV2Rz2UvvizSjMN/BxPUN9XHQrUaacFue44MPnxXvY3dwiFSvazuhGG0x1RLt7nDf6U/mIkDsKmS8qVxG1KDn71KFNJtNColpjGShhrFkd49SxJARI1F57y6UiaWqbRX5MT5ucHU1L6EqbWNEhoSxHwubn7oPH5cWUWSfxfsqyAj9WbW3nQk5LaEDx2rwcH1KJLnzw/seQ5BBMQmDs5fUjFDK+eKU6p2jAU2Cq8cwzLyhk4miOY99DxFQ2xiPPkMvE4QdNxpzTzkLX0vlILHsLLaoTr2zIwq0wLmRU+6NTTwIkkB29/R8WY2sPbqwHo2g16XKcLmX0JWKvZkjrlrBYcCLbgnIB4T4UDZsQRWl6rWoV9ntAi9aEEgoR7OIAL4NyQRM+JAUD9BsJofNxvY4OtxOdy95G0+L3cPyp53KHf3S1lzePbN7ZxdMW5INplqkM+B0YMagCSxctxIZ1KznMm0mO//CTWYt3ibVvfGeV5PQuef+9d7F9R61REHRZkZ2FahAgZ15iocDPq5q48po/o6CwDAvuuhaJTAs+a5SxttXgM5YIF08v49RRvXnT3UgHCMBoCwwJsGZdidSkCIeDz/bOar01RhIm0g1cSHjLWq9WIgFyEo8FacMe2LZYPULJhdF27nsFlPBiNpXCuk4D62MKcpU4Prn9UuTlFeCIE8/kc80HVebiqzVNiCfTotxm1SzpfXICXvi9Kl77x/OQZYYQnT+u7P4zkcRe1NcPb27BA/FYl/TUE09xTSSCJtZHyOiEmoikTd7WNmZwP9x4882I123AlsevR0GJA+9s05HRTY6Bd/I0iNAitoD1aBJu6noxXXQM7YnTMHhiVnIJrjMxFbdPibKPBuutNgkvy2N1offgcC2HXnymJZA2ZpJJ/N9i5KEw40Rc/NZWDUpIwaonb0G4dhvO/e2V8OSUYuKQHLR3pbByUxv38QTcyGaPBFS3Bwu+XIyPP/oIIyYejPJ+A8HSKQwfPhx7c/1YzWWPNpt4wQmvSc7CNXOfe0Oub24zQzwAsEl4hV4gIB/lprJM5oPvzzv7TBx25LGoffVxxL96Gtl8N+atz1qjA4WzLWrHQlKFLyRg2FyTWTc+vY877WLyCc3/5v4aFwbGo0jhewnhtIWNAIcUHRJDIxSZNwlnLLtst2vaAiaCVSH0pMkoUKHmZEo4U+H8lS0akO9B/edv4JOHbsGsw+Zg/MyTUJ6rID/kxcdL63gehWq0fDqxafIR0aoqI+D14sVnn4amZbH/oSdBZ3ufL/bHLmS8u3n+jTfprn5TX2ptbJIeuu8Bk5AVvJt6j4Q1ZenDCYFlp6Tt/fffh5LSCnz2lz9A2bEMq7JuPL8mA5onT2S/XNAs/0xsutAiu/lWNijW5UQXJGTSWi+4ggsGCRY5/QIDJpAc4jkhcBKissy70XOouVdRdjOntnDR35SNIe3FSZEN0QL6znYNa9IuODu24MNrz0VJSQnOuOgKuJwypo0owjeb29HQ2s2HkFFAQUNWCQZFSI+y0gJEW7fg43++ispBE5FfNQoZ6iX91mxz/NSF7AsqG8vKkJ/NlT2hrQ8//LC8efN2M484IqzkbF/zSZCgaFLnZqNmYD/MfeYZGFlg3iVnQU5ux6cdbry7McNnL3kcInVA/hQ3fX1widw/svw2boCq+2MdGFanhN8jnHpLa9Gx/H2sx6yZ4YIqVEKn1b09esJ4gWjVSSyFprR1Mb1HjgsodAuBSxjAOzt0bEo5kcfa8NJvf45UJIaL/nwXHP4SHDSmkDfyfv71LmtgqjDhHPBoMuRT7i/gxuMP3oNUKo7Js09GMmU3H+wt0PU+I2Q8nSEl3jmt1ZObf00slpZvv+V2kzBiu12PFjaeNrXbmpJLWP7DD52FJ554DK07N+Kdi4+Hau7A601ufLwlCx81x/IhC0S2QikL9NxsTUPmN6oDVYcfjaxDxdt6Gl2mxvNllJoQWk/4b1zk7QSelfhKyio+0dKIMYb+R5+AOH9PIUj0PDfHksSZhUIOUVCgq+WNLVm81+hAviuMj/50Ipo3rMIfrr4FJQMmYVKND0GfG69+shHpTIZTK9BFRWeK3s/lVDB4QAWWL/4U77z5D4zb70gMHr0fFGQFFJufuP84/nsuAzhZSfx6xauKN2/h88+/oL759vtGbq5XkM4JYexBtVKDSUd3hpdhNE3DWWeeir/echvaNq/D4st/hkLXLnwSceG11WlOfkfU61QfJOEin4homwh2Q/4QlaCScQ25Y0dg8rkXYpeh4SPZQBsz0UHc+0xCJ4S2ipE5hYQkZHRJClpVFz6QGBamujHy2OMwfNZBiKeoM51PyuTajejTC12CQTFtSLxH9L2dGtYkXfDLbXjuN6di69KvcP7vr8D4A0/CqGoPBlfl4Y0vtqC5LcoBmfYkEqqEkFYjSI+hJfDovXfA7c3H0af+hvt5fh9FlsYPosl+bAXy71nzGG6UTHflpAsTTYklV112pfegGdNZfsAnhWPpHsp1Mht0T855Q3sSlYVe3lr35ysv55tw1RWX44MLjsRJd/0dnwcnYseiNM4Y40SBT0YsQybHblIRyVqeh5MkXqg+8pZb+RCtta+9gl1crMnRVrh5pG9AuTAXCSiVkiiiZFleTho5ZRp+/ugTPY22JGBeRfDz078JEUtQHCoZvVWrwQi6UG7W4rHfnIS6NSvxm0uvx6zjz0VZroopI0qw4JsGrNvSBr9H9JqSCuM9EAaDz+eE0+PHEw/egvVrv8YxZ1wCxVuCRHcEOUS0QhWLXjDcXls/cnNpreuv53eJw16IOotGOLZu2Spfd+1N8LtVnuUmgjfydehGXBAm8bTqBurbE5w1hyLOKy+/DA898jiat23AY6cciJYlz2CL4sZD35hY00R+GjhcmyNpORke+V7CDBJnq8FcmPLAC5j58Fz4pk7DtoAHayUN25DFLqZhCzRshIYtLIs2J4NvxDDMuPYvOPrZD5B1FyJNw0soMavSrEthlklbpk2a8WTg3WYGT4kLXSs+wd0nHcgF7A9X3YzDTjwXlfkOLmCL1jTji6/r4HYp3MHnuTr6/ZrwCIuLC1G3ZTH+8cxDqB40EuMOPBHpRAwetwOFOTSrU5BJ7eVZEfuIJrvRYgLsXJs10p0pSXW6HnrwXjZuwkTp5NNPRUNjmDf5cjSoRWZHgQBxm21v6ka/kgAMQ8OFF/waxaUluOi88/HpVb/C5DNXIf/8a/FSaz5WtmVx1AAZ5SEFWVPifp1w8q1mJJMhGU/Dc/RZGH3oqaio34ZUrJsXrzn8iAiPecTK4PV7Udy/BszrRSaaRUAS6RVadAS17BHQgjqrVnWYiDgcqMwz8fHjd+Dhay5HMJCLm+56EoPHH4qqQhfGDCrE4jXN+OzrnfDQhDmO4LTZDiQOGyovK4RLSuCRu++ApIRwzC8uAyQfFCnCu7pKC/1wOqzt3stStm8ImbVmjl4Tnv9WY6Rf/6qcTNrAH377W9QMqsGYiZPQ3NzJfSvaALvPg/wemn+5tSGKgWVBSJqGOccdg8GDhuC3v/kd5j97H3Ys/gIHXXojvhl5HDZuACYGsjiwUkb/XPKdZMSzjAcGhJotDLrQEY3zGZqB0oHwlQk6c7LS1GhLZR1Ci1CKhDqiPMk0Bhc64aZIlspE1O1OPLbdBr5uM5GUnKgqUBBeuxy33Xw1lnz+EcaOm4A/XXcPcsqGoKZYwZDqfMxf2YhFq+vh96j8ItozdVOQnwOP1407//I7bF63mqd/P37taRz5y2shq264NBqSYc9qN/8zufe/Wi3JUzyQnnDnFlbj+NPPwV8u+QV+deaZePu995BTWIG2jjBUWRFcIhzrRdwYFOIb2N4QRVVpEKpiYMjgwXj6xZfx+CMP4v7b78C8C47HwNlHY8xZl+ObAQdg7SZgmF/H+EITVSEFAQc1bYhBqwT9oY3uTuvQOH26gBTxNjMHmUPRVEy5Op/DxcmQye8iQa3rZtgSByJMhScHCLbsxOs33IcXH76X/76zzv8jjj31N8jLzcGY/l6E/G688fk2rNveAb/HzujbxClE52EgFPKipKQIzz32N3zx4ZsYM3EmKior8e4bz0N6Djj2139Dd0LjiWSJT+1V/6PJ/qtleAslKC45kUwgt3IsTjj7Msx77AacedrP8cY/30dZcS7aO6IWW45VKySYNjdpJrY1RFCc6+VY+NI8H/50+Z9x9DHH4MF778TLLz6P7Z98gOGHHYf+J56D9KiDsLHVDbUeGObVMTzH5Pm1EMGciT074OTBQU+LpZWz45SjRNWUYWhNGLws1JCQkICK/BDgDQHZ7Zsx/+Fn8OoTjyMRD2PStJk45Ve/R7/hU1EcUjB9ZB66Ylk8//4mNLZG4fcoPMHaW80nX4zB73dj4MAqPmPpxSfvwYAhY3H3Y3/HweOrcfU1/fC3v96M0Kv34eTzb+RlJ5moqyQnsH7vjiLcp8zlOOSnt2W70zQoMB3rxoSZc2Ckw3j97/fj3F+ei5dfeY6bjvb2sIBE9+FZpb+p6NzUEUd3Iov+ZUEU+BimjB+NUU/9HXNO+SWefOxRfPTuq9jw4WsoHjkRQw6fg7KJh2Jz/5HY3OmElhUQ6ZBMVOga/Cqjig/cqszTEJTm0gkuTWbWkJEXAIoChPEHnB3N2LnkS7z7yov47J9v8/TL4OFjcdrZd2HS/ofy3NfgMheK8nz4ekMrvlrVAMOg4Q+KGFDfh6uKBCwYcKOqugpLPn0Vt91wOXIKB+Kuh57G9FHVaGjpwl9v/gsSiTTuu/dOlJeW4ZCbbrLqECYwYu86ZfuUkMXyeK5CImKRqtIctLaHcfIvL4FDUvHyM3fjZ3NOwhPPvYhQTg46O7s4ypSiS9ogAbMhv0jitb2Nu7pQlOdFaa4bQSfD7FkHYdK06di+6TK89PzzeGveK1hw51UA/oriEcNRPHYKikZORm6/4WgOFCPrDCLk86JQURCSaIiD0GgENoSWgBbtQGZLLVZvWI2FH3+EFV8tQCpJBSYZB86ajVPO/DVqRu8Pp9OLmiIFpQU+hLvTmPfJZmyr7+Lfk1cx6A2tZLOYBcCQF/Ki/4BqLPn8Lfz1zxfD4fThiWfm4vhZ49ARjuGbLREezd57zx1obmnFKy/ci6piD5/gAkY9UXt37f1C1v9scZV0+H3M9cEfSraOnjCy8sa7nzFb28MyXeQOtwcvPnoD5r/7IqZNn4m5z72IguJSNDW3W0Vve46k8J0oL0UCQa37DlVBWYEPxTlu3jjME7AZYGfdLixdtBjv/vM9LPvyU7S0EgEyLQXwBRAoKEJeSRV8/iDcPjdHTGRi3ch0R9HZ2oiOxtqeZmDVEcCkiWNw/IlzMOOg2ehfMwwulwJibvW5JcSSBr5e34qNtR3QDXLSwfsnBfWGYAji9VFuIj3o178SKxa8g1v+/HtOO/rwU8/h3NOOg6HrHM8fT2SxfEMLKkqCqCpyYfYhh2PhwgVmMH+AHIsn69jU84dg/o3pPfiM/k83bx8TstKtYyaOqLzuzqfN1o5uWZZMaFkdqseDBW8+jJfmPohhwyfiybmPY+DwcWhsaOVXPzVl2IxPgkdVNNKSoiAfyudSUVLgQ2GOG07O4yVzWHVrOImO9nbs2rkDa9eswaYN69HasBNtzc0IdyWQiCeRzSZ5ZOt0OhAMhlBUlI/q/pUYPHw0xo2bgCGDB6F//37I8TtFKUrXeUmrK65hzbYObNjejlQqCyefvSPMoU0yw388mXoFCASD8Pi9+PC1x/HsQ7fBHyrDY089idPmHIbu7jj8fl/PaKDueAZL1jRjeE0h3HICBx9ygrl21TbZmZuzKxvZ1g/MtMKj/wjZdwhZydaR44dXPvDUP8xkMi13dSf5GBlKTlZXFOH9V5/E3bdei4A/gFvuvB/Hn3wG4vEEvxEKgpeJCTlK5HMWasLutibcP+WScoMeLmwhHzE5EpqUEKcURQqIDrFup9MppJNJJJJJzkHBGKUwVOTlBlGQG4TDKaaYKCSuXCUJGifCtTV3JrG1LoyW9jgyNISV/EaqmxLa1uI4o5oodbCTsBHhX3l5KfxeE4/efTPee+0Z5BWV44m/v4ITD5+Ozq5uxBMZlBbmcApQChLIVSBBW7O5DZNGV6CttYntt9+BrL6hK+sdNeek5OrH3wUmOIAV2t7QEPuWkP2+cOvIsYMrP/j0c7OpNSLXtcb4ZtBmeZwKgqEQ5n/8Ou7565XIpOK44MJLcO0N10F1B1Df2CYIeTmZtdjQ3re3CRAF8pXKPz63ytEMuQEX10LEmMN9vB5GT3sunLWsgRY9/+AMkQzprM5RE52RNO+HDEeTSCYJ30Z0TgY3jXQcaTD7O3HNK/6Axx9CV8tGPPfQLVj3zSIMGrUfrvnbXZgzexIikShiSSr4q/B4HMgL+bhW5XQPisw/r60zjpqqfKxbt9Y84shT5MbWcIe3sHRWovGbNXuji3yfcvzBCXxkHo77aMM5y40Mp0fhAkYmJRGLYvpBx6Gq3yA8fs/NePSRu/HlwgX4yy03Y7+Zh6Kts5tvDIdwWxJhd4RTcED3ZC7pPp3JoiGRQX0L4yyGxP9FBDDUEeT1OPi9QDbY78OQyRJNk85JT3ROGSrxeqHA3hu8KlAQckHzOTmvfipLOSziVJP583QcOfskakVFhVAVE+++9hyee+Rm3jQy85hz8cfLr8DkIaVYvaGOz5nKD7o4KTHBi6KyhFDQJxqhicrU54KhU54wQudEHjCwzGxvayhIdzZ+7Kw59qDstrc3EAABmPd/Jmj7oCYr3Tpp6ojKhQs+MdfVdsrUzErpg5I8Dy+C0wRbOuFujxeylMWLTz+Ipx+9hwvnub++GOdd/FsUllYj0hVBKp3mgQGZSps4hWsSi87dflzwfAkeVq5xLICg6AkQx9qCKvhaTV55IAH0eVT43A7emsZzXRwaLbShmFho9HlfwfMK2QkmO9FUuxpvvPAUFs9/F05fPvpNPhnnnHMOTty/GnXNXeiIZpAbdPILjoSeKgLUbEOMi7k5fsGPa10sTRGGSy86Ez+bcxz8wZB+7LEnqKq3dJM7NOCAePP8DhvJ/n+xefuWJsNWodlpjI1DQJUJs0+lHNrAmrIQNx3haAYwMsgYEk4553JMmDoTjz9wD5584n68+upLuPTyq3Hq6acjP78UHR1diMWTPHVL2oSWCAqthgyrbmlbR/KPyGk3+LxNmznbCiZohDUJEkWvVJg3GWJJHdG4JsbQ8CkmghyZgg0+LpqOpecgwR8IIT8/gNbGHXjlucfw/hvPgzEfJh1xPoqHzMC2XWG8/cEipOLdqCzJ5ykJ0oYcxkaTiIlLgwMiMzDCJkJBL/8BadmNx++7FtvlXIw78XQM9UC95757jT/+/qqhWWfrR8WjLz2gdc1dyf8rQds3UBj2GkT2klISYsgqCRmZSKIpF8SGJkb2z8WA8iB3wj3UOR6PYsCQ8bjhrsdwwaU3wOn24dqr/oDDDz4Qz5CG0+MY2K8M+TT3iMYgEpqD+MWsBK7d0SQESvhsdne2bV5tCgQyd+Qf2s+LwRKsp7eSw7qJwyOZQXc3RaUaDzJGDa3GuBHVQLIRf3/wRlx81sl47/W/I1hQQeNKwLQ0Bg+qxpTRFYhlFLy7YDOa2iPIaAzReJZ3jdON+PqJy420YjKVRSKeguJx46F778abXyzEkX++H8+sz6A1nsEffvcb5cabr9azkfpxkZ2vP4+TX7F7jKWftrl8j7k+OKpq66TJNZXLlnxmbqrrkls7EtxMBrwODK4M8D5IKlTXtcaxvT7CHXVd15DRAY8vgEy8A++88Q+8/NzjSHR3oqR0ME49bQ4OPOQoVA0YCrfbg2QyiXiChCBrEQX3tuGRxuxpy7Ocfhtyw+E3lvChJ2I1RUKVpgHzmVCkhVWUF+eitDgX2XQSWzeuxttvvI43X3sF2UwUOcX9cfAJ52Do2APw1tO3Y/2yd3HISRfj6F/8EStWrMfW7S0IBDw4fL8hyAuRm8AQ8jn5OQh4nXA7ZeSHXCgtzse9Dz2J2+6+Ewff8xHG9C9HoVvMOTitH+B3qbjwot/pjz7yhOrKG/hKJrz+55Z1E1/4f3Hz9i0hO7r/1omTBlR+veRTc0t9RG7rTHAzQf7IoIog1yYcik1Ul51JbG+MiUlzRDSX0WAQ5EZ1oaujCcsXfow35v0DrY0b+YdMnHIwjj3uWOw/4wDkFVZAdri5iSNzGk+muOahQjNnVqQXWN1GWT4eUdzos6mATX6XvUtejwt5uX74fR4Q+TbLJhBu24mli7/Ce++8hXWrl/HjKmumYNJBR6By6H7w+4LQMhkoqok3nroem1YuwqkX3YCph56CFSvWYkddGHk5fhw6bQByA26ohP6g2ZxgGDu4EKWlebjhpttx57334dj7P8WU8UNx1kANBW4HvukwsLTVxGkDRdH/F2eeo73wwjyHK7fk1kzX9qssQdN/oj4ZLbsbUmC4xJgiid9s5h8+3s9kqCjyIz/kwaZdEZ5UpRwYmcJEshvBQB6OnHMWJh5wOFZ+vQDL5n+I5cuWY/nSz6DILgwfOQ6Tpk7DtOlTMXjwEOQF8uB0heAivicr18bNI/HZck1nEbpQaxqxJ1pDtBTJhCobiHR1YMWiL/H10sX45usVWLdmBRjLAGoAY6YdgfH7HYmaEZOhOjwgenktTYBuqre6cfYlt+GxWy7BSw9fh1BuLvbf/0ho+krsao7j/cXbceT0gSjI8SIcTWBkTTEUdwAX/fZyPD33SQz43WuIBGtweKmGHJeKhM4wroAYKIEXdzCcPRh4+unHHK3tYe2Tj7640lM8UU+1Lr8WmKkC8/WfqCar2jpxQv/Kr5fNN3c0RuXWzjhPL1AU17/U12OqaJNt/Dvd1bfGUdfSzVME3Dk2TT78XTMkuNwezhFRt2s7Vi//CmuWL8D2zVuQSbbz93K6gqioHgQarTxgQA1Ky0pRWFAAl9cHl9MBr8fDk5+6riOVSqGrO4bW1ja0tzahsaEJO3bUoXbrOiQTYfFjXAXoVzMEw8ZNw7BxM5BfVM3nnmdTCX4BUdLYnl1AptDt9SKeCOPhGy9CLBzBb667H7nVY7Bm9SbUt0Th98o4bNpAzJg8DLUNbbjj+t9h/bYdGHrxi/BWDUNhIXD8IC/mDKChE4zTHvgUYEkHMXCbmFmqojMcZocecrSxcuUG1ZtX8qtkePMzFgml9tMTsqOqtk6Y2L9y+bL5Zm1TVG7pSHAfTAiZX6QAOOxGwH1spCzlvoh5cFt9BG1dKeFfEa8r5aVoZE6GmLYVSIoDppHlJCXN9duxfvUybN+yAY11TYh3EZ35nukkm5TT7tbcc7mgevM4Irdm2CgMHD4B+WU1vMmD5kLB0AAzY1FiCRPbO4DLopLXdZiqG9HOJjx56x9pICp+c/2jqBoyFvO/Wo1oIoOB/UthRGrx8kNXIeMtxoBfPQolrwIjhgbg9zloSjXOGerEgWUKkroYBURls2e3m6hyGTiynxO7dtWx2YccZWzb3qK6i4eelm7+8qX/jarAviVkW5jrgyGVlpAtMOtaYjIRv1FKgbLzpMkohyUafBUebYrykUgxUCKUnmuPpLGzqRtd8UzPYAfiwCeeCyoycxMIBQ4H1Rpp4q3GzVeksxUd7a2IRjt50JBOJZCMp5BIprgmpMYSf9DPZ0p6fLnIKyxBMKcQTk8InkAuvF4vvwhSdLye5YKlKKqYT2lPl+v5wULQBLc/mWcdvkAIrQ2bcP/15/PC/G9ufAxpRwHWr1iKHSs+woaln8I55kRUHvcngt2huDIPebk+Dtn2OGkOu4yzhzoxuYjGLjJ83srgkRmWtwMT8wxMK3Vi46bN7OCDj2Qtbd0Zd17Roen2DV/+v1YF9kUh2zJ+fL+qFcsXmnWtMbm1M8GFzOtSUF3iFcyHMjFYk6YXg1cFZFlMkSOvjeg0yY8ijdbQnkA0nunRgJznNasjS5QEvDmFJqHQWymQFbUHm0ZCbGsbPlyMoyXIFxM5D8H4IxpaqDWPape2LynJSs/r6b3sTbDpRO2aqtBoIlnMZ31qGpzeINasWIC5d/weOfllKCzvh22rF4B5+sE19ULo+WOQEzIw5rCpvP+Aiu4UbRJql9OIKgqO7acg5JKxuRuYUyVha9TE/BaGw0pMjC50YtHiJeYRR5wox5KIOQcd+rPMxuc+AJvzP64K7G0h69v+ij0Sf/Zz3xU+cyGb+Tlzz59VvX3cuMqylcu/7BEyMpcep8qFTAyUUCwhE3mtDE0hsUyPoIgSH8CFzWQIx7Joao9zvlVOu8Tb4gSdAKUvKFgwePGaMvni32LaGplc0nXCybfH5NhCIVkmULITpdZQLxIe22e0p+DREwRBoiWmh5g8e09D6zmLEKVIaLhEOgNDdmPV4vfwxlN385JVyTHnIjD9PNQtXsNTIjBlDJ85Cv0mDIaU1eB0ifIX8eZSbyYlg8t9Mko55xsZexpSAXRlgOMqGIbmOfH+hx+bJxx/qqyp+clAWfXI6JaPa/+nGk3di0LFU0o2oSBdzQKrTpxfAr/Om0/F+q4fI7W3QyXQqZ2Z58vaIBIAmwFR5FAFBos2k6bHkUbrM1JLUEhZxfKikItz4NO4mPZoGu3htJiuZtDrTZ7fMh128tXqWLdLSvaX40ImIlxRmurzOHphW8KZt1G74iIQQinKVvRcyO/iSVo6R41tMX4xcAXJGE+H0KS3Aw+bg4JcD566+yYktixF8VEXomrSCOz4cjUfNb3pq83w5fhQMrgSWiYLySHzuQHUk+CTTLQlGe9N6BeQYVBjMBin1vq8FQg4sjjisEPk555/0vjZSad74vWJj9wDjzs4vf2t+v9JnfP/QpPJewiV+CBFxWHvaK7F58wYkWjeMElH50QAw3pf5tzpcuV+Xjnp7Le3fXUroQ33LHHIBB4zTf8no8aOmrXmm6+Mpva4wmuXqsKz//3KiOCJ0AfkTzl60KT0I6m4bFMr2fAe+wzYGojMLm0yzXDqTmroiKa5SY2ndF5kJiHjDrrVP8D/baMlrCi2Z3aXxfdp2kJpkcQInhjxoaKZWBxL9AKUhqC0C/GwUWK3KZzFjvoIJxwmDcR9Rd0eBaQjr6AA7758P974+/3IG3MQyv/0IhJbwqhfuR7M4YAKEwecPAXeglyYGY3XT0VZTDS7EMlepV9BuU/ibN900ZNRp46qn1cylASceOyJp40Lzvuj4vDnfV2co89saGhI/f8tP/1vCFkf9ofeBB5dhaahK1VVB/drrF97oInYITIyExTFXdOvtBD9awagLCeAUDDIqQE2bN6Mtes2IZHMtvuCNdd3xjY/YgHr7B8jS7JiMrn0k/JS16yNa5cbGeZRahs7OTqCThAxQJPJIe3lUIW5FF+GOpYIA0ZEAvaQB8sX2m0iap+GEEt7kLmkkX6xRJb7bvR3Mp3lfKy2LO12ErlQW5rOmgfFeqoDzPpuNJdS5Ww8QcrU+xy8UZkQHrbA0+fUt6V4BEyVi0QqK7QdTZSzzDQ19RJY87W5t2LxBy8hf8YJGPD75xD5Zgu2r9wESXXD7zKx36kHwOH1QzaI11/Mm6J6r8qRG4JJqNhNdA1C8OI6CSFwWjUQdDtw6+1361ddcZ3qDJZ8mh10xrFYwVG1+FcFTfp/FKpeHmg6ubKKooGH94/VrZ2RTLccCGSnOSWlf25RoXP80BpMGzkME6sqMb6qQi9OJQFDl1BSSEOygUQcm3WD3f/Pj9Qn3v0YcOY+oqU7L4Kpk+nk4+e5EHsHPOSRYxetW7VIL60cqK7b2iRmPDKGqmI/h/zYKNU9v3Y6nebCZvtldqNJr2rrdcK5mFgmj8+kt4rPYqwf4dfE5Dq77GRn+23ItK3SJDJFNKdJsfoynQonJqa/BRhD8MzaqA4RVEi8qN7UkeLRJVFJNbTFEYtTECNGFvLylcVy7fF58OwDV2LZF+9i4Cl/wIiL78G6177CjvU7ITlUFBT5MOOUAyBJDjhg8gkv5J8RekXQZ4F3YVX7Zd70Qt+ZuDnynSZOrBL+7p8u/7N21x0PODx5Fa+kwpuo/PQvI2vV/6EJNLhQKSoGzv5TYeNXz0/Nxjr2U2VlVnrbhyOLckPuAePGYuLwIZhWXorxublGVUUpg56VsXOnjF21qjFoMNjgoWDBIKS2FkjvvoMhqoKHpoxjA2JR/YovllyY667Y2ZXcdTvAeqbeu7zu1anOCHbU7sSAgTVcK3AEBLFQp3UuZL1ELLvbRqoZkpDx727jc3ZTRb06jv/YnuJ475hqftJIYLzCp+yl+7IE1n7ge4hNTLvQToLFp0DYbEDiM+3vTM6+3WNJXVYVRQFsy0QspKzVjU4BigromoHTLrgB4c4ubPvHvfCXVWDwcZcimUijvSWMzo4Uln+wAgecMA3ZNFHTk6BKCLpofjlNAgbaUwwjciVMyFfwSSPNrWJoTEl4vY7hZ/0M3Hn73xyd4W7tmaee/Zm3YFhXsn3DRTxn8y/UOaV/QVv1vAkJ1exb630r7j1qVLxxyyFZxGfIwLjSYFF+dVExaswEDv/l6Zg1fapRVL+LgWVlaGkJ0ahkuj0wSGPVN0I+4WeQx46j+BrobAOrqwNeeRnGls0w1m6De8p4dnlDvfHAik1mfvn0kY2NC7YBE1RKCuYNOmRKeOvSJTfccJl5/fXXyOu3tXLkAZm2oN+F8kIv1wqkyXiurM++0+LjkfngeCsIsGVijxPSi3y1+Rh7De23BJh9/xll33qot7C++6daTI/WI6TFCPhIQsnpPomuPWNiez2NaCQ6UVMMGLOCHs1UkIh14qnbL0bjzs0Ydd2zyJ92Gta9+imiXUn+HkNGFGDcoRORSWki0uRQKYsilcZOO2QMDin8AqTjSbgjWQmj8yQcVSFAlSeedLr2z7c/cbhLBj2Qbln2u3+lzvldQtYb2VHeR3WiYtwZ/ZvXvb+/mWw9XIJ+QED1VtYMGIIJkydh8oxxmD6tzBhQpjOpPSg3frlLCn/+vjR2YCWMVApmNAE5EoOUzgBtXZASSUgjBwP9Kqm5DVJtA9iq9TA9EuREBkZtM5T+JYgdMFGf+Mo7am1CfsA0u3/HGOMljql/fMWz5J5zNkyZNr7f559/YtY2RuTOSAJ+L8GjZVSX5lgtcCZcLpcA7tk/lvtmBg8CxHm1vafdT0jfgKFHEneThz6hYx9Js1/B/hsxg3VAz/icPsqPvhMFGu1R4h4TZpg2nAIB+rxYQkN9s9BoVCITwQdN680AigfItOC2P5+DcDiK8be9jNCgg7DylS8Qj6ehpVMYPbU/xswaDy2R4QEAZy/iqRahUQmfV+2X4LPm7xAdfVSXMC5PwqwyBbF4Akceebz25cJlDk9Rv0tSbWvv+e/KT31/fe9FzJicXzFhcqJp45EaS812AGNLCso848ZOwAGzZmDaftVszGjZ8IbaJaBWRrJDgl4J5j8RkEdj0f33ouG+e3HS5NFgO5shNbQTrgQSia7XBZQVQNKzYE2dYOEkjEwWpqaJcTFOBTp0eKZPMn+zeav88Jb6Xb/97bohDzwwOMu1GVuuK0WTX1RiDacs+up9ffDwUerqzc3wedyc5jLP70DATeZG59rM7XJznn8x5k/IRSaT5fAfOrG7WbXvuOR6tJgtXFZ6RIhO7397TG+vJPaRHuwuXX3/1Udjcv/MBFrCKaupxMrTUe7O6heglje6qBpauq3RmiZ3Awj9SigTXzAXTTtX4a5rLqDpI5hy/7vISpVY/dqnnLPNMCXsd8hQDJo0BMnuFBc0+3rhFREKTGRgSI7M2bf5rCkZCGdl7FfIMKPUgZbWNnbY4ceZa9buUnwFJacmWr/5x38laD3JZuEAqAjm9T8l3bH1ygKPY8yQoWMxdso07DdzEqZOqzbKqzsYsE2GXi8TNY3O+oOp4yE7x0JScznbH9NTkB0efPjnP8H30j+w/7BB0DfU8gGjXMgoDU1fPmXASGRhMt0arkCUTeLKyhoZBMrL8aqhsbNqG8xg4fjxrW1L11pbbeYcdv8B0Y9v+uSCC05T77/vDvmzBavQ0hZFbl4IgVAOxo3oh5DfiWQyTREunE4nFzSRlxPvQpgxjlT9Lg5V66Hdo8ZeGRExwrdf1ytCYpJI72vYt46zhdf2Ann+TAJvk4unxIQ5PguKWCB51UG8BwlBIqmhrjnKewnsHEwiTeeSLi4Nbl8IndsX4t6/XgZn6QAM+ssbSIVdqFuwArrigJHNYvYxI1A8pBqJ7lSPJhOoEcG14HVIGJKjcH/NNp1RTcIhpcCEQgd21dWxWbOPNbdvrze8eaVzkh3r//l9ptN2e1npeW97k38/a66sR34255TT8Ovf/JGNnxAyVOcWCeiQoXVKRsqAKQ+D7JoAWS3uOY8SS4Bp6wBtNVhqDeAcCwQOxjOTJ2L2FhmV1MxAPlJBruiCpq6hVNaiKtdF+cWiPSedEzd0BDwurJaYeUIyKysjfjk5vP6Zr8f/6u1Q/cd3Vbc1bZ6MzM5bAZY/Ij+PpbNZKUGZdYrMOF9sDc696EKc9ItfwOV2IxqJcv+M0geUmqBFn8kRG98hLH0VUl9/zM5vcbO4x8v6umZ9I1PsmSexncQ+bNh2aYlyc5QEZpbjz6ej9PCQiQoEjXmmETeUQoknNZ4Coe4nLnA0n12jwAA4eL+RuOyCU/DFJ/9kBaP3x5g73pNSzUms+ucSLmgyM3Dw8WOQU14CLZnmo6NtrUq+Gn0nr1PGYNJofCS1GBfUrUk4ugwYke/A2nXr2axZR6O9K637iwfNjjcsXPBdBXX6bUrNb5na9kTee0Ve/eD75r6sH3nsTBnakzIyKWiO44k0CYojjxdxe65CYwegbwSyqwBtO2CErYy9CzRZRS07CfMeux8bL1yM64qKoBfkQinMBWuLwmwJQ9M16MxAlkJywsYTfz3BXCQFLaaJao8L/4TJzop1Q/EO/mc80RCUEBvskZylTmqerSrD+BHDMXHESAzMz4M7leI8rruSKXy2aAkWvPEmyocMxm2PP4EpB8xAJNwFp8vByzy8OmC54L2dl706ZjerZ/9pHdaTG/teFum+77jn++9xpKXlSNASSdHhREvrYbEWWoxSJVRqIhQJQXNE+kLCwPIQNm5vA7FN8soBmVdNx+jh/ZHo2IZzTzuepUy/ZCSiqD7iGDbk6rlS69e12LRwAxdgJ7KYfcZ0+AsKuKCRlrTrqWRGKZVBBM4UDHg4xF3MfifivhMrGQblOPDlV4vNI46YIyezcqunqN+sRP1X6/es2HAd6fMX3+bLtl/+yrufZWfOqnFm26+FZNRByb0Nkms8GEkNMyEpTkDfASTuANM6IDGCqVA5iN5T6b2is1HIsgMNEeCi/T/A08hDvouuMgYza3INSO+n8jSChCyo5KEjbGTwta7hawANkLCGNtXnxcCaAagcXIPhY8Zg9IABqFFhlEW6FCWbAVwuQCUgIU23zxe6vbQU9ZqGK6+6Fq8t/hovPP8i5pzyM8RjMTEYyVIzvT6V7YlbEJtvy9juWqtvRftb8WIfLdZbC0Cvie1NApO5zugkPAL8yK9djrgQPhiVmZIZg8+vpNwYCR5pOsKujR9ciJ2NXVi/o8MqTQHptIaK0iBGDK7ApeefbCz4fKHiGfKzP2S66w42mxcfW33aRcaw39+t7HhnNbZ9vQ5wuREIuDD7lKmQXB4+L93uKxX5PXGG6H5YngIPR/uKH0d7dmIFw8CQA/989wPzxBPPkE1HqNlbOPKg2M63N/ctP0lFI48Z3bnuneVXX/kH+cZbLpUzLX+UOM6JnnQfAMl/Xg8nK8xusMg1gLYZYKKEw3udzSygZwA9C8lIc7/MTCYgl5Tg9JPX4ldLdBya47F4zEnIJHQZOtakk1jETNB4Txr11yq7gYFDUF5QiFytG4efdAJmHXk4cpoaDHdTI7Viy2hpltDZDjOVgp7OAnNOhTRmLCQtA8nvg0n5peceh4Omkrq8uOiVN/DKN2vxwZdLMHrSBBgZQSkgcPt9rJmF27dkTfxlWzZb7fTx5/voO6vhpK9w9dFUUq8Y2oLGcW4kYFmTmz3eZMJTFVZbnJXc1a1jknzUjoEslX4UCYUhN3Y2RjgIk6oGtKgiQX8ftN8oopHSb7npKlX1ln2sJxsPRcVUjxzrWGFGW4cNvPh6Pf+oS9X6d+ajvbETTHEgP8+BmSdPo8IaB0/yCcLW7yYfjfQ+VQiG56t8sCtFxcRGSfPZz+jHUBFw4IUX/6GfccbZquopXO/NK5vR3bCEEJq8YiO5nXmPD8g1fr1ww0dGTuZpxYwuhpQzDLLTRyoJTB0AyT2ZC5OZ+ATQGiDJQbBMG6AlaQgkJEvIGEWMRPlt0IxGEyhJ4spLoqh+RcEcv4qvohGso0HtALYQqV2gAEVDR2D/CeMwacowDB0exIBKDX6/E1psELa/vwqJz97DhJwgtK4ocTRBcjohud2QmluAUD7kW24H01JgugbJ7QVbuw64+x7oK9ZCoprmSUfgpNfeRqe3AH9+9n3Ol19R7IXP6+hp7BBjZ/YwlX2UETeV32X49hC2PUtUuz8uiXqlFTEKrgura9wSLp6usFi86eWpjMk1GW9GoXpqIsu7zdvCCV7aCnhd3GmnSJryg6NH1OCteXONP//pMoU5CzqceSVjUvXlraRRlFmPzTa/efw9Fl7pGHzto6x65nnShte/QGtrlJwU9KsOYvoxE5BMi6l35PzbF4lqpTdI0Abn2qZT4pNZKKF7apWJfK8D9z3woPGH3/1ecQSql3v7HX5odO0j1EVNxsNRe8acUdXPvPpb6MtvkWSWBbx5kAJFYK5cEBmrRPcUOBABbrqdt27BIEiJzvsbeUBBwqUTxTlN0mJoqjfx6cImPDqXIVFHX1BFprgKI8dOxITJEzBsTDWGDguhX0UScJN/twVIxwE9BFMqhxw8GVCG4JMHHkBs7uM4YdBAaOu3QlZlSG4XpGQSGDscqBkAxEjIDOoUAVu8DGhqh6yqyKZScE0YgXWDBmDmo3/HGZfehnEHzoGDJVBdFkJ+jodTEBD9gF3mEWNv7IZdu2HETsj20XK7CWAfH83OWghQCGyfi+PLLGG2URr2KEIRWUpcS/EudM1EMmUgksjyYKA7nkU0lkE6q3HhpNwffR6ZUyppjxtWiv3HlePBBx5hl17yB4lJnpScU3OI3r7iq75JdbVy5v5688p3nKoZmnDvGyxQPUv+5o0vEE/q0DI6agbnYvyhY5FM6D10CzarkD3Bhe5rchT4aQ6CNcnF75JxUgVDoc+BG2/6q37D9X9THcGy97Tu7UdRtYbiVfPqK2ZJf7llGvSFj0JxuESSkpx4lxdSsAxw+MGMOGCmuHbjxTReGyFK56zwzWQGyckQCWdx08NhvDbPhMtXikkzZ2PWwRMxbvJA9K92Iac4CShbAWMnkE6AZf0wzEJAHQLJPRWyq6rH1DAtBsUZwJPnn4+yF1/AkcOGQ0/EIHNSVhMYNlCwBm/cAdad4BcB0fRwxAOfSykhkUwjOHIEZq9aga1lQ3DJ9U/BpZooLwpaZo5mEKnwex0cXetxE+UBEa2IWiMHIfaNHvvAe3bTWH2ksK9wMSuKtINKHmzsEUTw1INucn8rEsvwPFk4muXzwwm7RkJHJlQkTsWwVJoWXF3sx7Sx1cjx6LjuuhvYIw/dy2RHflyunHaFvuOfj+7hgPP0glIy5Qijs/ZdV0Axpt3/vuLLHS4teWMBp5vSMxqGjSvFqAPHINmdtgTNKp1ZLJL0BwnaIJr97hGDsKkphabZXTSIBE7FWWf/Rvv73Gccntzqi1JdGx/hpve2W07AZVdMgPbpvZBlEqCezIZg3fU5IAUD4IUy+pX8EtQhyxQQGMjGM9AJ0rzLwDl/TWBXpBDX3XwLfnbqfsgromYMLxVFgHQTjCwBCGmI4wDIzsGQ1MKeKqvEspCNHWD6LkCvBUt+DUkdhah0FO6aNBWXlg5BiCaCtHRColCf/EHaMaoBcp9IdGXzljXqb5RkbM+kUKn68BAyuALAAw+/A8UT4MnLoJeaQFzwuAkCbXU+cUAhTfslRKnKecSI84I3Equi0E3H2pBpcbyVhtgjXWGjMIw+/hZFiyQkGXLoMzriaV34XRmdR4+cgIUahImGlPvg1IdA2X3xWynay8/zYsLIagws9eHD99/D1Vdfj/XrVhpq/mjZLJ76W3PD4w99d7eRSC84CkedpXU2zQ32K9dnPPyumgz7seSthXwstZbSMGZ6NQZNGYZ0d4b4XnaLvsVvF9FnVUBGyC0jrQs+kP4+4NQBCuqjCfOgqTNRX9vQVTDuyJFqAIqZjkRl4hLlQ13JcZcIGixGwpgZkysvTx5tIA3UIwXMYGQ0xDtSiDVnkOnIoK3LxJkPGKismYyVS99CSdkGoP0yGNETAd9p1KJD2Q1I7l79zcwMkFkNSd/Bb8hsgqm3ANQqxgXdASPWhJyBFRhz9lh8ecdqHF1TASOWglJRAMnlAqtvAWV5ubDxFLi4MDhrDwwYmolWp4mhbj+Cna28D8CZX436unqe9ad+ys4o48LkdzvgdTt4/yIJBDEyJtLabnQFPRBp7C6UvaUhK9S3hIpZyAx7oAMJEAkSpVEEtFuYI9HSJ5gVKddFz9vkLQQCKC8MYEj/QlSV5cElZ7Fy2WJc/puH8d4/36CP1Rz9j3XoxTNr2eJLHoXURZHdd9QTef5K1TrWPaMUTRzUvWPLnxdfcZo++e631TGzJ2LlRyvg8LjwzeJdUF0u9BszAOl4ugctIhAjgnGbwMF1MQZPFjiuWsGBJTK+aDLx0pYM9hsSkE+88GLj3ksuyI/uXHmmyiTHxjXLtgyHMZ0xd4mst6+D5PZzR5qZVOsDtO4smBtw5LiRbs8i2ZpGJqwhk6ATwpAfknDvOwxxbyFe+vAxFPj+iWzt41Bd+ZByZnECEZrVwWf38cvfAZZ6D6z7OUALW3MgFWtilu3MGFw7SXoUrPlDjDuyBvPu/gpH0YYEPTApAKAIknBWFis0aTQOiaYSjGnAzYBVWgY13kIUFOQjv7UJ9976F0yYdRQmTZyIqop+vBJAL6KhpN2xBFIZ6hrXRfLW0k5cCCzodC8qttdK8sDb9rtskj0+D7MXWi0e620CJsAiaSgbSUvHEsSHTFEo4OQUULkhHwcylhQEEPTKaGyow1svz+UpmS+/pP4OHUruSIZ+Rzv0QI3GtOgVkCQDJ7+iYN48fM+iK1I12lZereQNquxcvegXq285V5vw11ccY9LDsfKzdXB63Vi5cBscHgdKaiq4pZL5nARr4BgHWpKmZRxhm9YBv0pDa4HXtinQHQy5Uw+XXHmlLNPdfZwKf847n3/VOGL9N3Xm8JEHIPM5mSvKxqvQ0zQcSqj97i1xaFocWorMkZAVsvIeScKiWgkfrTXw5Iu/QkHuV8hufgyO3MGAlAayywBfFRgTdJk85ZH8FOh+EpKRBTPIBAvYCqlSRmPdqOmCGZwDQqIuoI5VKCsYjKZSBY3RLCogQdu8S8B1aMi8ZHFSkIkxJTgZNdwaWMgMfAkDIxwu1KYyfBzNBx+/iQ8+fgPlVSMwbNhgjB4/CTMOmIrBA/thSFUpvD4f1yDkdGcyGufjT6QyvLmE/KNUWhPUTtaUXHtUDm8m4Z3lvVRSInI0LeiRqAsSaNDjcYqmDocCj9sJj8cFv8+NgM+FghwPH0lIPhrxvdZuX4P3XluJjz/5FEsWL0I00sldLTlvpCkXT2ZGznAJqmcDS7beiuVXzwOulzHvZ/8VPFrAtZghG11bz5Rza6qbPnlrhq/0cn385XeqyWgKG1bsgNPjwLIP12OaQ0VhFVUFMpDUXn+So5Ro1I5i4s3aLDZ0KogYMnY1dWHjN1EcdcxgqXDgEKlh+eIK1Tts9uPRZS9fdO+tS31PvHo+U8eeKBmr3oCeTcHQqQdRzOPjg0ZJjRPNUYrKnKKYHayQ8OZnWRRWFeKo46phbnoZDkraalFAcQPRF4HUakiOCpFnS68Dy9ZCkn1gsod/ceo7hJ1ro7SIQaUVAxL5X2T0Mgl4CpqQqZSw5oNGVOS4uM/Cp4ZTgdjCf1EfYRd01Bka1ps6djATbkgo9gXwbHs9mmQfcw451TS6a6XGjh1S44cfSZ98+C7uvd2JvPwC9OtXjsGDB2HgwIGoqqpGeWUVyorzUVqQj2AwF06O6hCjT+xgkqNUKbi2+DB4nyevwfIrCrzHwHLY6UZzLGkJ82sgnkhwZqG21npsbmpC7Y5abN68BRvWr8X2bbVob6cMIh8UYMJbxdTqyRJCw2H6KhXDlQuWarsXCy/4o3jX62Xgxn8FrWploQ3ZrDj4eDkb/2Drcw9M9uSXGjW/uFRJxFKo29wExaHi6/fXYsbJbgTycqClsjyHZpfjSDSIRZJKgZu7DD5ep601gvZtTdCVwZI/FAKYM1dtXfZ8bW5u5ZWvvbb54SHXfaD/6abZqur3QlvyCYy2Zu6oG9SPaFAB2wGH34tQpQN5xQaUPAYEGNb9JYvxk/vDoTYhs20tHAEVCHVDChaRMweWWg5JWwOmeCFRdt45EkyiKFYB9CRY90ZIyTZOkylZNJvCMgkaTK4JIo0oKJDQQQ6iqfKRzXRs1ADawVBnGqhjBjLM4AO1OFkeGIYXlSLj9+C9Le1w5EySMjnjFSkwCHJpBnK2y0SiyTTjTUpHvEXqWLYWy5dR1G8vFzxeP3JzA8jPy0VeTgj+YA4Ki4o4P6vf74XfH+DUBWReVVXl3ew8A8+TqTTnibSsyVG5sVgcyWQK0WgMnZ0diHaF0dHZjo7OCOsKd4GxdJ/6lpeGLTGpeDqUYIXCQgMV5i6BQTz8TKfz1sRSbddAj7/MHdiT5/13GmzPRcIoY+0jXa6aw+ekm9YvXnvvnyuUwhJj+PGnK5nnv0RrUxeYQ8Hid1bhwJMmQvX4oFEC3IKY04XCwy/Rj8PdlEQ0A6fXS83ETKOMLcwOCmuVrkjzI/6cqgNv/MtHP9tR22Jee8ssufSYQ+BvbwFiSYDKNw6KMuldY4h1xbF0QwRfvJPAN6s11O0ETj6jmOtQg1qySJ2n0pAjXZDySoC8SrC8aZAUqi1FICV3Ael6INtFDh/PtZFZEl0VNqiX/CtRURCZyQSyho520DzLDLabJrZqBnbROGSYvPRPdCNeC86cho4BuYU4ccAQnL9jI9uiSwxlB3bDSKyAoZUzSc01vJXF8PWTkZeEbKRMyUgxyUgyZMISy3TBTIeRSkWQ6kxKTY21MvcrBVv/v4I6/p4l0TclAI24V/0M3nxFKh4IxRWE7M4D3PmS6S6UmJrDMXe6QdwK0ifQ0kvBIpsg+7qgmcux4rIOa/gUMO9/1HzLYe2pbR80OIrHHK4b2rI1N13odeeXmmNOOlhe9tzn6I6lkUqaWPjackw9biJcAb+otFicbNwTIHiQx4ltm1vQWB/DuAnVCEfDrGXbesK4tYiEE5Ok+O93neP+++ET5z7/YfXSj9abs08cIO83oxgVVUWQZT+6OtuxcW0TvlzQglWrUmZzC1gKClRnBcuYjYrO2UFcHD1LiE1K+MtGGkq6Dkp3B2TSSlQfi5NwEZ+95TzTt6ScHEUYnAxOZLy5z0eCTZEjNQXqad6ythMM/TJJ1FPuFTQbnMHDTGRBggXEIHOOihnFFZhSUILrG3fi9bZWQx58pprJG/wQ5p97DTv5FSd2NbthJA+SZPk4wDyBqf4c5giBe7hBq65J5oC0hpGCpCXJE+SkY5KZpavJwibw7y/OttWtLpK0dmJMEvVSxSFBckiS4pCY4pIgu8AUj0TanRlGnM/BlBS/xfQYZcBOSU+9y2AuBTJbsfQ6QT3Ud3EH/2f2FOv/6eIALK119XqlcPyJRnjb2yuvOl2d/NCHbPSJ06WV/5jPC/exaAaLXluGEQcOQ0FZHlQnNdER+aCBdDKLptXbsW1dKxSnB/1GFKN13acs0bRTlnJr5gv1POE8B1Y8ruGAp+90ZNsv9Wx+NhuPrOcpknwaWMAgJURiAUBAkgIDFUdufyCnBqa/Avriv+Kc00rx5HMnIDHvAehahG8W52SVZT7Kz1XghVIcAqMp8HYmkmsvau8XqQEBrGcwqVEjoSMR1pEMZxFt1lBSJOOCp7NQlzIcTtyohFzgSQ4JOU4XAl4figI5GFhQhJDixI5MBrc2bcc/O9t1tfIIKV594iKkoifCH4tg/o27h/cHPFYKPTUchumHpA+TJOwPZgQZYxGJJmAxNhSyWsmjZH5W6PtT9CuaVkTOU2T7xBTdPshZyR7nRvJJH2tCMnSDSVKUSXIDTO1DSMa9yEQZV2PE76cqbVh6ddvuwnO9jJlWG0DRCIZ5J5OR+n8Rrj0XBx3KeSN/ZYZ3Pu2vKNXGP/axKidzpRVvLEFaowkHJq9tBvL98BIIlXJ4lN+LphHvjvPAa+h+wzHt2JHss4uOYrsWfqn7Bk4bb2d3JOAGCeOSJfD0vxfQTlZS7VAyYcjZCCcgkZx+mJ4imM4CuuK2MCbtgJnJhcP7jW/FreMm9E9N/mjFJUxa/Ymc2rwAcHu4CaTGXXp7V9ADz4BcXv7hmWN+uji0EzolJ+MGkp0GEh06UhEN6W4dmRhhzcjRlMB8DOc8xXB0NzDO4YTb7edCVZWbC5/LjWjWRFgGtutJfNbagPmdnWik9tzqY5RM5RFANjINy65cYl39Ro8fQ+u/82Wm3xaA5BkAphfDzBaASTlg2U2AQha6XDalIibJZSAdaDcoc4AWqTypG5LULIM1mmCU1NMhOVrgYhHM/5Nl7r5n0Xfl3289+xcd+v/HJZK1cs7Qq8xI099yR400pj70gZJpMbHinaWIdyegup0iss5mOSBUFDfpUgeqhhZh8AnT0T7vPn3lndeokq90LktsPfs70SzKpFsPM5zeQTR/FhImUnMxTAoX2TZA2gRnx/uYf6OweQByCgf/Ap1bn33p1dP0w48ZpabfuhdGuhWa5BKoCGrBd8oIjswVUz+SBjJRE8nOLJIdGjJRnadGtIyJbJq6eGj7JKgO6qwB+veT8PJmE48/4cbnI4bDBRlRSrKqMtZnk1jQ1YaV3WG0awafTaurAcbyR5nx0oMdurugXTbjfzKXXfMccL303ZvVR+Da1ks4ECafsXnyyTLmDf+/3eCTSehJK9G6QQLNj73xBhvL/b+pqf7VpUJSdCU4+CEjuvmi0v0PN6f99Xk5mcrBjsWr0VHfDoo+jVSCAz+dHg/yygsxbNpwhIYXYNvLj5jrbv+tzJT8Wm/umKmJ1o879uxwsKu8/8KPo3CZhL9ZKR42x9n50py1YwY5+n3+5QUsoHXKqc9eRjYVhaaoMHSqcTHIfgXZBJDsMLigEbaMmJP4h3IyXwVMpUlqMnKKFOQWyVC9BhauNvHLG3Qc6ajG9KAfr7fWYV2yG506Y+SHpSWHCW8pkwODVAT6Q/dXwfSWUPViA/TYz7H86nX/P8L771hMwvU3SNgwQuJCaJssWva//5VVZL2GayYuSD+EEP13SzRUyw7IJVNazKZFxQWDJ7Ehv71eKpl4BB9g1tyURndHBG6PAn9+CEqOE3J8F7bOvQubXnrAlFwFKans4PPN2ldeIFyZ9L1XV9+TRyen77/n3yDS8WLxImywdMKhyeZVH55wWKX57LyTJbcZkZKLFiLTWItUPANqpqGylUYVI4NcNhkS1QQ9CgKFLoQKAUeuCfi5j4/aeh3rd5hYttLEa29rPFbwS0A9A4vKLhOeIrh8/RQEq6H7qmG6CyhVQCK7WWLaJklWPzL1tlew4rYoZl6vfssP+8/6nsUkzLzBJWUD90BSzpUaP5HNXYtkII7i8dNROu0QOKqGQPEW8KJ4tGUXmpd8iPZF7yHbHYWUO1rHoFNU5i6egwVnv0GEx//6VfhfL/KAjbycooujkdb7D5lexu55ZLY8dHQAaA9D37EdydY2GAQsJOZAtwJ3Ds30ExPfU2kDmzdHsXZdBqvWZ7FqlYaNW8G6qDMfYCFHHkOwDAlXmax5ymTdVw44gxw0ByAqyVgjMfMN08y8h+XXbfmWw7xX/Jl/g3Wy5a9Ou3O6pHq+gpYit9KUM62Mta6UzJZVFL9Le8QKIuLxVEMunWqw0gPIaXuF1S08Cw3DM3Tu/3eETGgKAzP/frGzdf5djk1zkedh6glnDJd+/vNhGDKQIai08BmRxNfV0pzCxs1RrNmQxJoNGjZsTGDHLj4kj/pimKoUSsgboBg5A8A8JdBdRYDqA6jkZKRikJW1kI2/O2R1kaazDnx9ecvuJ+tkBW3DpT007n/Wv7QsN2hq8CJJli+AkRnBo2qC3xOWMNXKqzkEFOXlP8rr+fuBecpFsl3rWs2UtgOw6I64ja353xQyHVNvuxnu8qsd0fVQ6z5AqvMbMweQKgsBJQDJoI4izUS0A6wtydFpUOCB5CqRHPn9ZTOnBrqnGKYjREnIKCDtgmk0STDbJdPcbErGYqjyJiy6wp4L2FfFK5hPQvofrfW/uCRMveswSTJ/D12rYhJyJMWZw922nmqM3AhZjtD+SIq6yjTSL2PZlQ24/noZN4q9+F8ylxwvyjD1rnJJMq5kUGZBknPVVFuJFNkCNd4EPdEFTU9DVl1w+Isgh6pheEuhu0IwFS8NRUiCyV+AafMV2dxsaNGl+PrGlu/98TTc4MftQO/b6+STCcmxe2rnkDt86Hb4gVbrAT+wNNv+7Qvbkgdr/W/5ZLuv4Sc7kTc2JBn+WxiUA8HMPElScrlHT+lx2VkLxbUKpq7CzG6WmbbWZJnlWH7d5m+/2Z5JyL2VM/rPgn0xD1/PbK30PUvCzOtFTu87rMn/BxPHM28NxZTmAAAAAElFTkSuQmCC" alt="MediLens logo" />
  <div class="brand-text">
    <h2>MediLens</h2>
    <p>Medicine label helper &nbsp;&middot;&nbsp; Local &nbsp;&middot;&nbsp; Powered by small AI models</p>
  </div>
</div>
"""


FORCE_LIGHT_THEME_JS = """
() => {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.replace(url.href);
    }
}
"""

# Runs in <head> before Gradio renders, so the page never appears in the
# device's dark/system theme: it redirects to ?__theme=light immediately.
FORCE_LIGHT_THEME_HEAD = """
<script>
(function () {
    try {
        var url = new URL(window.location);
        if (url.searchParams.get('__theme') !== 'light') {
            url.searchParams.set('__theme', 'light');
            window.location.replace(url.href);
        }
    } catch (e) {}
})();
</script>
"""


with gr.Blocks(
    title="MediLens",
    css=CUSTOM_CSS,
    head=FORCE_LIGHT_THEME_HEAD,
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        radius_size=gr.themes.sizes.radius_lg,
    ),
    js=FORCE_LIGHT_THEME_JS,
) as demo:
    gr.HTML(BRAND_HEADER_HTML, elem_id="brand-header-block")
    intro_markdown = gr.Markdown(
        f"# MediLens\n{UI_LABELS['English']['intro']}",
        elem_id="intro-text",
    )
    if IS_HOSTED:
        gr.HTML(HOSTED_BANNER_HTML, elem_id="hosted-note-block")
    browser_language_input = gr.Textbox(visible=False)
    browser_device_input = gr.Textbox(value=DEVICE_DESKTOP_OR_LAPTOP, visible=False)
    image_source_input = gr.Textbox(value="", visible=False)
    match_text_state = gr.State("")
    reachy_running_state = gr.State(False)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_id="top-left-panel"):
            image_input = gr.Image(
                label=UI_LABELS["English"]["image_label"],
                sources=["upload", "webcam"],
                type="numpy",
                height=190,
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
                read_button = gr.Button("Search", variant="primary", elem_id="read-btn")
                processing_progress = gr.HTML("", visible=True, elem_id="processing-progress")

        with gr.Column(scale=1, elem_id="top-right-panel"):
            match_output = gr.HTML("", elem_id="match-output")
            explanation_output = gr.Textbox(label=UI_LABELS["English"]["explanation"], lines=5, elem_id="explanation-output")
            warning_output = gr.Textbox(label=UI_LABELS["English"]["warning"], lines=5, elem_id="warning-output")
            source_url_output = gr.HTML("", elem_id="source-url-output")

    model_note_markdown = gr.Markdown(UI_LABELS["English"]["model_note"], elem_id="model-note")

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
                image_timeout_input = gr.Number(
                    value=DEFAULT_IMAGE_IDENTIFICATION_SECONDS,
                    minimum=MIN_IMAGE_IDENTIFICATION_SECONDS,
                    maximum=MAX_IMAGE_IDENTIFICATION_SECONDS,
                    precision=0,
                    label="Max image processing time, seconds",
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
                max_attempts_input = gr.Number(
                    value=2,
                    minimum=1,
                    maximum=2,
                    precision=0,
                    label="Max attempts per image orientation",
                )
                ocr_output = gr.Textbox(label="Extracted OCR text", lines=4, elem_id="ocr-output")
            with gr.Column(scale=1, elem_id="technical-right-panel"):
                use_vision_ocr_input = gr.Checkbox(
                    value=not IS_HOSTED,
                    label="Use local MiniCPM-V 4.6 model with Tesseract (reads image)",
                    elem_id="use-vision-ocr-checkbox",
                )
                use_ai_model_input = gr.Checkbox(
                    value=not IS_HOSTED,
                    label="Use local Tiny Aya Global model (explains and translates)",
                    elem_id="use-ai-model-checkbox",
                )
                server_status_output = gr.Textbox(
                    label="Local model server status",
                    lines=2,
                    elem_id="server-status-output",
                )
                start_servers_button = gr.Button(
                    "Check/retry local model servers",
                    elem_id="start-servers-btn",
                )
                with gr.Accordion("Model server URLs (advanced)", open=False, elem_id="model-server-urls"):
                    vision_ocr_url_input = gr.Textbox(
                        value=DEFAULT_VISION_OCR_URL,
                        label="Local MiniCPM-V 4.6 server URL",
                        placeholder="http://127.0.0.1:8081/v1/chat/completions",
                    )
                    model_url_input = gr.Textbox(
                        value=DEFAULT_MODEL_URL,
                        label="Local Tiny Aya server URL",
                        placeholder="http://127.0.0.1:8080/v1/chat/completions",
                    )
                reachy_status_output = gr.Textbox(
                    label="Reachy Mini MediLens status",
                    value=reachy_start_reminder(),
                    lines=6,
                    elem_id="reachy-status-output",
                )
                reachy_button = gr.Button(
                    "Start Reachy Mini MediLens",
                    elem_id="reachy-mini-btn",
                )

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
    ).then(
        fn=begin_model_start_for_image,
        inputs=[image_input, vision_ocr_url_input, model_url_input],
        outputs=[read_button, server_status_output, processing_progress],
    ).then(
        fn=start_local_model_servers_for_image,
        inputs=[image_input, vision_ocr_url_input, model_url_input],
        outputs=[server_status_output],
    ).then(
        fn=enable_read_button_and_clear_progress,
        inputs=[],
        outputs=[read_button, processing_progress],
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
        show_progress="hidden",
    )

    manual_label_input.change(
        fn=clear_manual_prompt_after_typing,
        inputs=[manual_label_input],
        outputs=[
            manual_label_helper,
            match_text_state,
            image_input,
            match_output,
            explanation_output,
            warning_output,
            source_url_output,
        ],
    ).then(
        fn=begin_model_start_for_manual_text,
        inputs=[manual_label_input, vision_ocr_url_input, model_url_input],
        outputs=[read_button, server_status_output, processing_progress],
    ).then(
        fn=start_local_model_servers_for_manual_text,
        inputs=[manual_label_input, vision_ocr_url_input, model_url_input],
        outputs=[server_status_output],
    ).then(
        fn=enable_read_button_and_clear_progress,
        inputs=[],
        outputs=[read_button, processing_progress],
    )

    start_servers_button.click(
        fn=start_local_model_servers,
        inputs=[vision_ocr_url_input, model_url_input],
        outputs=[server_status_output],
    )

    reachy_event = reachy_button.click(
        fn=begin_reachy_toggle,
        inputs=[reachy_running_state],
        outputs=[
            reachy_button,
            reachy_status_output,
            image_input,
            manual_label_input,
            match_output,
            explanation_output,
            warning_output,
            source_url_output,
            ocr_output,
            match_text_state,
        ],
    )

    reachy_event.then(
        fn=toggle_reachy_mini_medilens,
        inputs=[reachy_running_state, vision_ocr_url_input, model_url_input],
        outputs=[reachy_button, reachy_status_output, reachy_running_state],
    ).then(
        fn=enable_reachy_button,
        inputs=[],
        outputs=[reachy_button],
    )

    if hasattr(gr, "Timer") and not IS_HOSTED:
        reachy_status_timer = gr.Timer(3.0)
        reachy_status_timer.tick(
            fn=refresh_reachy_status,
            inputs=[reachy_running_state, reachy_status_output],
            outputs=[
                reachy_status_output,
                reachy_button,
                reachy_running_state,
                ocr_output,
                match_output,
                explanation_output,
                warning_output,
                source_url_output,
                match_text_state,
            ],
            queue=False,
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
            max_attempts_input,
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
    favicon_file = Path(__file__).with_name("icon_update.png")
    if favicon_file.exists():
        launch_kwargs["favicon_path"] = str(favicon_file)
    demo.launch(**launch_kwargs)
