from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "medicines.csv"
GLOSSARY_FILE = PROJECT_ROOT / "translation_glossary.csv"
GLOSSARY_SUGGESTIONS_FILE = PROJECT_ROOT / "translation_glossary_suggestions.csv"
WINDOWS_TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

DEFAULT_MODEL_URL = os.getenv("TINY_AYA_MODEL_URL", "http://127.0.0.1:8080/v1/chat/completions")
DEFAULT_VISION_OCR_URL = os.getenv("MINICPM_V_OCR_URL", "http://127.0.0.1:8081/v1/chat/completions")
TINY_AYA_MODEL_REF = os.getenv("TINY_AYA_MODEL_REF", "CohereLabs/tiny-aya-global-GGUF:Q4_K_M")
MINICPM_V_MODEL_REF = os.getenv("MINICPM_V_MODEL_REF", "openbmb/MiniCPM-V-4.6-gguf:Q4_K_M")

MEDIUM_CONFIDENCE_MIN_SCORE = 80
HIGH_CONFIDENCE_MIN_SCORE = 90
MANUAL_MATCH_MIN_SCORE = 90
OCR_MATCH_MIN_SCORE = 90
DEFAULT_IMAGE_IDENTIFICATION_SECONDS = 60
MIN_IMAGE_IDENTIFICATION_SECONDS = 10
MAX_IMAGE_IDENTIFICATION_SECONDS = 99
ROBOT_IMAGE_IDENTIFICATION_SECONDS = 90

ORIENTATION_NORMAL_FIRST = "Normal first"
ORIENTATION_MIRRORED_FIRST = "Mirrored first"
ORIENTATION_FULL_AUTO = "Full auto"
DEVICE_PHONE_OR_TABLET = "phone_or_tablet"
DEVICE_DESKTOP_OR_LAPTOP = "desktop_or_laptop"
IMAGE_SOURCE_UPLOAD = "upload"
IMAGE_SOURCE_WEBCAM = "webcam"


def normalize_image_timeout_seconds(timeout_seconds) -> int:
    try:
        seconds = int(timeout_seconds)
    except (TypeError, ValueError):
        seconds = DEFAULT_IMAGE_IDENTIFICATION_SECONDS
    return max(MIN_IMAGE_IDENTIFICATION_SECONDS, min(MAX_IMAGE_IDENTIFICATION_SECONDS, seconds))
