from dataclasses import dataclass
import time

import pandas as pd
from PIL import Image

from .config import DEFAULT_VISION_OCR_URL, OCR_MATCH_MIN_SCORE, ORIENTATION_FULL_AUTO, ROBOT_IMAGE_IDENTIFICATION_SECONDS
from .database import load_medicines
from .matching import apply_manual_match_safety, capitalize_name, find_best_match
from .models import normalize_local_url


@dataclass
class MedicineIdentification:
    found: bool
    medicine_name: str = ""
    generic_name: str = ""
    matched_name: str = ""
    confidence: str = "low"
    score: int = 0
    common_uses: str = ""
    safety_warning: str = ""
    source_url: str = ""
    ocr_text: str = ""
    timed_out: bool = False
    message: str = ""
    vision_attempts: int = 0


def _display_medicine_name(match: dict) -> str:
    row = match["row"]
    if row is None:
        return ""
    matched_name = match["matched_name"]
    generic_name = row["generic_name"]
    if matched_name.lower() == generic_name.lower():
        return capitalize_name(generic_name)
    return f"{capitalize_name(matched_name)} / {capitalize_name(generic_name)}"


def _polish_warning_text(text: str) -> str:
    cleaned = " ".join(str(text).split()).strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _result_from_match(
    match: dict,
    ocr_text: str,
    timed_out: bool = False,
    vision_attempts: int = 0,
) -> MedicineIdentification:
    row = match["row"]
    if row is None or match["confidence"] == "low":
        return MedicineIdentification(
            found=False,
            confidence=match.get("confidence", "low"),
            score=int(match.get("score", 0)),
            ocr_text=ocr_text,
            timed_out=timed_out,
            message="I do not know what this medicine is. Try the MediLens app on your device.",
            vision_attempts=vision_attempts,
        )

    return MedicineIdentification(
        found=True,
        medicine_name=_display_medicine_name(match),
        generic_name=row["generic_name"],
        matched_name=match["matched_name"],
        confidence=match["confidence"],
        score=int(match["score"]),
        common_uses=f"It is used for {row['common_uses']}.",
        safety_warning=_polish_warning_text(row["safety_warning"]),
        source_url=row.get("source_url", ""),
        ocr_text=ocr_text,
        timed_out=timed_out,
        vision_attempts=vision_attempts,
    )


def identify_medicine_from_text(
    label_text: str,
    medicines: pd.DataFrame | None = None,
) -> MedicineIdentification:
    medicines = medicines if medicines is not None else load_medicines()
    match = apply_manual_match_safety(find_best_match(label_text or "", medicines))
    return _result_from_match(match, label_text or "")


def identify_medicine_from_image(
    image: Image.Image,
    *,
    medicines: pd.DataFrame | None = None,
    vision_ocr_url: str = DEFAULT_VISION_OCR_URL,
    orientation_mode: str = ORIENTATION_FULL_AUTO,
    timeout_seconds: int = ROBOT_IMAGE_IDENTIFICATION_SECONDS,
    max_vision_attempts_per_orientation: int = 2,
) -> MedicineIdentification:
    from .ocr import run_staged_vision_ocr

    medicines = medicines if medicines is not None else load_medicines()
    vision_ocr_url = normalize_local_url(vision_ocr_url, DEFAULT_VISION_OCR_URL)
    timeout_seconds = max(1, int(timeout_seconds))
    deadline = time.monotonic() + timeout_seconds

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")

    try:
        selected_image, orientation_note, vision_text, vision_match, attempts, timed_out = run_staged_vision_ocr(
            image,
            vision_ocr_url,
            medicines,
            orientation_mode,
            deadline,
            max_attempts_per_orientation=max_vision_attempts_per_orientation,
        )
    except Exception as error:
        return MedicineIdentification(
            found=False,
            message="I do not know what this medicine is. Try the MediLens app on your device.",
            ocr_text=f"MiniCPM-V 4.6 failed: {error}",
        )

    vision_attempts = len(attempts)
    ocr_text = "\n\n".join(part for part in [orientation_note, vision_text] if part)
    if not ocr_text:
        ocr_text = "MiniCPM-V 4.6 returned no readable text."
    if vision_match["score"] < OCR_MATCH_MIN_SCORE:
        try:
            from .ocr import choose_readable_image_orientation

            fallback_image, tesseract_text, fallback_note = choose_readable_image_orientation(
                image,
                medicines,
                orientation_mode,
            )
            selected_image = fallback_image
            tesseract_match = find_best_match(tesseract_text, medicines)
            if fallback_note:
                ocr_text = f"{ocr_text}\n\n{fallback_note}"
            if tesseract_text:
                ocr_text = f"{ocr_text}\n\nTesseract OCR:\n{tesseract_text}"
            if tesseract_match["score"] >= OCR_MATCH_MIN_SCORE:
                return _result_from_match(
                    tesseract_match,
                    ocr_text,
                    timed_out=timed_out,
                    vision_attempts=vision_attempts,
                )
        except Exception as error:
            ocr_text = f"{ocr_text}\n\nTesseract fallback failed: {error}"
        vision_match["confidence"] = "low"
    return _result_from_match(
        vision_match,
        ocr_text,
        timed_out=timed_out,
        vision_attempts=vision_attempts,
    )


def spoken_response(result: MedicineIdentification) -> str:
    if not result.found:
        return result.message or "I do not know what this medicine is. Try the MediLens app on your device."

    warning = result.safety_warning
    return f"It looks like {result.medicine_name}. {result.common_uses} Speak with a pharmacist or doctor if you are not sure it is safe for you. {warning}"
