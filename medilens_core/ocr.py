from io import BytesIO
import base64
import shutil
import time

import pandas as pd
import pytesseract
import requests
from PIL import Image, ImageFilter, ImageOps

from .config import (
    OCR_MATCH_MIN_SCORE,
    ORIENTATION_FULL_AUTO,
    ORIENTATION_MIRRORED_FIRST,
    ORIENTATION_NORMAL_FIRST,
    WINDOWS_TESSERACT_PATH,
)
from .matching import find_best_match, clean_text


if shutil.which("tesseract") is None and WINDOWS_TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(WINDOWS_TESSERACT_PATH)


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
    tokens = [token for token in clean_text(text).split() if token]
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
        "If the image is mirrored, mentally unmirror it before reading. "
        "If there is no readable medicine label, say: unreadable."
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


def run_staged_vision_ocr(
    image: Image.Image,
    vision_ocr_url: str,
    medicines: pd.DataFrame,
    orientation_mode: str,
    deadline: float,
    max_attempts_per_orientation: int = 1,
):
    attempts = []
    for note, candidate_image in readable_image_candidates(image, orientation_mode):
        for attempt_index in range(max(1, int(max_attempts_per_orientation))):
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                break
            vision_text = run_vision_ocr(candidate_image, vision_ocr_url, timeout=max(1.0, remaining_seconds))
            vision_match = find_best_match(vision_text, medicines)
            attempts.append((vision_match["score"], note, candidate_image, vision_text, vision_match))
            if vision_match["score"] >= OCR_MATCH_MIN_SCORE:
                return candidate_image, note, vision_text, vision_match, attempts, False
            if vision_text.strip() and attempt_index == 0:
                break
        if time.monotonic() >= deadline:
            break

    if not attempts:
        empty_match = {"score": 0, "matched_name": "", "row": None, "confidence": "low"}
        return image, "", "", empty_match, attempts, True

    _score, note, candidate_image, vision_text, vision_match = max(attempts, key=lambda attempt: attempt[0])
    return candidate_image, note, vision_text, vision_match, attempts, time.monotonic() >= deadline
