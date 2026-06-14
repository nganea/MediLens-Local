"""
Reachy Mini adapter for MediLens.

This file is intentionally small: the medicine logic lives in medilens_core,
and this adapter only coordinates speech, camera capture, and movement.
Replace the methods in ReachyMiniHooks with the Reachy Mini SDK calls used by
your hackathon environment.
"""

from pathlib import Path
import threading

from PIL import Image

from medilens_core.config import DEFAULT_VISION_OCR_URL, ORIENTATION_NORMAL_FIRST, ROBOT_IMAGE_IDENTIFICATION_SECONDS
from medilens_core.pipeline import identify_medicine_from_image
from medilens_core.robot_responses import PICTURE_TAKEN, START_PROMPT, medicine_response


CUE_PHRASE = "what's this medicine for"


class ReachyMiniHooks:
    def speak(self, text: str) -> None:
        print(f"Reachy says: {text}")

    def capture_image(self) -> Image.Image:
        raise NotImplementedError("Connect this to the Reachy Mini camera API.")

    def start_thinking_motion(self) -> None:
        print("Reachy starts thinking motion.")

    def stop_thinking_motion(self) -> None:
        print("Reachy stops thinking motion.")


def identify_once(
    hooks: ReachyMiniHooks,
    *,
    vision_ocr_url: str = DEFAULT_VISION_OCR_URL,
    timeout_seconds: int = ROBOT_IMAGE_IDENTIFICATION_SECONDS,
) -> None:
    hooks.speak(START_PROMPT)
    image = hooks.capture_image()
    hooks.speak(PICTURE_TAKEN)

    stop_motion = threading.Event()

    def thinking_loop() -> None:
        hooks.start_thinking_motion()
        try:
            while not stop_motion.wait(1.5):
                hooks.start_thinking_motion()
        finally:
            hooks.stop_thinking_motion()

    motion_thread = threading.Thread(target=thinking_loop, daemon=True)
    motion_thread.start()
    try:
        result = identify_medicine_from_image(
            image,
            vision_ocr_url=vision_ocr_url,
            orientation_mode=ORIENTATION_NORMAL_FIRST,
            timeout_seconds=timeout_seconds,
        )
    finally:
        stop_motion.set()
        motion_thread.join(timeout=3)

    hooks.speak(medicine_response(result))


class ImageFileHooks(ReachyMiniHooks):
    def __init__(self, image_path: str):
        self.image_path = Path(image_path)

    def capture_image(self) -> Image.Image:
        return Image.open(self.image_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the MediLens Reachy Mini flow with an image file.")
    parser.add_argument("image", help="Path to a medicine label image.")
    parser.add_argument("--vision-ocr-url", default=DEFAULT_VISION_OCR_URL)
    parser.add_argument("--timeout", type=int, default=ROBOT_IMAGE_IDENTIFICATION_SECONDS)
    args = parser.parse_args()

    identify_once(
        ImageFileHooks(args.image),
        vision_ocr_url=args.vision_ocr_url,
        timeout_seconds=args.timeout,
    )
