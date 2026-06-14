"""
Reachy Mini adapter for MediLens.

This file is intentionally small: the medicine logic lives in medilens_core,
and this adapter only coordinates speech, camera capture, and movement.
Replace the methods in ReachyMiniHooks with the Reachy Mini SDK calls used by
your hackathon environment.
"""

from pathlib import Path
from io import BytesIO
import base64
import json
import sys
import threading
from urllib.request import Request, urlopen

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medilens_core.config import (
    DEFAULT_VISION_OCR_URL,
    ORIENTATION_FULL_AUTO,
    ORIENTATION_MIRRORED_FIRST,
    ORIENTATION_NORMAL_FIRST,
    ROBOT_IMAGE_IDENTIFICATION_SECONDS,
)
from medilens_core.pipeline import identify_medicine_from_image
from medilens_core.robot_responses import PICTURE_TAKEN, START_PROMPT, medicine_response


CUE_PHRASE = "what's this medicine for"


def command_matches_cue(transcript: str) -> bool:
    cleaned = " ".join((transcript or "").lower().replace("?", "").split())
    return "reachy" in cleaned and "medicine" in cleaned


class ReachyMiniHooks:
    def speak(self, text: str) -> None:
        print(f"Reachy says: {text}")

    def capture_image(self) -> Image.Image:
        raise NotImplementedError("Connect this to the Reachy Mini camera API.")

    def start_thinking_motion(self) -> None:
        print("Reachy starts thinking motion.")

    def stop_thinking_motion(self) -> None:
        print("Reachy stops thinking motion.")


class RemoteMediLensClient:
    def __init__(
        self,
        service_url: str,
        orientation_mode: str = ORIENTATION_FULL_AUTO,
        max_vision_attempts_per_orientation: int = 2,
    ):
        self.service_url = service_url.rstrip("/")
        self.orientation_mode = orientation_mode
        self.max_vision_attempts_per_orientation = max(1, int(max_vision_attempts_per_orientation))

    def identify_payload(self, image: Image.Image, timeout_seconds: int = ROBOT_IMAGE_IDENTIFICATION_SECONDS) -> dict:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        payload = {
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "timeout_seconds": timeout_seconds,
            "orientation_mode": self.orientation_mode,
            "max_vision_attempts_per_orientation": self.max_vision_attempts_per_orientation,
        }
        request = Request(
            f"{self.service_url}/identify",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout_seconds + 15) as response:
            return json.loads(response.read().decode("utf-8"))

    def identify(self, image: Image.Image, timeout_seconds: int = ROBOT_IMAGE_IDENTIFICATION_SECONDS) -> str:
        data = self.identify_payload(image, timeout_seconds=timeout_seconds)
        return data.get("spoken_response") or "I do not know what this medicine is. Try the MediLens app on your device."


def identify_once(
    hooks: ReachyMiniHooks,
    *,
    medilens_client: RemoteMediLensClient | None = None,
    vision_ocr_url: str = DEFAULT_VISION_OCR_URL,
    orientation_mode: str = ORIENTATION_FULL_AUTO,
    timeout_seconds: int = ROBOT_IMAGE_IDENTIFICATION_SECONDS,
    max_vision_attempts_per_orientation: int = 2,
    show_result: bool = False,
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
        if medilens_client is None:
            result = identify_medicine_from_image(
                image,
                vision_ocr_url=vision_ocr_url,
                orientation_mode=orientation_mode,
                timeout_seconds=timeout_seconds,
                max_vision_attempts_per_orientation=max_vision_attempts_per_orientation,
            )
            response_text = medicine_response(result)
        else:
            data = medilens_client.identify_payload(image, timeout_seconds=timeout_seconds)
            response_text = data.get("spoken_response") or "I do not know what this medicine is. Try the MediLens app on your device."
            if show_result:
                print(json.dumps(data, indent=2))
    finally:
        stop_motion.set()
        motion_thread.join(timeout=3)

    hooks.speak(response_text)


class ImageFileHooks(ReachyMiniHooks):
    def __init__(self, image_path: str):
        self.image_path = Path(image_path)

    def capture_image(self) -> Image.Image:
        return Image.open(self.image_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the MediLens Reachy Mini flow with an image file.")
    parser.add_argument("image", help="Path to a medicine label image.")
    parser.add_argument("--service-url", help="Optional desktop MediLens robot service URL, for example http://192.168.1.25:8765")
    parser.add_argument("--vision-ocr-url", default=DEFAULT_VISION_OCR_URL)
    parser.add_argument(
        "--orientation-mode",
        choices=[ORIENTATION_NORMAL_FIRST, ORIENTATION_MIRRORED_FIRST, ORIENTATION_FULL_AUTO],
        default=ORIENTATION_FULL_AUTO,
        help="Image orientation strategy. Full auto tries normal, mirrored, upside-down, and flipped candidates.",
    )
    parser.add_argument(
        "--max-vision-attempts",
        type=int,
        default=2,
        help="Maximum MiniCPM-V 4.6 attempts per orientation. A second attempt is used only after empty/weak output.",
    )
    parser.add_argument("--timeout", type=int, default=ROBOT_IMAGE_IDENTIFICATION_SECONDS)
    parser.add_argument("--show-result", action="store_true", help="Print the service response for debugging.")
    args = parser.parse_args()

    identify_once(
        ImageFileHooks(args.image),
        medilens_client=RemoteMediLensClient(
            args.service_url,
            args.orientation_mode,
            args.max_vision_attempts,
        ) if args.service_url else None,
        vision_ocr_url=args.vision_ocr_url,
        orientation_mode=args.orientation_mode,
        timeout_seconds=args.timeout,
        max_vision_attempts_per_orientation=args.max_vision_attempts,
        show_result=args.show_result,
    )
