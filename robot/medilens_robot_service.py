"""
Small local HTTP service for Reachy Mini demos.

Run this on the laptop/desktop that has MiniCPM-V 4.6 and the MediLens
dependencies available. Reachy Mini can POST a base64 image to /identify and
receive a short English answer to speak.
"""

from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
import base64
import json
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medilens_core.config import DEFAULT_VISION_OCR_URL, ORIENTATION_FULL_AUTO, ROBOT_IMAGE_IDENTIFICATION_SECONDS
from medilens_core.models import is_port_reachable
from medilens_core.pipeline import identify_medicine_from_image
from medilens_core.robot_responses import medicine_response


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class MediLensRobotHandler(BaseHTTPRequestHandler):
    server_version = "MediLensRobotService/1.0"

    def do_GET(self) -> None:
        if self.path == "/health":
            _json_response(self, 200, {"ok": True, "service": "medilens-robot"})
            return
        if self.path == "/status":
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "service": "medilens-robot",
                    "minicpm_v_4_6_url": DEFAULT_VISION_OCR_URL,
                    "minicpm_v_4_6_reachable": is_port_reachable(DEFAULT_VISION_OCR_URL),
                },
            )
            return
        else:
            _json_response(self, 404, {"ok": False, "error": "not_found"})
            return

    def do_POST(self) -> None:
        if self.path != "/identify":
            _json_response(self, 404, {"ok": False, "error": "not_found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            request_body = self.rfile.read(content_length)
            payload = json.loads(request_body.decode("utf-8"))
            image_base64 = payload["image_base64"]
            timeout_seconds = int(payload.get("timeout_seconds", ROBOT_IMAGE_IDENTIFICATION_SECONDS))
            vision_ocr_url = payload.get("vision_ocr_url", DEFAULT_VISION_OCR_URL)
            orientation_mode = payload.get("orientation_mode", ORIENTATION_FULL_AUTO)
            max_vision_attempts = int(payload.get("max_vision_attempts_per_orientation", 2))

            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            result = identify_medicine_from_image(
                image,
                vision_ocr_url=vision_ocr_url,
                orientation_mode=orientation_mode,
                timeout_seconds=timeout_seconds,
                max_vision_attempts_per_orientation=max_vision_attempts,
            )
            response = {
                "ok": True,
                "spoken_response": medicine_response(result),
                "result": asdict(result),
            }
            _json_response(self, 200, response)
        except Exception as error:
            _json_response(
                self,
                500,
                {
                    "ok": False,
                    "error": str(error),
                    "spoken_response": "I do not know what this medicine is. Try the MediLens app on your device.",
                },
            )

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


def run_server(host: str = "0.0.0.0", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), MediLensRobotHandler)
    print(f"MediLens robot service listening on http://{host}:{port}")
    print("Health check: /health")
    print("Identify endpoint: POST /identify")
    server.serve_forever()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the MediLens robot API service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
