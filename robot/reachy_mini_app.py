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
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import wave
from urllib.request import Request, urlopen

from PIL import Image

# Make console output safe for non-ASCII (e.g. Romanian diacritics) on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# Natural offline neural voice (Kokoro via ONNX). Loaded once and reused.
_VOICES_DIR = Path(__file__).resolve().parent / "voices"
KOKORO_MODEL_PATH = _VOICES_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = _VOICES_DIR / "voices-v1.0.bin"
KOKORO_VOICE = "bf_emma"  # warm British English female
KOKORO_LANG = "en-gb"
_kokoro = None


def _get_kokoro():
    """Load the Kokoro model once and cache it."""
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro

        _kokoro = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))
    return _kokoro


def _synthesize_with_kokoro(text: str, wav_path: Path) -> None:
    import numpy as np

    kokoro = _get_kokoro()
    samples, sample_rate = kokoro.create(text, voice=KOKORO_VOICE, speed=1.0, lang=KOKORO_LANG)
    pcm = (np.clip(np.asarray(samples), -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


# Non-English neural voices (Piper). English uses Kokoro; every other language
# in the app uses its Piper voice. Each voice is loaded once and cached.
PIPER_VOICE_FILES = {
    "Romanian": "ro_RO-mihai-medium.onnx",
    "German": "de_DE-thorsten-medium.onnx",
    "French": "fr_FR-siwis-medium.onnx",
    "Italian": "it_IT-paola-medium.onnx",
    "Spanish": "es_ES-davefx-medium.onnx",
}
_piper_cache: dict[str, object] = {}


def _get_piper(language: str):
    if language not in _piper_cache:
        from piper import PiperVoice

        path = _VOICES_DIR / PIPER_VOICE_FILES[language]
        _piper_cache[language] = PiperVoice.load(str(path))
    return _piper_cache[language]


def _synthesize_with_piper(text: str, wav_path: Path, language: str) -> None:
    voice = _get_piper(language)
    with wave.open(str(wav_path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)


def _synthesize_with_sapi(text: str, wav_path: Path) -> None:
    # Built-in Windows TTS fallback: robotic but always available offline.
    text_path = wav_path.with_suffix(".txt")
    text_path.write_text(text, encoding="utf-8")
    powershell_script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{wav_path}'); "
        f"$t = Get-Content -Raw -Encoding UTF8 -Path '{text_path}'; "
        "$s.Speak($t); "
        "$s.Dispose()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", powershell_script],
        check=True,
        capture_output=True,
    )


def synthesize_speech_to_wav(text: str, language: str = "English") -> Path:
    """Turn text into a WAV file using offline TTS, in the given language.

    English uses the natural Kokoro voice; Romanian uses the Piper Romanian
    voice. Both fall back to built-in Windows SAPI if the neural voice fails.
    No internet is required for any path. Returns a temporary WAV file path.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="medilens_tts_"))
    wav_path = temp_dir / "speech.wav"
    try:
        if language in PIPER_VOICE_FILES:
            _synthesize_with_piper(text, wav_path, language)
        else:
            _synthesize_with_kokoro(text, wav_path)
    except Exception as error:
        print(f"(Neural voice unavailable, using basic Windows voice: {error})")
        _synthesize_with_sapi(text, wav_path)
    return wav_path


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
    return frames / float(rate) if rate else 0.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medilens_core.config import (
    DEFAULT_MODEL_URL,
    DEFAULT_VISION_OCR_URL,
    ORIENTATION_FULL_AUTO,
    ORIENTATION_MIRRORED_FIRST,
    ORIENTATION_NORMAL_FIRST,
    ROBOT_IMAGE_IDENTIFICATION_SECONDS,
)
from medilens_core.explanations import call_local_chat_model
from medilens_core.pipeline import identify_medicine_from_image
from medilens_core.robot_responses import medicine_response


CUE_PHRASE = "what's this medicine for"

# Whisper mishears the made-up name "Reachy" differently per model/language
# ("Reiki", "Riki", "Richie", ...). Match on these word prefixes so the wake
# word still triggers. The required "medicine" word keeps false positives low.
REACHY_NAME_PREFIXES = (
    "reach",
    "reak",
    "reik",
    "reek",
    "riki",
    "rikk",
    "ricky",
    "rich",
    "ritchie",
    "riichi",
    "ricci",
)


# Word stems for "medicine" across all supported languages. We match these as
# substrings of the accent-stripped transcript, which naturally handles:
#  - enclitic (suffixed) articles: RO "medicamentul", "medicamentului", "medicamentele"
#  - plurals/inflections: "medicamente", "medicamentos", "Medikamente", "medicinali"
#  - separate-article languages (DE/FR/IT/ES) where the noun is its own word
MEDICINE_STEMS = (
    "medicine",     # EN
    "medication",   # EN
    "medicament",   # RO, ES, FR (médicament -> medicament after accent strip)
    "medikament",   # DE
    "medicin",      # EN medicine, ES/IT/RO medicina, medicinale
    "mediz",        # DE Medizin
    "medecin",      # FR médecine
    "farmac",       # IT/ES farmaco / fármaco
)


def _normalize(text: str) -> str:
    """Lowercase and strip accents so diacritics do not block matching."""
    lowered = (text or "").lower()
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def command_matches_cue(transcript: str) -> bool:
    cleaned = _normalize(transcript).replace("?", "").replace(",", "")
    words = cleaned.split()
    has_name = any(word.startswith(REACHY_NAME_PREFIXES) for word in words)
    has_medicine = any(stem in cleaned for stem in MEDICINE_STEMS)
    return has_name and has_medicine


# Stop-word prefixes across the supported languages (whole-word prefix match, so
# they do not collide with medicine names like "paracetamol"). EN/IT/DE "stop",
# RO "oprește/opriți", FR "arrête", IT "basta/ferma", DE "halt/Schluss", RO "gata".
STOP_PREFIXES = (
    "stop",
    "oprest",
    "opri",
    "opreste",
    "arret",
    "basta",
    "ferma",
    "gata",
    "halt",
    "schluss",
    "aufhor",
)


def command_matches_stop(transcript: str) -> bool:
    cleaned = _normalize(transcript).replace("?", "").replace(",", "").replace("!", "")
    words = cleaned.split()
    has_name = any(word.startswith(REACHY_NAME_PREFIXES) for word in words)
    has_stop = any(word.startswith(STOP_PREFIXES) for word in words)
    return has_name and has_stop


# Fixed spoken lines per language. Only the dynamic medicine answer is machine
# translated (via Tiny Aya); these short prompts are pre-translated for speed.
ROBOT_PHRASES = {
    "English": {
        "start": "Okay, hold the medicine label in front of me.",
        "picture": "I have taken a picture. I am checking it now.",
        "listening": "I am listening. When you are ready, ask me what a medicine is.",
        "okay": "Okay.",
        "ready_again": "Ask me about another medicine whenever you are ready.",
        "goodbye": "Okay, I will stop now. Goodbye.",
        "trouble": "Sorry, I could not read the medicine. Please try again.",
    },
    "Romanian": {
        "start": "Bine, țineți eticheta medicamentului în fața mea.",
        "picture": "Am făcut o poză. O verific acum.",
        "listening": "Vă ascult. Când sunteți gata, întrebați-mă ce este un medicament.",
        "okay": "Bine.",
        "ready_again": "Întrebați-mă despre alt medicament când sunteți gata.",
        "goodbye": "Bine, mă opresc acum. La revedere.",
        "trouble": "Îmi pare rău, nu am putut citi medicamentul. Vă rog încercați din nou.",
    },
    "German": {
        "start": "Okay, halte das Etikett des Medikaments vor mich.",
        "picture": "Ich habe ein Foto gemacht. Ich überprüfe es jetzt.",
        "listening": "Ich höre zu. Wenn du bereit bist, frag mich, was ein Medikament ist.",
        "okay": "Okay.",
        "ready_again": "Frag mich nach einem anderen Medikament, wann immer du bereit bist.",
        "goodbye": "Okay, ich höre jetzt auf. Auf Wiedersehen.",
        "trouble": "Entschuldigung, ich konnte das Medikament nicht lesen. Bitte versuche es noch einmal.",
    },
    "French": {
        "start": "D'accord, tenez l'étiquette du médicament devant moi.",
        "picture": "J'ai pris une photo. Je la vérifie maintenant.",
        "listening": "Je vous écoute. Quand vous êtes prêt, demandez-moi ce qu'est un médicament.",
        "okay": "D'accord.",
        "ready_again": "Demandez-moi un autre médicament quand vous voulez.",
        "goodbye": "D'accord, j'arrête maintenant. Au revoir.",
        "trouble": "Désolé, je n'ai pas pu lire le médicament. Veuillez réessayer.",
    },
    "Italian": {
        "start": "Va bene, tieni l'etichetta del medicinale davanti a me.",
        "picture": "Ho scattato una foto. La sto controllando adesso.",
        "listening": "Ti ascolto. Quando sei pronto, chiedimi che cos'è un medicinale.",
        "okay": "Va bene.",
        "ready_again": "Chiedimi di un altro medicinale quando vuoi.",
        "goodbye": "Va bene, mi fermo adesso. Arrivederci.",
        "trouble": "Scusa, non sono riuscita a leggere il medicinale. Riprova per favore.",
    },
    "Spanish": {
        "start": "De acuerdo, sostén la etiqueta del medicamento frente a mí.",
        "picture": "He tomado una foto. La estoy revisando ahora.",
        "listening": "Te escucho. Cuando estés listo, pregúntame qué es un medicamento.",
        "okay": "De acuerdo.",
        "ready_again": "Pregúntame por otro medicamento cuando quieras.",
        "goodbye": "De acuerdo, me detengo ahora. Adiós.",
        "trouble": "Lo siento, no pude leer el medicamento. Por favor, inténtalo de nuevo.",
    },
}


def phrase(key: str, language: str) -> str:
    table = ROBOT_PHRASES.get(language, ROBOT_PHRASES["English"])
    return table[key]


def translate_text(text: str, language: str, model_url: str = DEFAULT_MODEL_URL) -> str:
    """Translate English robot text into the target language via Tiny Aya.

    Returns the original English text if the language is English or if the
    Tiny Aya server cannot be reached, so the demo never stalls on translation.
    """
    if language == "English" or not text:
        return text
    prompt = (
        f"Translate the following message into {language}. "
        "Keep it short and natural for speaking aloud. Keep medicine names unchanged. "
        f"Return only the translation, nothing else.\n\n{text}"
    )
    try:
        translated = call_local_chat_model(model_url, prompt, max_tokens=160).strip()
    except Exception as error:
        print(f"(Translation unavailable, speaking English: {error})")
        return text
    return translated or text


class ReachyMiniHooks:
    def speak(self, text: str, language: str = "English") -> None:
        print(f"Reachy says: {text}")

    def capture_image(self) -> Image.Image:
        raise NotImplementedError("Connect this to the Reachy Mini camera API.")

    def start_thinking_motion(self) -> None:
        print("Reachy starts thinking motion.")

    def stop_thinking_motion(self) -> None:
        print("Reachy stops thinking motion.")

    def ready_gesture(self) -> None:
        print("Reachy wiggles its antennas (ready to listen).")

    def stop_gesture(self) -> None:
        print("Reachy lowers its antennas (stopping).")


class ReachySdkHooks(ReachyMiniHooks):
    """Reachy Mini SDK hooks for camera capture and simple head movement."""

    def __init__(self, mini):
        self.mini = mini
        self._thinking_step = 0

    def speak(self, text: str, language: str = "English") -> None:
        print(f"Reachy says: {text}")
        try:
            wav_path = synthesize_speech_to_wav(text, language=language)
        except Exception as error:
            print(f"(TTS synthesis failed, showing text only: {error})")
            return
        try:
            self.mini.media.play_sound(str(wav_path))
            # Block while the clip plays so phrases do not overlap. A small
            # buffer covers upload/playback start latency on the robot.
            time.sleep(wav_duration_seconds(wav_path) + 0.5)
        except Exception as error:
            print(f"(Reachy audio playback failed, text only: {error})")

    def capture_image(self, warmup_seconds: float = 8.0, poll_interval: float = 0.2) -> Image.Image:
        # The camera stream needs a moment after connecting before the first
        # frame arrives, so poll get_frame() until we get a real frame.
        deadline = time.monotonic() + warmup_seconds
        frame = None
        while time.monotonic() < deadline:
            frame = self.mini.media.get_frame()
            if frame is not None:
                break
            time.sleep(poll_interval)
        if frame is None:
            raise RuntimeError(
                "Reachy Mini camera returned no frame. "
                "Check the camera is connected and the media stream is running."
            )
        return Image.fromarray(frame).convert("RGB")

    def start_thinking_motion(self) -> None:
        from reachy_mini.utils import create_head_pose
        import numpy as np

        self._thinking_step += 1
        yaw_degrees = 8 if self._thinking_step % 2 else -8
        pitch_degrees = 5 if self._thinking_step % 3 else -3
        self.mini.goto_target(
            head=create_head_pose(
                pitch=np.deg2rad(pitch_degrees),
                yaw=np.deg2rad(yaw_degrees),
                degrees=False,
                mm=False,
            ),
            duration=0.6,
            method="minjerk",
        )

    def stop_thinking_motion(self) -> None:
        from reachy_mini.utils import create_head_pose

        self.mini.goto_target(
            head=create_head_pose(),
            duration=0.8,
            method="minjerk",
        )

    # Antenna resting pose (~10 deg), matching the SDK's init position.
    ANTENNAS_NEUTRAL = [-0.1745, 0.1745]

    def ready_gesture(self) -> None:
        """Perk and waggle the antennas to signal 'ready to listen'."""
        up = [0.6, -0.6]
        try:
            for pose in (up, self.ANTENNAS_NEUTRAL, up, self.ANTENNAS_NEUTRAL):
                self.mini.goto_target(antennas=pose, duration=0.3, method="minjerk")
        except Exception as error:
            print(f"(antenna ready gesture failed: {error})")

    def stop_gesture(self) -> None:
        """Lower the antennas as a 'goodbye / stopping' signal."""
        droop = [-1.2, 1.2]
        try:
            self.mini.goto_target(antennas=droop, duration=0.7, method="minjerk")
            self.mini.goto_target(antennas=self.ANTENNAS_NEUTRAL, duration=0.7, method="minjerk")
        except Exception as error:
            print(f"(antenna stop gesture failed: {error})")


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
    language: str = "English",
) -> None:
    hooks.speak(phrase("start", language), language=language)
    image = hooks.capture_image()
    hooks.speak(phrase("picture", language), language=language)

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

    hooks.speak(translate_text(response_text, language), language=language)


def run_listen_loop(
    hooks: ReachyMiniHooks,
    *,
    identify_kwargs: dict,
    model_size: str = "base.en",
    input_device=None,
    language: str = "en",
) -> None:
    """Listen on the laptop microphone and run the identify flow on the cue."""
    from voice_listener import VoiceListener

    print(f"Loading speech recognition (language={language})...")
    listener = VoiceListener(
        model_size=model_size,
        input_device=input_device,
        language=language,
    )

    def announce_ready():
        # Fires only after the mic is calibrated, so this is the real "go ahead".
        hooks.ready_gesture()  # antennas waggle to show it is ready
        hooks.speak(phrase("listening", "English"))
        print(
            "\nReady. Ask: \"Reachy, what is this medicine?\"  "
            "Say \"Reachy, stop\" to finish.\n"
        )

    print("Loading and calibrating... please wait for Reachy to say it is listening.")
    for text in listener.listen(on_ready=announce_ready):
        spoken_language = listener.last_language
        print(f"Heard [{spoken_language}]: {text!r}")

        if command_matches_stop(text):
            listener.muted = True
            hooks.speak(phrase("goodbye", spoken_language), language=spoken_language)
            hooks.stop_gesture()  # antennas lower as a goodbye
            print("Stop command heard. Goodbye.")
            break

        if command_matches_cue(text):
            listener.muted = True  # ignore Reachy's own voice while it answers
            try:
                hooks.speak(phrase("okay", spoken_language), language=spoken_language)
                identify_once(hooks, language=spoken_language, **identify_kwargs)
                hooks.speak(phrase("ready_again", spoken_language), language=spoken_language)
            except Exception as error:
                # A slow model, timeout, or service hiccup must not end the
                # session. Apologise and keep listening for the next question.
                print(f"(identify failed: {error})")
                hooks.speak(phrase("trouble", spoken_language), language=spoken_language)
            finally:
                hooks.ready_gesture()  # waggle again: ready for the next question
                listener.flush()
                listener.muted = False
                print("Listening again... (say \"Reachy, stop\" to finish)")


class ImageFileHooks(ReachyMiniHooks):
    def __init__(self, image_path: str):
        self.image_path = Path(image_path)

    def capture_image(self) -> Image.Image:
        return Image.open(self.image_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the MediLens Reachy Mini flow with an image file.")
    parser.add_argument("image", nargs="?", help="Path to a medicine label image.")
    parser.add_argument("--service-url", help="Optional desktop MediLens robot service URL, for example http://192.168.1.25:8765")
    parser.add_argument("--vision-ocr-url", default=DEFAULT_VISION_OCR_URL)
    parser.add_argument("--use-reachy", action="store_true", help="Capture the image from Reachy Mini instead of an image file.")
    parser.add_argument(
        "--reachy-media-backend",
        default="default",
        choices=["default", "local", "webrtc", "no_media"],
        help="Reachy Mini SDK media backend. Use default unless you need to force a backend.",
    )
    parser.add_argument(
        "--reachy-connection-mode",
        choices=["localhost_only", "network"],
        help="Optional Reachy Mini SDK connection mode. Omit this to let the SDK auto-detect.",
    )
    parser.add_argument(
        "--reachy-host",
        default="reachy-mini.local",
        help="Reachy Mini daemon host or IP address. Defaults to reachy-mini.local.",
    )
    parser.add_argument(
        "--reachy-port",
        type=int,
        default=8000,
        help="Reachy Mini daemon port. Defaults to 8000.",
    )
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
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Hands-free mode: listen on the laptop microphone and run the identify flow when you say the cue (Reachy + medicine). Requires --use-reachy.",
    )
    parser.add_argument(
        "--whisper-model",
        default="base.en",
        help="faster-whisper model size for speech recognition (e.g. tiny.en, base.en, small.en).",
    )
    parser.add_argument(
        "--language",
        choices=["en", "ro", "de", "fr", "it", "es", "auto"],
        default="en",
        help="Spoken language for hands-free mode: en (English, fast); ro/de/fr/it/es (that language); auto (detect each question). Non-English answers use a multilingual model and need Tiny Aya running on 8080 for translation.",
    )
    parser.add_argument(
        "--input-device",
        help="Optional microphone name or index for listening. Defaults to the system default input.",
    )
    args = parser.parse_args()

    if args.listen and not args.use_reachy:
        parser.error("--listen requires --use-reachy (hands-free mode drives the robot).")

    input_device = args.input_device
    if input_device is not None and input_device.isdigit():
        input_device = int(input_device)

    client = RemoteMediLensClient(
        args.service_url,
        args.orientation_mode,
        args.max_vision_attempts,
    ) if args.service_url else None

    if args.use_reachy:
        from reachy_mini import ReachyMini

        reachy_kwargs = {
            "host": args.reachy_host,
            "port": args.reachy_port,
            "media_backend": args.reachy_media_backend,
        }
        if args.reachy_connection_mode:
            reachy_kwargs["connection_mode"] = args.reachy_connection_mode

        identify_kwargs = dict(
            medilens_client=client,
            vision_ocr_url=args.vision_ocr_url,
            orientation_mode=args.orientation_mode,
            timeout_seconds=args.timeout,
            max_vision_attempts_per_orientation=args.max_vision_attempts,
            show_result=args.show_result,
        )
        with ReachyMini(**reachy_kwargs) as mini:
            hooks = ReachySdkHooks(mini)
            if args.listen:
                try:
                    run_listen_loop(
                        hooks,
                        identify_kwargs=identify_kwargs,
                        model_size=args.whisper_model,
                        input_device=input_device,
                        language=args.language,
                    )
                except KeyboardInterrupt:
                    print("\nStopped listening.")
            else:
                identify_once(hooks, **identify_kwargs)
            time.sleep(0.5)
    else:
        if not args.image:
            parser.error("image is required unless --use-reachy is set")

        identify_once(
            ImageFileHooks(args.image),
            medilens_client=client,
            vision_ocr_url=args.vision_ocr_url,
            orientation_mode=args.orientation_mode,
            timeout_seconds=args.timeout,
            max_vision_attempts_per_orientation=args.max_vision_attempts,
            show_result=args.show_result,
        )
