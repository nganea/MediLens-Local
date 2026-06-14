"""
Offline microphone listener for the MediLens Reachy Mini demo.

Captures audio from the laptop microphone, splits it into spoken utterances
with a simple energy-based voice-activity detector, and transcribes each
utterance with faster-whisper. Everything runs locally; no internet is needed
after the Whisper model has been downloaded once.

Reachy Mini's own microphone is not used here because the SDK mic capture
requires a local "Reachy Mini Audio USB device" that is not present in this
setup (hence the "No Reachy Mini Audio USB device found!" warning at startup).
"""

from collections import deque
import queue
import sys
import time

import numpy as np

# Make console output safe for non-ASCII (e.g. Romanian diacritics) on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SAMPLE_RATE = 16000  # Whisper's native rate
FRAME_SECONDS = 0.03  # 30 ms analysis frames
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_SECONDS)

# Whisper guesses unusual words badly, so bias it toward our vocabulary.
WHISPER_PROMPT = "Reachy is a small robot. Conversation about Reachy and what a medicine is."
# Short hint for multilingual/auto mode: the name plus the medicine word across
# the supported languages, so it spells them well without forcing a language.
MULTILINGUAL_HINT = "Reachy. medicine. medicament. Medikament. médicament. medicina."


def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame))) + 1e-9)


class VoiceListener:
    def __init__(
        self,
        model_size: str = "base.en",
        end_silence_seconds: float = 1.4,
        min_speech_seconds: float = 0.5,
        max_utterance_seconds: float = 14.0,
        start_factor: float = 3.0,
        floor_threshold: float = 0.02,
        input_device=None,
        language: str = "en",
        multilingual_model: str = "small",
        reachy_media=None,
    ):
        # When reachy_media is set, listen through Reachy Mini's own microphone
        # (any object exposing get_audio_sample() -> (N,2) float32 @ 16 kHz)
        # instead of the laptop microphone.
        self.reachy_media = reachy_media
        self.end_silence_frames = int(end_silence_seconds / FRAME_SECONDS)
        self.min_speech_frames = int(min_speech_seconds / FRAME_SECONDS)
        self.max_utterance_frames = int(max_utterance_seconds / FRAME_SECONDS)
        self.start_factor = start_factor
        self.floor_threshold = floor_threshold
        self.input_device = input_device
        self._audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._silence_rms = 0.0
        self.muted = False  # when True, incoming audio is dropped (e.g. while Reachy talks)
        self.language = language  # "en", "ro", or "auto"
        self.last_language = "English"  # detected language name of the last utterance
        self.debug = False  # when True, print calibration + speech-activity diagnostics

        from faster_whisper import WhisperModel

        # English-only mode keeps the fast *.en model; bilingual/auto needs
        # a multilingual model (e.g. "small") that supports Romanian.
        chosen = model_size if language == "en" else multilingual_model
        self.model = WhisperModel(chosen, device="cpu", compute_type="int8")

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            print(f"(microphone status: {status})", file=sys.stderr)
        if self.muted:
            return
        self._audio_queue.put(indata[:, 0].copy())

    def flush(self) -> None:
        """Drop any audio buffered so far (used after Reachy finishes talking)."""
        if self.reachy_media is not None:
            # Drain Reachy's audio appsink backlog (e.g. its own voice).
            for _ in range(5000):
                if self.reachy_media.get_audio_sample() is None:
                    break
            return
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def _frames(self):
        """Yield fixed-size mono frames from the laptop microphone queue."""
        buffer = np.zeros(0, dtype=np.float32)
        while True:
            buffer = np.concatenate([buffer, self._audio_queue.get()])
            while len(buffer) >= FRAME_SAMPLES:
                yield buffer[:FRAME_SAMPLES]
                buffer = buffer[FRAME_SAMPLES:]

    def _reachy_frames(self):
        """Yield fixed-size mono frames from Reachy Mini's microphone.

        Polls get_audio_sample() (stereo float32 @ 16 kHz), mixes to mono, and
        chunks into analysis frames. Drops audio while muted (e.g. while Reachy
        is speaking) so the robot does not transcribe its own voice.
        """
        buffer = np.zeros(0, dtype=np.float32)
        while True:
            sample = self.reachy_media.get_audio_sample()
            if sample is None:
                time.sleep(0.01)
                continue
            if self.muted:
                buffer = np.zeros(0, dtype=np.float32)
                continue
            mono = np.asarray(sample, dtype=np.float32).mean(axis=1)
            buffer = np.concatenate([buffer, mono])
            while len(buffer) >= FRAME_SAMPLES:
                yield buffer[:FRAME_SAMPLES]
                buffer = buffer[FRAME_SAMPLES:]

    _LANG_NAME = {
        "en": "English",
        "ro": "Romanian",
        "de": "German",
        "fr": "French",
        "it": "Italian",
        "es": "Spanish",
    }

    def transcribe(self, utterance: np.ndarray) -> str:
        # Force a language ("en"/"ro") or let Whisper detect it ("auto").
        forced = None if self.language == "auto" else self.language
        # Use the full English hint in English mode; a short multilingual hint
        # otherwise so it spells "Reachy"/medicine words without strongly
        # biasing language detection.
        prompt = WHISPER_PROMPT if self.language == "en" else MULTILINGUAL_HINT
        segments, info = self.model.transcribe(
            utterance.astype(np.float32),
            language=forced,
            initial_prompt=prompt,
            vad_filter=True,
        )
        self.last_language = self._LANG_NAME.get(info.language, "English")
        # Drop low-confidence / non-speech segments. These are the source of
        # Whisper hallucinations ("thank you", "I don't need one") on noise.
        kept = [
            segment.text
            for segment in segments
            if segment.no_speech_prob < 0.6 and segment.avg_logprob > -1.0
        ]
        return " ".join(kept).strip()

    def listen(self, on_ready=None, should_stop=None, on_speech_start=None, on_transcribing=None):
        """Yield transcribed text for each spoken utterance, forever.

        on_ready: optional callback invoked once, after the microphone has been
        calibrated, to signal that the system is ready to be spoken to. The
        caller must stay quiet until on_ready fires (calibration needs silence).
        should_stop: optional callback checked between audio frames.
        on_speech_start: optional callback invoked when speech crosses the
        voice-activity threshold.
        on_transcribing: optional callback invoked when an utterance is ready
        for speech-to-text.
        """
        import contextlib

        if self.reachy_media is not None:
            frames = self._reachy_frames()
            source_context = contextlib.nullcontext()
        else:
            import sounddevice as sd

            source_context = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME_SAMPLES,
                device=self.input_device,
                callback=self._callback,
            )
            frames = self._frames()

        with source_context:
            # Calibrate the silence floor from the first ~1 second of audio.
            # Stay quiet during this; the caller is told to wait via on_ready.
            print("[listener] calibrating microphone, please stay quiet...", file=sys.stderr)
            calibration = [_rms(next(frames)) for _ in range(int(1.0 / FRAME_SECONDS))]
            self._silence_rms = float(np.median(calibration))
            start_threshold = max(self._silence_rms * self.start_factor, self.floor_threshold)
            print(
                f"[listener] calibrated: silence={self._silence_rms:.4f} "
                f"trigger threshold={start_threshold:.4f}",
                file=sys.stderr,
            )

            if on_ready is not None:
                on_ready()
                # Drop anything captured while the ready cue played (e.g. Reachy's
                # own prompt) so it is not mistaken for the user's first request.
                self.flush()

            collecting = False
            collected: list[np.ndarray] = []
            preroll: "deque[np.ndarray]" = deque(maxlen=5)
            silence_run = 0
            idle_peak = 0.0
            idle_frames = 0

            for frame in frames:
                if should_stop is not None and should_stop():
                    return

                level = _rms(frame)
                if not collecting:
                    preroll.append(frame)
                    if self.debug:
                        idle_peak = max(idle_peak, level)
                        idle_frames += 1
                        if idle_frames >= int(1.0 / FRAME_SECONDS):
                            print(
                                f"[listener] idle peak last 1s={idle_peak:.4f} "
                                f"(need >= {start_threshold:.4f} to trigger)",
                                file=sys.stderr,
                            )
                            idle_peak = 0.0
                            idle_frames = 0
                    if level >= start_threshold:
                        collecting = True
                        collected = list(preroll)
                        silence_run = 0
                        if on_speech_start is not None:
                            on_speech_start()
                        if self.debug:
                            print(f"[listener] speech started (level={level:.4f})", file=sys.stderr)
                    continue

                collected.append(frame)
                silence_run = silence_run + 1 if level < start_threshold else 0

                utterance_done = (
                    silence_run >= self.end_silence_frames
                    or len(collected) >= self.max_utterance_frames
                )
                if utterance_done:
                    speech_frames = len(collected) - silence_run
                    if self.debug:
                        print(
                            f"[listener] speech ended: {len(collected) * FRAME_SECONDS:.1f}s total, "
                            f"{speech_frames * FRAME_SECONDS:.1f}s speech -> transcribing",
                            file=sys.stderr,
                        )
                    if speech_frames >= self.min_speech_frames:
                        utterance = np.concatenate(collected)
                        if on_transcribing is not None:
                            on_transcribing()
                        text = self.transcribe(utterance)
                        if text:
                            yield text
                    collecting = False
                    collected = []
                    preroll.clear()
                    silence_run = 0


if __name__ == "__main__":
    # Standalone microphone test: prints what it hears. No robot needed.
    import argparse

    parser = argparse.ArgumentParser(description="Test the microphone + speech recognition.")
    parser.add_argument("--whisper-model", default="base.en")
    parser.add_argument(
        "--language",
        choices=["en", "ro", "de", "fr", "it", "es", "auto"],
        default="en",
        help="en = English only (fast); ro/de/fr/it/es = that language; auto = detect each phrase (uses multilingual model).",
    )
    parser.add_argument("--input-device", help="Microphone name or index (default: system default).")
    parser.add_argument("--debug", action="store_true", help="Print calibration and speech-activity diagnostics.")
    args = parser.parse_args()

    device = args.input_device
    if device is not None and device.isdigit():
        device = int(device)

    print(f"Loading speech recognition (whisper, language={args.language})...")
    listener = VoiceListener(
        model_size=args.whisper_model,
        input_device=device,
        language=args.language,
    )
    listener.debug = args.debug

    # Show whether each phrase would trigger the robot (the "Reachy + medicine" cue).
    try:
        from reachy_mini_app import command_matches_cue
    except Exception:
        command_matches_cue = None

    def announce_ready():
        # '\a' rings the terminal bell so you get an audible "go ahead" too.
        print("\a")
        print("=" * 50)
        print("  READY - SPEAK NOW")
        print('  Try: "Reachy, what is this medicine?"')
        print("=" * 50, flush=True)

    print("Loading and calibrating microphone... please stay quiet for a moment.")
    try:
        for heard in listener.listen(on_ready=announce_ready):
            cue = ""
            if command_matches_cue is not None:
                cue = "  <<< CUE MATCH (would start the robot)" if command_matches_cue(heard) else "  (no cue)"
            print(f"Heard: [{listener.last_language}] {heard!r}{cue}")
    except KeyboardInterrupt:
        print("\nStopped.")
