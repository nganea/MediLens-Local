"""
Quick check: does Reachy Mini stream its microphone audio to the laptop?

Run this with the robot on and connected. It listens to Reachy's microphone
for a few seconds and reports whether audio actually arrives and how loud it is.
Speak toward the robot while it runs.

    py -3 robot/test_reachy_mic.py --reachy-host reachy-mini.local

If it reports that the mic works, you can use the hands-free app with --mic reachy.
"""

import argparse
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether Reachy Mini streams its microphone audio.")
    parser.add_argument("--reachy-host", default="reachy-mini.local")
    parser.add_argument("--reachy-port", type=int, default=8000)
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()

    from reachy_mini import ReachyMini

    with ReachyMini(host=args.reachy_host, port=args.reachy_port) as mini:
        print(f"Connected. Listening to Reachy's microphone for {args.seconds:.0f}s -- speak to the robot now!")
        deadline = time.monotonic() + args.seconds
        chunks = 0
        frames = 0
        peak = 0.0
        while time.monotonic() < deadline:
            sample = mini.media.get_audio_sample()  # (N, 2) float32 @ 16 kHz, or None
            if sample is None:
                time.sleep(0.01)
                continue
            chunks += 1
            mono = sample.mean(axis=1)
            frames += len(mono)
            peak = max(peak, float(np.abs(mono).max()))

        print(f"\naudio chunks received: {chunks}")
        print(f"total audio: {frames} frames (~{frames / 16000:.1f}s at 16 kHz)")
        print(f"peak level: {peak:.4f}")
        if chunks == 0:
            print("\nRESULT: No audio from Reachy's microphone. The robot is not streaming mic audio.")
        elif peak < 0.005:
            print("\nRESULT: Stream is present but nearly silent -- the mic may be muted or not capturing.")
        else:
            print("\nRESULT: Reachy's microphone works! You can run the app with --mic reachy.")


if __name__ == "__main__":
    main()
