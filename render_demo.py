"""
Render a short chord progression to a WAV file, so you can hear the guitar
sound without a webcam:

    python render_demo.py               # writes demo.wav
    python render_demo.py --play        # also plays it through the speakers
"""
import argparse
import wave

import numpy as np

from audio_player import AudioPlayer
from config import SAMPLE_RATE

BPM = 96
# One bar per chord; D = down strum, U = up strum, - = rest (eighth notes)
PATTERN = "D-DU-UDU"
PROGRESSION = ["C", "G", "Am", "F", "C", "G", "Em", "D", "OPEN"]


def render(progression=PROGRESSION, pattern=PATTERN, bpm=BPM) -> np.ndarray:
    player = AudioPlayer(enable_output=False)
    eighth = 60 / bpm / 2
    block = 256
    out = []
    elapsed = 0.0

    def advance(seconds):
        nonlocal elapsed
        target = elapsed + seconds
        while elapsed < target:
            out.append(player.render(block))
            elapsed += block / SAMPLE_RATE

    for i, chord in enumerate(progression):
        last = i == len(progression) - 1
        for step in ("D" if last else pattern):
            if step == "D":
                player.strum(chord, "down", 0.75)
            elif step == "U":
                player.strum(chord, "up", 0.5)
            advance(eighth)
    advance(2.5)  # let the last chord ring
    return np.concatenate(out)


def write_wav(path, samples):
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("-o", "--output", default="demo.wav")
    parser.add_argument("--play", action="store_true")
    args = parser.parse_args()

    samples = render()
    write_wav(args.output, samples)
    print(f"Wrote {args.output} ({len(samples) / SAMPLE_RATE:.1f} s)")

    if args.play:
        import sounddevice as sd
        sd.play(samples, SAMPLE_RATE)
        sd.wait()


if __name__ == "__main__":
    main()
