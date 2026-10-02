"""
Guitar sound for the air guitar.

Notes are synthesized with the Karplus-Strong plucked-string algorithm, so
there are no sample files to download and any chord in chords.py just works.
A small mixer plays overlapping notes through one low-latency output stream;
re-striking a string cuts off whatever that string was already playing, like
a real guitar.
"""
import threading
from functools import lru_cache
from typing import List, Optional

import numpy as np

from chords import chord_midi_notes
from config import MASTER_VOLUME, NOTE_SECONDS, SAMPLE_RATE, STRUM_SPREAD_SEC

CUT_SECONDS = 0.015   # fade when a string is re-struck or muted
UP_STRUM_STRINGS = 4  # up strums mostly catch the high strings


def midi_to_freq(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


@lru_cache(maxsize=256)
def pluck(midi: int, brightness: int = 2, sample_rate: int = SAMPLE_RATE,
          seconds: float = NOTE_SECONDS) -> np.ndarray:
    """
    One plucked string note as float32 samples in -1..1.
    brightness 0..3: higher = sharper, harder pick attack.
    """
    rng = np.random.default_rng(midi * 10 + brightness)
    period = sample_rate / midi_to_freq(midi)
    n = int(period)
    frac = period - n  # fractional delay keeps high strings in tune
    length = int(seconds * sample_rate)

    # Excitation: a burst of noise, smoothed for softer plucks, with a comb
    # notch to imitate picking a little way from the bridge.
    burst = rng.uniform(-1, 1, n)
    smooth = 4 - brightness
    if smooth > 1:
        burst = np.convolve(burst, np.ones(smooth) / smooth, mode="same")
    burst -= np.roll(burst, max(1, int(n * 0.13)))
    burst -= burst.mean()

    # Lower notes ring longer, like a real guitar
    t60 = seconds * (0.9 if midi < 52 else 0.7)
    decay = 10 ** (-3 / (t60 * midi_to_freq(midi)))

    # y[i] = decay * ((1-frac) * y[i-n] + frac * y[i-n-1])
    # Each period only depends on the one before it, so compute a whole
    # period at a time instead of looping over every sample.
    # z is y shifted by one so y[-1] = 0 doesn't need special-casing.
    z = np.zeros(length + 1)
    z[1:n + 1] = burst
    for start in range(n, length, n):
        end = min(start + n, length)
        z[start + 1:end + 1] = decay * (
            (1 - frac) * z[start - n + 1:end - n + 1] + frac * z[start - n:end - n]
        )
    y = z[1:]

    # Short fade in/out to avoid clicks
    attack = int(0.002 * sample_rate)
    release = int(0.05 * sample_rate)
    y[:attack] *= np.linspace(0, 1, attack)
    y[-release:] *= np.linspace(1, 0, release)

    peak = np.max(np.abs(y))
    return (y / peak if peak > 0 else y).astype(np.float32)


class _Voice:
    __slots__ = ("data", "pos", "delay", "gain", "string", "fade_left", "fade_len")

    def __init__(self, data, delay, gain, string):
        self.data = data
        self.pos = 0
        self.delay = delay
        self.gain = gain
        self.string = string
        self.fade_left: Optional[int] = None
        self.fade_len = 1


class AudioPlayer:
    def __init__(self, sample_rate=SAMPLE_RATE, enable_output=True):
        self.sample_rate = sample_rate
        self.voices: List[_Voice] = []
        self.lock = threading.Lock()
        self.stream = None
        if enable_output:
            self._open_stream()

    def _open_stream(self):
        try:
            import sounddevice as sd
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                dtype="float32",
                blocksize=256,      # ~6 ms per block keeps strums responsive
                latency="low",
                callback=self._callback,
            )
            self.stream.start()
        except Exception as e:  # no audio device, PortAudio missing, etc.
            print(f"[audio] Sound disabled: {e}")
            self.stream = None

    def _callback(self, outdata, frames, time_info, status):
        mono = self.render(frames)
        outdata[:, 0] = mono
        outdata[:, 1] = mono

    def strum(self, chord: str, direction: str = "down", strength: float = 0.7):
        """
        Play a chord. Down strums go low string -> high string, up strums go
        high -> low over the top strings only. Faster strums are louder,
        brighter and tighter.
        """
        strength = float(np.clip(strength, 0.0, 1.0))
        notes = chord_midi_notes(chord)
        order = list(range(6))
        if direction == "up":
            order = order[::-1][:UP_STRUM_STRINGS]

        spread = int(STRUM_SPREAD_SEC * (1.6 - strength) * self.sample_rate)
        gain = (0.35 + 0.65 * strength) * (0.8 if direction == "up" else 1.0)
        brightness = min(3, int(strength * 4))

        with self.lock:
            # The fretting hand mutes strings the chord doesn't use
            for string, midi in enumerate(notes):
                if midi is None:
                    self._cut_string(string)

            for k, string in enumerate(order):
                midi = notes[string]
                if midi is None:
                    continue
                self._cut_string(string)
                self.voices.append(_Voice(pluck(midi, brightness, self.sample_rate),
                                          k * spread, gain, string))

    def play_strum_down(self, chord, strength=0.7):
        self.strum(chord, "down", strength)

    def play_strum_up(self, chord, strength=0.7):
        self.strum(chord, "up", strength)

    def _cut_string(self, string: int):
        fade = int(CUT_SECONDS * self.sample_rate)
        for v in self.voices:
            if v.string == string and v.fade_left is None:
                v.fade_left = fade
                v.fade_len = fade

    def render(self, frames: int) -> np.ndarray:
        """Mix the next `frames` samples. Used by the stream and by tests/demos."""
        out = np.zeros(frames, dtype=np.float32)
        with self.lock:
            alive = []
            for v in self.voices:
                if v.delay >= frames:
                    v.delay -= frames
                    alive.append(v)
                    continue
                start = v.delay
                v.delay = 0
                n = min(frames - start, len(v.data) - v.pos)
                seg = v.data[v.pos:v.pos + n] * v.gain
                if v.fade_left is not None:
                    n = min(n, v.fade_left)
                    seg = seg[:n] * ((v.fade_left - np.arange(n)) / v.fade_len)
                    v.fade_left -= n
                out[start:start + n] += seg
                v.pos += n
                if v.pos < len(v.data) and (v.fade_left is None or v.fade_left > 0):
                    alive.append(v)
            self.voices = alive
        # Soft clip so big chords never hard-distort
        return np.tanh(out * MASTER_VOLUME * 2.0) / 2.0

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
