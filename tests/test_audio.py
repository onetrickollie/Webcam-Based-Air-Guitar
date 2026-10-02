import numpy as np
import pytest

from audio_player import AudioPlayer, midi_to_freq, pluck
from chords import CHORD_FRETS, chord_midi_notes
from config import SAMPLE_RATE


def measured_pitch(samples, sr=SAMPLE_RATE):
    """Fundamental frequency by autocorrelation, refined with parabolic interpolation."""
    x = samples[int(0.05 * sr):int(0.35 * sr)].astype(float)
    x -= x.mean()
    corr = np.correlate(x, x, mode="full")[len(x) - 1:]
    lo, hi = int(sr / 1000), int(sr / 60)
    lag = lo + int(np.argmax(corr[lo:hi]))
    a, b, c = corr[lag - 1], corr[lag], corr[lag + 1]
    lag = lag + 0.5 * (a - c) / (a - 2 * b + c)
    return sr / lag


@pytest.mark.parametrize("midi", [40, 45, 50, 55, 59, 64, 67])
def test_pluck_is_in_tune(midi):
    freq = measured_pitch(pluck(midi))
    cents = 1200 * np.log2(freq / midi_to_freq(midi))
    assert abs(cents) < 10, f"{midi}: {freq:.1f} Hz is {cents:+.1f} cents off"


def test_pluck_decays_and_stays_in_range():
    y = pluck(45)
    assert np.max(np.abs(y)) <= 1.0
    early = np.abs(y[:4410]).mean()
    late = np.abs(y[-8820:-4410]).mean()
    assert late < early * 0.1


def test_chord_notes():
    assert chord_midi_notes("OPEN") == [40, 45, 50, 55, 59, 64]
    assert chord_midi_notes("C") == [None, 48, 52, 55, 60, 64]  # C E G C E
    assert set(CHORD_FRETS) >= {"OPEN", "C", "G", "D", "Am", "Em", "F"}


def render_seconds(player, seconds):
    return np.concatenate([player.render(256) for _ in range(int(seconds * SAMPLE_RATE / 256))])


def test_strum_produces_sound_without_clipping():
    player = AudioPlayer(enable_output=False)
    player.strum("F", "down", 1.0)
    player.strum("F", "up", 1.0)
    out = render_seconds(player, 1.0)
    assert np.abs(out).max() > 0.05
    assert np.abs(out).max() < 0.5  # soft clipper caps at 0.5


def test_down_strum_starts_on_low_string():
    player = AudioPlayer(enable_output=False)
    player.strum("OPEN", "down", 0.3)
    starts = sorted((v.delay, v.string) for v in player.voices)
    assert [s for _, s in starts] == [0, 1, 2, 3, 4, 5]


def test_up_strum_starts_on_high_string_and_skips_bass():
    player = AudioPlayer(enable_output=False)
    player.strum("OPEN", "up", 0.3)
    starts = sorted((v.delay, v.string) for v in player.voices)
    assert [s for _, s in starts] == [5, 4, 3, 2]


def test_muted_strings_are_not_played():
    player = AudioPlayer(enable_output=False)
    player.strum("D", "down")  # D chord mutes low E and A
    assert sorted(v.string for v in player.voices) == [2, 3, 4, 5]


def test_restrumming_cuts_previous_notes_on_same_string():
    player = AudioPlayer(enable_output=False)
    player.strum("G", "down")
    render_seconds(player, 0.2)
    player.strum("C", "down")
    render_seconds(player, 0.1)  # longer than the cut fade
    assert len(player.voices) == 5  # just the new C notes; G voices faded out


def test_voices_are_cleaned_up_after_they_finish():
    player = AudioPlayer(enable_output=False)
    player.strum("Em", "down")
    render_seconds(player, 3.5)
    assert player.voices == []
