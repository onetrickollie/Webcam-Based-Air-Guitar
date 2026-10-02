import json

from calibration import CalibrationRecorder, load_templates, reset_templates, save_templates
from chords import DEFAULT_TEMPLATES
from gestures_left import FingerCurls


def test_defaults_when_no_file(tmp_path):
    assert load_templates(tmp_path / "missing.json") == DEFAULT_TEMPLATES


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "cal.json"
    templates = load_templates(path)
    templates["C"] = [0.6, 0.7, 0.5, 0.1]
    save_templates(templates, path)
    assert load_templates(path)["C"] == [0.6, 0.7, 0.5, 0.1]
    assert load_templates(path)["G"] == DEFAULT_TEMPLATES["G"]


def test_bad_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text("{not json")
    assert load_templates(path) == DEFAULT_TEMPLATES
    path.write_text(json.dumps({"C": [1, 2], "Nope": [0, 0, 0, 0]}))
    assert load_templates(path) == DEFAULT_TEMPLATES


def test_reset_deletes_file(tmp_path):
    path = tmp_path / "cal.json"
    save_templates({"C": [0, 0, 0, 0]}, path)
    assert reset_templates(path) == DEFAULT_TEMPLATES
    assert not path.exists()


def test_recorder_averages_frames():
    rec = CalibrationRecorder(frames=4)
    rec.start("Am")
    results = [rec.feed(FingerCurls(0.1 * i, 0.5, 0.5, 0.0)) for i in range(4)]
    assert results[:3] == [None, None, None]
    chord, shape = results[3]
    assert chord == "Am"
    assert shape == [0.15, 0.5, 0.5, 0.0]
    assert not rec.active


def test_recorder_ignores_frames_when_inactive():
    rec = CalibrationRecorder(frames=2)
    assert rec.feed(FingerCurls(1, 1, 1, 1)) is None
