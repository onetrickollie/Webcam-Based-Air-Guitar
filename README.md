# Webcam Based Air Guitar
This is a project by Kai. A webcam-based air guitar: make chord shapes with your left hand, strum with your right, and hear real guitar chords.

## Setup

Requires Python 3.9–3.12 (mediapipe doesn't support newer versions yet).

```bash
python3.12 -m venv ~/.venvs/air-guitar
source ~/.venvs/air-guitar/bin/activate
pip install -r requirements.txt
```

> **Mac users:** if this folder is on your Desktop or in Documents with iCloud sync on, keep the virtual environment *outside* it (like `~/.venvs/` above). iCloud deletes files out of virtual environments and breaks them.

## Run

```bash
python main.py
```

The first time, macOS asks for camera access for your terminal app. If the camera won't open, allow it in **System Settings → Privacy & Security → Camera**.

Options: `--camera 1` uses a different camera, `--no-sound` runs silently, `--debug` prints finger curls and chord matching.

## How to play

Stand back so both hands fit on screen. The video is mirrored, like a mirror.

- **Left hand (green FRET box):** make a chord shape. The current chord is shown at the top left.
- **Right hand (blue STRUM box):** swipe down or up to strum. Faster strums are louder.

Default chord shapes, by which fingers are curled:

| Key | Chord | Curled fingers |
|----:|-------|----------------|
| 1 | OPEN | none (open hand) |
| 2 | C  | index, middle, ring |
| 3 | G  | middle, ring, pinky |
| 4 | D  | index, middle |
| 5 | Am | middle, ring |
| 6 | Em | ring, pinky |
| 7 | F  | all four (fist) |

If your left hand leaves the box, strums play the open strings.

### Calibrate your own shapes (recommended)

Everyone's hands curl differently. Hold a shape with your left hand and press its number key (1–7): the app records the shape for about a second and saves it to `chord_calibration.json`. You can use any shape you like for any chord. Press `r` to reset to the defaults.

| Key | Action |
|-----|--------|
| 1–7 | Calibrate the chord shape you're holding |
| r | Reset calibration |
| d | Toggle debug output |
| h | Show/hide help |
| Esc / q | Quit |

## Hear it without a webcam

```bash
python render_demo.py --play   # writes demo.wav and plays a short chord progression
```

## How it works

1. Read webcam frames with OpenCV and flip them to mirror view
2. Run MediaPipe Hands to get 21 3D landmarks per hand
3. Assign each hand by which box it's in (MediaPipe's Left/Right label can flip between frames)
4. **Left hand:** measure each finger's curl from its joint angles (wrist→MCP→PIP→DIP→tip). Angles stay the same whether the hand is near, far, or tilted, unlike raw positions. Curls are smoothed, matched to the nearest chord shape, and a chord only switches after 3 steady frames.
5. **Right hand:** track the palm's vertical speed (per second, so frame rate doesn't matter). A strum fires once when the speed passes a threshold, and the same direction re-arms only after the hand slows or reverses.
6. **Sound:** each string is synthesized with the Karplus-Strong plucked-string algorithm, with no sample files needed. Down strums go low string to high, and up strums go high to low over the top four strings. Re-striking a string cuts its previous note, like a real guitar.

Settings like zone sizes, strum sensitivity, and smoothing live in `config.py`.

## Tests

```bash
python -m pytest
```

The tests use simulated hands, so no camera is needed. They check that curl readings ignore hand rotation and distance, that strum direction and once-per-strum firing work, that every chord note is in tune, and that calibration saves and loads.

## Ideas for next steps
- Single-string picking mode (pluck one string at a time)
- More chords (7ths, power chords) and a capo setting
- A chord chart overlay for playing along with songs
