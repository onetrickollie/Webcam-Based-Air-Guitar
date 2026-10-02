# xmin, ymin, xmax, ymax (fractions of the mirrored camera frame)
# The left zone is wider than before so the fretting hand doesn't drop out
# of tracking as easily.
LEFT_ZONE = (0.05, 0.30, 0.45, 0.95)
RIGHT_ZONE = (0.55, 0.30, 0.95, 0.95)

# --- left hand: finger curl ---------------------------------------------------

# Total bend (MCP + PIP + DIP, radians) that counts as a fully curled finger.
# Calibrating your own chord shapes (keys 1-7) matters more than this number.
CURL_FULL_BEND_RAD = 3.0

# 0..1, higher = follows the hand faster but jitters more
CURL_SMOOTHING = 0.5

# A chord must win this many frames in a row before it switches (stops flicker)
CHORD_STABLE_FRAMES = 3

# Max distance (RMS of curl differences) for a shape to count as a chord.
# Anything further from every template keeps the previous chord.
CHORD_MAX_DISTANCE = 0.35

# Frames recorded and averaged when calibrating a chord shape
CALIBRATION_FRAMES = 20
CALIBRATION_FILE = "chord_calibration.json"

# --- right hand: strumming ----------------------------------------------------

# Hand speed in screen-heights per second. Speeds are per second (not per
# frame) so strumming feels the same at any camera frame rate.
STRUM_TRIGGER_SPEED = 0.9   # moving faster than this fires a strum
STRUM_REARM_SPEED = 0.3     # must slow below this before the same direction fires again
STRUM_SMOOTHING = 0.6       # 0..1 EMA on velocity
STRUM_MAX_SPEED = 4.0       # speed that counts as a full-volume strum

# --- audio --------------------------------------------------------------------

SAMPLE_RATE = 44100
NOTE_SECONDS = 3.0
STRUM_SPREAD_SEC = 0.012    # delay between strings in a strum (slower strum = wider)
MASTER_VOLUME = 0.35

# chord display text (just a number for cv2 font)
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX is usually 0
