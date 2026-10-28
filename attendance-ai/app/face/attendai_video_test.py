import cv2
import os
import re
import csv
import time
import json
import shutil
import sys
import argparse     # [TEST]
import subprocess   # [TEST]
import datetime
import logging
import warnings
from collections import deque

import numpy as np
import torch

# ============================================================
# ATTENDAI - CCTV VIDEO TEST  (same pipeline as attendai_v3.py)
# ============================================================
#
# Pipeline (unchanged):
#
# Video file (instead of camera)                    <- [TEST]
#   ↓
# YOLO Person Detection
#   ↓
# BoT-SORT Tracking
#   ↓
# Door zone -> ENTRY / EXIT events                  <- [TEST] new
#   ↓
# Recognition Zone
#   ↓
# InsightFace
#   ↓
# Identity Confirmation
#   ↓
# Temporary Daily Session Storage
#   ↓
# Attendance Log
#
# Everything that differs from attendai_v3.py is marked  # [TEST]
#
# RUN (from the project root, same place you run attendai_v3.py):
#
#   python .\app\face\attendai_video_test.py --video "D:\videos\C-304-FRONT....mp4"
#   python .\app\face\attendai_video_test.py --video "front.mp4" "back.mp4"
#
# Useful options:
#   --camera FRONT|BACK|AUTO   door/zone preset (AUTO reads the file name)
#   --start 60 --end 180       only test a part of the video (seconds)
#   --stride 2                 process every 2nd frame (2x faster)
#   --imgsz 1280               YOLO input size (default 1280 GPU / 640 CPU)
#   --no-show                  no window (faster, just writes results)
#   --no-save-video            do not write the annotated .mp4
#   --pick-zone                click the door polygon on the first frame
#
# Keys while the window is open:  Q = quit,  SPACE = pause
#
# ============================================================


# ============================================================
# [TEST] COMMAND LINE
# ============================================================

parser = argparse.ArgumentParser(
    description="AttendAI - test the pipeline on CCTV video files"
)

parser.add_argument("--video", nargs="+", required=True)
parser.add_argument("--camera", default="AUTO")
parser.add_argument("--start", type=float, default=0.0)
parser.add_argument("--end", type=float, default=None)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--no-show", action="store_true")
parser.add_argument("--no-save-video", action="store_true")
parser.add_argument("--pick-zone", action="store_true")
parser.add_argument("--output", default="test_results")
parser.add_argument("--imgsz", type=int, default=None)

ARGS = parser.parse_args()


# [TEST] Several videos -> run each one in its own fresh process, so the
# tracker, sessions and re-ID cache never leak from one video to the next.
if len(ARGS.video) > 1:

    for _video in ARGS.video:

        _cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--video", _video,
            "--camera", ARGS.camera,
            "--start", str(ARGS.start),
            "--stride", str(ARGS.stride),
            "--output", ARGS.output
        ]

        if ARGS.end is not None:
            _cmd += ["--end", str(ARGS.end)]

        if ARGS.imgsz is not None:
            _cmd += ["--imgsz", str(ARGS.imgsz)]

        if ARGS.no_show:
            _cmd.append("--no-show")

        if ARGS.no_save_video:
            _cmd.append("--no-save-video")

        print("\n>>> Running:", _video)

        subprocess.run(_cmd)

    raise SystemExit(0)


VIDEO_PATH = ARGS.video[0]

if not os.path.isfile(VIDEO_PATH):

    print("ERROR: video not found:", VIDEO_PATH)

    raise SystemExit(1)


# Load the CUDA 12 / cuDNN 9 DLLs that onnxruntime-gpu needs
# (same fix as attendai_v3.py).
try:

    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):

        ort.preload_dlls(
            cuda=True,
            cudnn=True,
            msvc=True,
            directory=""
        )

except Exception as e:

    print("onnxruntime preload_dlls skipped:", e)

from ultralytics import YOLO
from insightface.app import FaceAnalysis

import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.attendance.attendance_engine import AttendanceEngine


# ============================================================
# SILENCE LOG SPAM
# ============================================================

class _HalfDeprecationFilter(logging.Filter):

    def filter(self, record):

        return "'half' is deprecated" not in record.getMessage()


try:

    from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER

    ULTRALYTICS_LOGGER.addFilter(
        _HalfDeprecationFilter()
    )

    for _handler in ULTRALYTICS_LOGGER.handlers:
        _handler.addFilter(_HalfDeprecationFilter())

except Exception:

    pass

warnings.filterwarnings(
    "ignore",
    message=".*'half' is deprecated.*"
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="insightface"
)


# ============================================================
# SETTINGS
# ============================================================

YOLO_MODEL = "yolo11n.pt"

TRACKER_CONFIG = "trackers/fast_botsort.yaml"

# [TEST] if your custom tracker yaml is not found, use the built-in one
if not os.path.exists(TRACKER_CONFIG):

    TRACKER_CONFIG = "botsort.yaml"

KNOWN_FACES_DIR = "app/face/known_faces"


# ============================================================
# FACE RECOGNITION
# ============================================================

MATCH_THRESHOLD = 0.50

STRONG_MATCH_THRESHOLD = 0.62

REQUIRED_CONFIRMATIONS = 2

# [TEST] CCTV faces are much smaller than webcam faces (camera is on the
# ceiling, 5-10 m away). 45 px rejected almost every face in the BACK
# camera. Known-face ENROLMENT below still uses the old 45 px value.
MIN_FACE_SIZE = 28

ENROLL_MIN_FACE_SIZE = 45


# ============================================================
# YOLO
# ============================================================

YOLO_CONFIDENCE = 0.20

YOLO_NMS_IOU = 0.50

YOLO_IMAGE_SIZE = 640
YOLO_HALF_PRECISION = True

# [TEST] The CCTV frame is 2560x1440. At 640 the students in the last
# rows become ~15 px tall and disappear. 1280 keeps them detectable.
YOLO_MODEL_GPU = "yolo11s.pt"
YOLO_IMAGE_SIZE_GPU = 1280


# ============================================================
# [TEST] DISPLAY / OUTPUT
# ============================================================

DISPLAY_WIDTH = 1280            # window + saved video width

SAVE_ENTRY_SNAPSHOTS = True     # person crop at the moment of ENTRY


# ============================================================
# SESSION MEMORY
# ============================================================

SESSION_MEMORY_SECONDS = 10.0


# ============================================================
# TRAJECTORY
# ============================================================

MAX_TRAJECTORY_POINTS = 100

TRAJECTORY_DRAW_POINTS = 18

SHOW_TRAJECTORY = True


# ============================================================
# RE-ID CACHE
# ============================================================

REID_CACHE_SECONDS = 20.0

REID_MATCH_THRESHOLD = 0.55

LOST_TO_CACHE_SECONDS = 1.0

LOST_TO_CACHE_SECONDS_FAST = 0.35


# ============================================================
# FAST-MOTION TRACKING
# ============================================================

VELOCITY_SMOOTHING = 0.6

# [TEST] tuned at 960 px wide; scaled to the real video width later
FAST_SPEED_PX_PER_SEC_AT_960 = 550
FAST_SPEED_PX_PER_SEC = FAST_SPEED_PX_PER_SEC_AT_960

INTERPOLATE_ZONE_CROSSING = True

MAX_URGENT_RECOGNITIONS_PER_FRAME = 2

MAX_ZONE_ENTRY_RECOGNITIONS_PER_FRAME = 2

FAST_CROP_EXTRA_MARGIN_RATIO = 0.12


# ============================================================
# FACE CROP
# ============================================================

CROP_MARGIN_RATIO = 0.15

FACE_CROP_HEIGHT_RATIO = 0.50
FACE_CROP_MIN_HEIGHT = 240


# ============================================================
# FAST RECOGNITION SCHEDULING
# ============================================================

RECOGNITION_INTERVAL_FRAMES = 8
CONFIRMED_RECOGNITION_INTERVAL_FRAMES = 24
MAX_RECOGNITION_GAP_SECONDS = 2.0

MAX_RECOGNITIONS_PER_FRAME = 1


# ============================================================
# ZONE ENTRY SNAPSHOT RECOGNITION
# ============================================================

ZONE_ENTRY_RECOGNITION = True


# ============================================================
# [TEST] CAMERA PRESETS  (C-304 classroom)
# ============================================================
# All values are ratios of the frame (0.0 - 1.0), so they work at any
# resolution. "door" is a polygon on the FLOOR: a person is "in the door"
# when their FEET (bottom-center of the YOLO box) are inside it.
#
# FRONT camera: students walk in at the bottom-right corner.
# BACK  camera: students walk in through the far door, left of the board.
#
# These were measured from the 28-09-2026 footage. For another room run
# with --pick-zone and paste the printed polygon here.

CAMERA_PRESETS = {

    "FRONT": {

        "door": [
            (0.84, 0.60),
            (1.00, 0.60),
            (1.00, 1.00),
            (0.84, 1.00)
        ],

        # (x1, y1, x2, y2) recognition zone
        "zone": (0.05, 0.03, 0.99, 0.99)
    },

    "BACK": {

        # only the doorway itself - the front-row desks left of it
        # are kept OUT, otherwise seated students "visit the door"
        "door": [
            (0.355, 0.08),
            (0.43, 0.08),
            (0.43, 0.215),
            (0.355, 0.215)
        ],

        "zone": (0.02, 0.03, 0.98, 0.99)
    }
}

# ENTRY: a track's feet were in the door, then stay OUTSIDE the door for
#        ENTRY_CONFIRM_SECONDS after moving at least ENTRY_MIN_TRAVEL_RATIO
#        of the frame height into the room.
# EXIT : a track that was in the room is LOST (not seen for
#        EXIT_LOST_SECONDS) while its feet were in the door.
ENTRY_CONFIRM_SECONDS = 0.5
EXIT_LOST_SECONDS = 1.0
ENTRY_MIN_TRAVEL_RATIO = 0.06

# False = recognise everybody in the zone (normal attendai_v3 behaviour,
#         seated students are recognised too).
# True  = only recognise people who came in through the door.
RECOGNIZE_ONLY_ENTRANTS = False

# ============================================================
# [HEADCOUNT] PEOPLE INSIDE THE CLASSROOM
# ============================================================
# Head count = YOLO person boxes in the current frame, minus people
# standing in the door polygon (they are not inside yet).
# Raw per-frame count flickers (occlusion), so the shown value is the
# MEDIAN over the last HEADCOUNT_SMOOTH_SECONDS.

HEADCOUNT_MIN_CONFIDENCE = 0.20     # raise to 0.30 if chairs/bags get counted
HEADCOUNT_EXCLUDE_DOOR = True
HEADCOUNT_SMOOTH_SECONDS = 1.0
HEADCOUNT_LOG_INTERVAL = 1.0        # write one row to headcount.csv every N s

# ============================================================
# [TEST] VIDEO NAME -> CAMERA + REAL CCTV CLOCK
# ============================================================
# NVR file names look like
#   C-304-FRONT_..._20260928080000_20260928080414_20574546.mp4
# The first 14-digit block is the real start time of the clip. All
# timing in the pipeline (session expiry, re-ID cache, speed, snapshot
# interval, attendance time) now runs on VIDEO time, not the PC clock,
# so results are the same whether your PC processes at 5 FPS or 50 FPS.

VIDEO_NAME = os.path.basename(VIDEO_PATH)

if ARGS.camera.upper() == "AUTO":

    if "FRONT" in VIDEO_NAME.upper():
        CAMERA_NAME = "FRONT"
    elif "BACK" in VIDEO_NAME.upper():
        CAMERA_NAME = "BACK"
    else:
        CAMERA_NAME = "FRONT"

else:

    CAMERA_NAME = ARGS.camera.upper()

if CAMERA_NAME not in CAMERA_PRESETS:

    print("ERROR: unknown camera preset:", CAMERA_NAME)

    raise SystemExit(1)

PRESET = CAMERA_PRESETS[CAMERA_NAME]

ZONE_X1_RATIO, ZONE_Y1_RATIO, ZONE_X2_RATIO, ZONE_Y2_RATIO = PRESET["zone"]

DOOR_POLYGON_RATIO = PRESET["door"]

_stamp = re.search(r"_(\d{14})_\d{14}_", VIDEO_NAME)

if _stamp:

    VIDEO_START_EPOCH = datetime.datetime.strptime(
        _stamp.group(1),
        "%Y%m%d%H%M%S"
    ).timestamp()

else:

    VIDEO_START_EPOCH = time.time()


# ============================================================
# [TEST] OUTPUT FOLDER (keeps test data away from real attendance)
# ============================================================

RUN_NAME = (
    f"{CAMERA_NAME}_"
    f"{datetime.datetime.fromtimestamp(VIDEO_START_EPOCH).strftime('%Y%m%d_%H%M%S')}"
)

RUN_DIRECTORY = os.path.join(
    ARGS.output,
    RUN_NAME
)

ENTRY_SNAPSHOT_DIRECTORY = os.path.join(
    RUN_DIRECTORY,
    "entry_snapshots"
)

os.makedirs(ENTRY_SNAPSHOT_DIRECTORY, exist_ok=True)

ENTRIES_CSV_PATH = os.path.join(RUN_DIRECTORY, "entries.csv")
RECOGNITIONS_CSV_PATH = os.path.join(RUN_DIRECTORY, "recognitions.csv")
SUMMARY_JSON_PATH = os.path.join(RUN_DIRECTORY, "summary.json")
ANNOTATED_VIDEO_PATH = os.path.join(RUN_DIRECTORY, "annotated.mp4")
HEADCOUNT_CSV_PATH = os.path.join(RUN_DIRECTORY, "headcount.csv")   # [HEADCOUNT]


# ============================================================
# ATTENDANCE
# ============================================================

ATTENDANCE_LOG_PATH = os.path.join(RUN_DIRECTORY, "attendance_log.csv")  # [TEST]


# ============================================================
# TEMPORARY DAY STORAGE
# ============================================================

TEMP_STORAGE_ROOT = os.path.join(RUN_DIRECTORY, "temporary_day_data")  # [TEST]

SAVE_PERSON_SNAPSHOTS = True

SAVE_FACE_SNAPSHOTS = True

SAVE_EMBEDDINGS = True

SNAPSHOT_INTERVAL = 1.5

attendance_engine = AttendanceEngine(
    storage_root=TEMP_STORAGE_ROOT
)


# ============================================================
# GPU
# ============================================================

GPU_AVAILABLE = torch.cuda.is_available()

if GPU_AVAILABLE:

    DEVICE = 0

    YOLO_MODEL = YOLO_MODEL_GPU
    YOLO_IMAGE_SIZE = YOLO_IMAGE_SIZE_GPU

    print("======================================")
    print("GPU AVAILABLE")
    print("GPU:", torch.cuda.get_device_name(0))
    print("YOLO:", YOLO_MODEL, "@", YOLO_IMAGE_SIZE)
    print("======================================")

else:

    DEVICE = "cpu"

    print("======================================")
    print("GPU NOT AVAILABLE")
    print("Using CPU")
    print("YOLO:", YOLO_MODEL, "@", YOLO_IMAGE_SIZE)
    print("======================================")


# [TEST] --imgsz overrides the YOLO input size
if ARGS.imgsz is not None:

    YOLO_IMAGE_SIZE = ARGS.imgsz

    print("YOLO image size override:", YOLO_IMAGE_SIZE)


# ============================================================
# DATE / TIME HELPERS
# ============================================================
# [TEST] "now" = the CCTV clock of the frame being processed

CURRENT_VIDEO_TIME = VIDEO_START_EPOCH


def get_today_string():

    return datetime.datetime.fromtimestamp(
        CURRENT_VIDEO_TIME
    ).strftime("%Y-%m-%d")


def get_timestamp_string():

    return datetime.datetime.fromtimestamp(
        CURRENT_VIDEO_TIME
    ).strftime("%Y-%m-%d %H:%M:%S")


def get_safe_name(name):

    if not name:
        return "UNKNOWN"

    return "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in name
    )


# ============================================================
# TEMPORARY SESSION STORE
# ============================================================

class TemporarySessionStore:

    def __init__(self, root_directory):

        self.root_directory = root_directory

        self.today = get_today_string()

        self.day_directory = os.path.join(
            self.root_directory,
            self.today
        )

        self.sessions_directory = os.path.join(
            self.day_directory,
            "sessions"
        )

        self.people_directory = os.path.join(
            self.day_directory,
            "people"
        )

        self.face_directory = os.path.join(
            self.day_directory,
            "faces"
        )

        self.embedding_directory = os.path.join(
            self.day_directory,
            "embeddings"
        )

        os.makedirs(self.sessions_directory, exist_ok=True)
        os.makedirs(self.people_directory, exist_ok=True)
        os.makedirs(self.face_directory, exist_ok=True)
        os.makedirs(self.embedding_directory, exist_ok=True)

        self.memory = {}

    # --------------------------------------------------------
    # CREATE SESSION
    # --------------------------------------------------------

    def create_session(self, track_id, timestamp):

        session_id = f"track_{track_id}"

        self.memory[session_id] = {

            "session_id": session_id,

            "track_id": int(track_id),

            "identity": "UNKNOWN",

            "status": "TRACKING",

            "best_confidence": 0.0,

            "confirmations": 0,

            "first_seen": timestamp,

            "last_seen": timestamp,

            "zone_entered": False,

            "zone_entry_recognition_done": False,

            "attendance_marked": False,

            # [TEST]
            "entry_state": "NONE",
            "entry_events": [],

            "trajectory": [],

            "snapshots": [],

            "face_snapshots": [],

            "embedding_saved": False,
            "recognition_frame_counter": 0,
            "last_recognition_time": 0.0,
            "recognition_count": 0,

            "appearance": {

                "person_crop_available": False,

                "face_crop_available": False,

                "clothing_analysis": "NOT_ANALYZED",

                "watch_analysis": "NOT_ANALYZED",

                "ring_analysis": "NOT_ANALYZED",

                "other_attributes": "NOT_ANALYZED"

            }

        }

        return session_id

    # --------------------------------------------------------
    # UPDATE SESSION
    # --------------------------------------------------------

    def update_session(
        self,
        track_id,
        session
    ):

        session_id = f"track_{track_id}"

        if session_id not in self.memory:

            self.create_session(
                track_id,
                get_timestamp_string()
            )

        data = self.memory[session_id]

        data["identity"] = session["name"]

        data["status"] = session["status"]

        data["best_confidence"] = float(
            session["best_score"]
        )

        data["confirmations"] = int(
            session["confirmations"]
        )

        data["last_seen"] = datetime.datetime.fromtimestamp(
            session["last_seen"]
        ).strftime("%Y-%m-%d %H:%M:%S")

        data["zone_entered"] = bool(
            session["zone_entered"]
        )

        data["zone_entry_recognition_done"] = bool(
            session["zone_entry_recognition_done"]
        )

        data["attendance_marked"] = bool(
            session["attendance_marked"]
        )

        # [TEST]
        data["entry_state"] = session["entry_state"]
        data["entry_events"] = list(session["entry_events"])

        trajectory = list(
            session["trajectory"]
        )

        data["trajectory"] = [

            {
                "x": int(x),
                "y": int(y)
            }

            for x, y in trajectory
        ]

    # --------------------------------------------------------
    # SAVE PERSON SNAPSHOT
    # --------------------------------------------------------

    def save_person_snapshot(
        self,
        track_id,
        frame,
        bbox,
        current_time
    ):

        if not SAVE_PERSON_SNAPSHOTS:
            return None

        session_id = f"track_{track_id}"

        if session_id not in self.memory:
            return None

        timestamp = datetime.datetime.fromtimestamp(
            current_time
        ).strftime("%H%M%S_%f")

        x1, y1, x2, y2 = bbox

        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        filename = (
            f"track_{track_id}_"
            f"{timestamp}.jpg"
        )

        path = os.path.join(
            self.people_directory,
            filename
        )

        cv2.imwrite(path, crop)

        relative_path = os.path.relpath(
            path,
            self.day_directory
        )

        self.memory[session_id]["snapshots"].append(
            relative_path
        )

        self.memory[session_id]["appearance"][
            "person_crop_available"
        ] = True

        return path

    # --------------------------------------------------------
    # SAVE FACE SNAPSHOT
    # --------------------------------------------------------

    def save_face_snapshot(
        self,
        track_id,
        frame,
        bbox,
        current_time
    ):

        if not SAVE_FACE_SNAPSHOTS:
            return None

        session_id = f"track_{track_id}"

        if session_id not in self.memory:
            return None

        timestamp = datetime.datetime.fromtimestamp(
            current_time
        ).strftime("%H%M%S_%f")

        x1, y1, x2, y2 = bbox

        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        filename = (
            f"track_{track_id}_face_"
            f"{timestamp}.jpg"
        )

        path = os.path.join(
            self.face_directory,
            filename
        )

        cv2.imwrite(path, crop)

        relative_path = os.path.relpath(
            path,
            self.day_directory
        )

        self.memory[session_id]["face_snapshots"].append(
            relative_path
        )

        self.memory[session_id]["appearance"][
            "face_crop_available"
        ] = True

        return path

    # --------------------------------------------------------
    # SAVE EMBEDDING
    # --------------------------------------------------------

    def save_embedding(
        self,
        track_id,
        embedding
    ):

        if not SAVE_EMBEDDINGS:
            return None

        if embedding is None:
            return None

        session_id = f"track_{track_id}"

        if session_id not in self.memory:
            return None

        path = os.path.join(
            self.embedding_directory,
            f"track_{track_id}.npy"
        )

        np.save(
            path,
            np.asarray(
                embedding,
                dtype=np.float32
            )
        )

        self.memory[session_id][
            "embedding_saved"
        ] = True

        return path

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    def save_session(
        self,
        track_id
    ):

        session_id = f"track_{track_id}"

        if session_id not in self.memory:
            return

        data = self.memory[session_id]

        path = os.path.join(
            self.sessions_directory,
            f"track_{track_id}.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=4
            )

    # --------------------------------------------------------
    # SAVE ALL
    # --------------------------------------------------------

    def save_all(self):

        for session_id in self.memory:

            track_id = self.memory[
                session_id
            ]["track_id"]

            self.save_session(
                track_id
            )

    # --------------------------------------------------------
    # GET SUMMARY
    # --------------------------------------------------------

    def get_summary(self):

        total = len(self.memory)

        confirmed = sum(
            1
            for item in self.memory.values()
            if item["status"] == "CONFIRMED"
        )

        return {
            "date": self.today,
            "total_sessions": total,
            "confirmed_people": confirmed
        }


# ============================================================
# INITIALIZE TEMPORARY STORAGE
# ============================================================

temporary_store = TemporarySessionStore(
    TEMP_STORAGE_ROOT
)

print()
print("======================================")
print("TEMPORARY DAY STORAGE")
print("======================================")
print(
    "Storage:",
    temporary_store.day_directory
)
print("======================================")
print()


# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO...")

model = YOLO(YOLO_MODEL)

print("YOLO loaded.")


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

def get_face_app_providers(app):

    providers = set()

    for m in getattr(app, "models", {}).values():

        session = getattr(m, "session", None)

        if session is not None:

            providers.update(
                session.get_providers()
            )

    return providers


print("\nLoading InsightFace...")

try:

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]
    )

    face_app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    if "CUDAExecutionProvider" in get_face_app_providers(face_app):

        FACE_DEVICE = "GPU"

        print(
            "InsightFace GPU initialized."
        )

    else:

        FACE_DEVICE = "CPU"

        print(
            "InsightFace is running on CPU "
            "(CUDA 12 DLLs for onnxruntime not found)."
        )

except Exception as e:

    print(
        "\nInsightFace GPU initialization failed."
    )

    print("Reason:", e)

    print(
        "Falling back to CPU..."
    )

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=[
            "CPUExecutionProvider"
        ]
    )

    face_app.prepare(
        ctx_id=-1,
        det_size=(640, 640)
    )

    FACE_DEVICE = "CPU"

    print(
        "InsightFace CPU initialized."
    )


# ============================================================
# HELPERS
# ============================================================

def cosine_similarity(a, b):

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    denom = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denom == 0:
        return 0.0

    return float(
        np.dot(a, b) / denom
    )


def normalize_embedding(
    embedding
):

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:
        return embedding

    return embedding / norm


# ============================================================
# LOAD KNOWN FACES
# ============================================================

known_faces = {}

print()
print("======================================")
print("SCANNING KNOWN FACES")
print("======================================")


if not os.path.exists(
    KNOWN_FACES_DIR
):

    print(
        "ERROR: Known faces directory does not exist:"
    )

    print(
        KNOWN_FACES_DIR
    )

    exit()


for person_name in os.listdir(
    KNOWN_FACES_DIR
):

    person_folder = os.path.join(
        KNOWN_FACES_DIR,
        person_name
    )

    # --------------------------------------------------------
    # SINGLE IMAGE
    # --------------------------------------------------------

    if os.path.isfile(
        person_folder
    ):

        if not person_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            continue

        image = cv2.imread(
            person_folder
        )

        if image is None:
            continue

        faces = face_app.get(
            image
        )

        if len(faces) == 0:

            print(
                "No face found:",
                person_name
            )

            continue

        face = max(
            faces,
            key=lambda f:
            (
                f.bbox[2] - f.bbox[0]
            )
            *
            (
                f.bbox[3] - f.bbox[1]
            )
        )

        embedding = normalize_embedding(
            face.normed_embedding
        )

        name = os.path.splitext(
            person_name
        )[0]

        known_faces[name] = embedding

        print(
            f"Loaded: {name}"
        )

        continue

    # --------------------------------------------------------
    # PERSON FOLDER
    # --------------------------------------------------------

    if not os.path.isdir(
        person_folder
    ):

        continue

    person_embeddings = []

    for filename in os.listdir(
        person_folder
    ):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            continue

        path = os.path.join(
            person_folder,
            filename
        )

        image = cv2.imread(
            path
        )

        if image is None:
            continue

        faces = face_app.get(
            image
        )

        if len(faces) == 0:
            continue

        face = max(
            faces,
            key=lambda f:
            (
                f.bbox[2] - f.bbox[0]
            )
            *
            (
                f.bbox[3] - f.bbox[1]
            )
        )

        x1, y1, x2, y2 = (
            face.bbox.astype(int)
        )

        width = x2 - x1
        height = y2 - y1

        if (
            width < ENROLL_MIN_FACE_SIZE   # [TEST]
            or
            height < ENROLL_MIN_FACE_SIZE
        ):

            continue

        embedding = normalize_embedding(
            face.normed_embedding
        )

        person_embeddings.append(
            embedding
        )

        print(
            f"Loaded: "
            f"{person_name}/{filename}"
        )

    if len(person_embeddings) > 0:

        average_embedding = np.mean(
            person_embeddings,
            axis=0
        )

        average_embedding = normalize_embedding(
            average_embedding
        )

        known_faces[
            person_name
        ] = average_embedding

        print(
            f"Template created: "
            f"{person_name} "
            f"({len(person_embeddings)} images)"
        )


print()
print("======================================")
print(
    f"Known people loaded: "
    f"{len(known_faces)}"
)
print(
    "Known identities:",
    list(known_faces.keys())
)
print("======================================")
print()


# ============================================================
# LIVE SESSIONS
# ============================================================

sessions = {}


# ============================================================
# RE-ID CACHE
# ============================================================

reid_cache = {}


# ============================================================
# SNAPSHOT TIMING
# ============================================================

last_snapshot_time = {}


# ============================================================
# [TEST] ENTRY / EXIT + RECOGNITION LOGS
# ============================================================

entry_events = []

recognition_log = []
# ============================================================
# [HEADCOUNT] STATE
# ============================================================

headcount_history = deque()     # (video_time, raw_count) for smoothing
headcount_log = []              # rows for headcount.csv
last_headcount_log_time = None

head_count_raw = 0
head_count = 0
peak_head_count = 0
peak_head_count_time = None

# ============================================================
# CREATE SESSION
# ============================================================

def create_session(
    track_id,
    current_time
):

    sessions[track_id] = {

        "track_id": track_id,

        "name": "UNKNOWN",

        "best_score": 0.0,

        "confirmations": 0,

        "status": "TRACKING",

        "first_seen": current_time,

        "last_seen": current_time,

        "zone_entered": False,
        "zone_entry_recognition_done": False,
        "attendance_marked": False,

        "trajectory": deque(
            maxlen=MAX_TRAJECTORY_POINTS
        ),

        "last_position": None,

        "last_embedding": None,

        "recognition_frame_counter": 0,
        "last_recognition_time": 0.0,
        "recognition_count": 0,
        "last_face_box": None,

        "cached": False,
        "merged": False,

        "position_time": None,
        "velocity": (0.0, 0.0),
        "speed": 0.0,

        # [TEST] door / entry state machine
        "entry_state": "NONE",          # NONE / ENTERED / EXITED
        "door_visit_point": None,
        "door_crop": None,
        "last_foot": None,
        "last_crop": None,
        "was_in_room": False,
        "in_door": False,
        "door_since": None,
        "room_since": None,
        "entry_events": [],

    }

    temporary_store.create_session(
        track_id,
        get_timestamp_string()
    )


# ============================================================
# UPDATE POSITION
# ============================================================

def update_session_position(
    session,
    center_x,
    center_y,
    current_time
):

    previous_position = session["last_position"]
    previous_time = session["position_time"]

    session["last_position"] = (
        center_x,
        center_y
    )

    session["position_time"] = current_time

    session["trajectory"].append(
        (
            center_x,
            center_y
        )
    )

    if (
        previous_position is not None
        and
        previous_time is not None
    ):

        dt = current_time - previous_time

        if dt > 0:

            raw_vx = (center_x - previous_position[0]) / dt
            raw_vy = (center_y - previous_position[1]) / dt

            old_vx, old_vy = session["velocity"]

            vx = (
                VELOCITY_SMOOTHING * raw_vx
                +
                (1 - VELOCITY_SMOOTHING) * old_vx
            )

            vy = (
                VELOCITY_SMOOTHING * raw_vy
                +
                (1 - VELOCITY_SMOOTHING) * old_vy
            )

            session["velocity"] = (vx, vy)
            session["speed"] = (vx ** 2 + vy ** 2) ** 0.5

    return previous_position


# ============================================================
# [TEST] DOOR ZONE -> ENTRY / EXIT
# ============================================================

def point_in_door(
    point,
    door_polygon_px
):

    return cv2.pointPolygonTest(
        door_polygon_px,
        (float(point[0]), float(point[1])),
        False
    ) >= 0


def crop_person(
    frame,
    person
):

    h, w = frame.shape[:2]

    x1 = max(0, person["x1"])
    y1 = max(0, person["y1"])
    x2 = min(w, person["x2"])
    y2 = min(h, person["y2"])

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


def update_entry_state(
    session,
    person,
    frame,
    current_time,
    frame_height,
    door_polygon_px
):
    """Returns "ENTRY" or None for this frame (EXIT is decided when the
    track is lost - see check_door_event_on_lost)."""

    # feet = bottom-center of the person box (floor position)
    foot = (
        person["center_x"],
        person["y2"]
    )

    in_door = point_in_door(
        foot,
        door_polygon_px
    )

    session["last_foot"] = foot
    session["last_crop"] = crop_person(frame, person)

    if in_door:

        if not session["in_door"]:
            session["door_since"] = current_time

        session["room_since"] = None

        # first point of this door visit
        if session["door_visit_point"] is None:
            session["door_visit_point"] = foot

        # keep the latest picture of them at the door (for EXIT snapshot)
        session["door_crop"] = crop_person(frame, person)

    else:

        if session["in_door"] or session["room_since"] is None:
            session["room_since"] = current_time

        session["door_since"] = None

        if current_time - session["room_since"] >= ENTRY_CONFIRM_SECONDS:
            session["was_in_room"] = True

    session["in_door"] = in_door

    # ------------------------------------------------
    # ENTRY : was in the door, now clearly inside the room.
    # Works for a track born at the door AND for a track whose ID
    # got swapped onto the person while they stood at the door.
    # ------------------------------------------------
    if (
        session["entry_state"] != "ENTERED"
        and
        session["door_visit_point"] is not None
        and
        not in_door
        and
        session["room_since"] is not None
        and
        current_time - session["room_since"] >= ENTRY_CONFIRM_SECONDS
    ):

        ox, oy = session["door_visit_point"]

        travel = (
            (foot[0] - ox) ** 2
            +
            (foot[1] - oy) ** 2
        ) ** 0.5

        if travel >= ENTRY_MIN_TRAVEL_RATIO * frame_height:

            session["entry_state"] = "ENTERED"
            session["door_visit_point"] = None

            return "ENTRY"

    return None


def check_door_event_on_lost(
    session,
    current_time,
    frame_height
):
    """Called for every session each frame. Once a track has been LOST
    for EXIT_LOST_SECONDS:

    EXIT  - it was inside the room and vanished with its feet in the
            door. (A tracker ID-swap at the door, where the ID jumps to
            someone else and continues into the room, is therefore NOT
            an exit.)
    ENTRY - it came from the door, walked into the room far enough, but
            was lost in the crowd before ENTRY_CONFIRM_SECONDS passed
            (fast walkers who sit down right next to the door)."""

    if current_time - session["last_seen"] < EXIT_LOST_SECONDS:
        return None

    if (
        session["entry_state"] != "EXITED"
        and
        session["in_door"]
        and
        session["was_in_room"]
    ):

        session["entry_state"] = "EXITED"

        return "EXIT"

    if (
        session["entry_state"] != "ENTERED"
        and
        not session["in_door"]
        and
        session["door_visit_point"] is not None
        and
        session["last_foot"] is not None
    ):

        ox, oy = session["door_visit_point"]
        fx, fy = session["last_foot"]

        travel = ((fx - ox) ** 2 + (fy - oy) ** 2) ** 0.5

        session["door_visit_point"] = None

        if travel >= ENTRY_MIN_TRAVEL_RATIO * frame_height:

            session["entry_state"] = "ENTERED"

            return "ENTRY"

    return None


def record_entry_event(
    event,
    session,
    snapshot_image,
    current_time
):

    clock = datetime.datetime.fromtimestamp(current_time)

    snapshot_path = ""

    if (
        SAVE_ENTRY_SNAPSHOTS
        and
        snapshot_image is not None
        and
        snapshot_image.size != 0
    ):

        snapshot_path = os.path.join(
            ENTRY_SNAPSHOT_DIRECTORY,
            f"{event}_{clock.strftime('%H%M%S')}_"
            f"track_{session['track_id']}.jpg"
        )

        cv2.imwrite(
            snapshot_path,
            snapshot_image
        )

    item = {
        "event": event,
        "track_id": int(session["track_id"]),
        "identity_at_event": session["name"],
        "status_at_event": session["status"],
        "video_seconds": round(current_time - VIDEO_START_EPOCH, 2),
        "clock_time": clock.strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot": snapshot_path
    }

    entry_events.append(item)

    session["entry_events"].append(
        {
            "event": event,
            "clock_time": item["clock_time"]
        }
    )

    print(
        f"[{event}] Track={session['track_id']} "
        f"| {session['name']} "
        f"| {item['clock_time']} "
        f"(video {item['video_seconds']:.1f}s)"
    )

    # an entrant gets a face-recognition attempt straight away
    if event == "ENTRY":

        session["recognition_frame_counter"] = (
            RECOGNITION_INTERVAL_FRAMES
        )

        session["last_recognition_time"] = 0.0


# ============================================================
# RE-ID REATTACH
# ============================================================

def try_reid_reattach(
    session,
    embedding,
    current_time
):

    best_name = None

    best_score = 0.0

    for name, entry in reid_cache.items():

        age = (
            current_time
            -
            entry["last_seen"]
        )

        if age > REID_CACHE_SECONDS:
            continue

        if entry.get("track_id") == session["track_id"]:
            continue

        score = cosine_similarity(
            embedding,
            entry["embedding"]
        )

        if score > best_score:

            best_score = score

            best_name = name

    if (
        best_name is not None
        and
        best_score >= REID_MATCH_THRESHOLD
    ):

        old_session = sessions.get(
            reid_cache[best_name].get("track_id")
        )

        if (
            old_session is not None
            and
            old_session is not session
        ):

            old_session["merged"] = True

            if old_session["attendance_marked"]:

                session["attendance_marked"] = True

        session["name"] = best_name

        session["status"] = "CONFIRMED"

        session["confirmations"] = (
            REQUIRED_CONFIRMATIONS
        )

        session["best_score"] = max(
            session["best_score"],
            best_score
        )

        del reid_cache[
            best_name
        ]

        return True

    return False


# ============================================================
# CONFIRM IDENTITY
# ============================================================

def confirm_identity(
    session,
    name,
    score,
    embedding,
    current_time
):

    session["last_seen"] = (
        current_time
    )

    session["last_embedding"] = (
        embedding
    )

    if (
        score
        >
        session["best_score"]
    ):

        session["best_score"] = (
            score
        )

    # --------------------------------------------------------
    # RE-ID CACHE
    # --------------------------------------------------------

    if (
        session["status"]
        ==
        "TRACKING"
        and
        session["confirmations"]
        ==
        0
    ):

        if try_reid_reattach(
            session,
            embedding,
            current_time
        ):

            return

    # --------------------------------------------------------
    # STRONG MATCH
    # --------------------------------------------------------

    if (
        score
        >=
        STRONG_MATCH_THRESHOLD
    ):

        session["name"] = name

        session["confirmations"] += 1

        session["status"] = (
            "CONFIRMED"
        )

    # --------------------------------------------------------
    # NORMAL MATCH
    # --------------------------------------------------------

    elif (
        score
        >=
        MATCH_THRESHOLD
    ):

        if (
            session["name"]
            ==
            "UNKNOWN"
            or
            session["name"]
            ==
            name
        ):

            session["name"] = name

            session["confirmations"] += 1

    # --------------------------------------------------------
    # FINAL CONFIRMATION
    # --------------------------------------------------------

    if (
        session["confirmations"]
        >=
        REQUIRED_CONFIRMATIONS
    ):

        session["status"] = (
            "CONFIRMED"
        )


# ============================================================
# ATTENDANCE
# ============================================================

def mark_attendance(
    name,
    current_time
):

    file_exists = os.path.exists(
        ATTENDANCE_LOG_PATH
    )

    today = datetime.datetime.fromtimestamp(
        current_time
    ).strftime("%Y-%m-%d")

    now_str = datetime.datetime.fromtimestamp(
        current_time
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    if file_exists:

        with open(
            ATTENDANCE_LOG_PATH,
            "r",
            newline=""
        ) as f:

            reader = csv.reader(f)

            next(
                reader,
                None
            )

            for row in reader:

                if (
                    len(row) >= 2
                    and
                    row[0] == name
                    and
                    row[1].startswith(
                        today
                    )
                ):

                    return

    with open(
        ATTENDANCE_LOG_PATH,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow(
                [
                    "name",
                    "timestamp"
                ]
            )

        writer.writerow(
            [
                name,
                now_str
            ]
        )

    print(
        f"[ATTENDANCE] "
        f"{name} marked present "
        f"at {now_str}"
    )


# ============================================================
# FACE CROP BOX
# ============================================================

def get_face_crop_box(
    person,
    frame_width,
    frame_height,
    margin_ratio,
    extra_margin_ratio=0.0
):

    px1 = person["x1"]
    py1 = person["y1"]

    px2 = person["x2"]
    py2 = person["y2"]

    pw = px2 - px1
    ph = py2 - py1

    margin_ratio = margin_ratio + extra_margin_ratio

    crop_height = max(
        int(ph * FACE_CROP_HEIGHT_RATIO),
        min(ph, FACE_CROP_MIN_HEIGHT)
    )

    py2 = py1 + crop_height

    mx = int(
        pw * margin_ratio
    )

    my = int(
        ph * margin_ratio
    )

    cx1 = max(
        0,
        px1 - mx
    )

    cy1 = max(
        0,
        py1 - my
    )

    cx2 = min(
        frame_width,
        px2 + mx
    )

    cy2 = min(
        frame_height,
        py2 + my
    )

    return (
        cx1,
        cy1,
        cx2,
        cy2
    )


# ============================================================
# RECOGNITION PRIORITY
# ============================================================

def recognition_priority(
    person
):

    s = sessions.get(
        person["id"]
    )

    if s is None:

        return (
            2,
            1,
            1,
            0.0
        )

    is_fast = s["speed"] >= FAST_SPEED_PX_PER_SEC

    return (

        1
        if s["status"] == "CONFIRMED"
        else 0,

        # [TEST] people who just walked in go first
        0 if s["entry_state"] == "ENTERED" else 1,

        0 if is_fast else 1,

        s["last_recognition_time"]

    )


# ============================================================
# [TEST] DRAW SCALE (text readable on a 2560x1440 frame)
# ============================================================

DRAW_SCALE = 1.0


def fs(scale):

    return scale * DRAW_SCALE


def th(thickness):

    return max(1, int(round(thickness * DRAW_SCALE)))


# ============================================================
# VIDEO   [TEST] replaces the CAMERA block
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print(
        "ERROR: Video could not be opened:",
        VIDEO_PATH
    )

    exit()

VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS) or 20.0

VIDEO_TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

DRAW_SCALE = max(1.0, VIDEO_WIDTH / 1280.0)

FAST_SPEED_PX_PER_SEC = (
    FAST_SPEED_PX_PER_SEC_AT_960 * VIDEO_WIDTH / 960.0
)

print("Video opened:", VIDEO_NAME)
print(
    f"Video mode: {VIDEO_WIDTH}x{VIDEO_HEIGHT} "
    f"@ {VIDEO_FPS:.1f} FPS | "
    f"{VIDEO_TOTAL_FRAMES / VIDEO_FPS:.0f} s"
)
print("Camera preset:", CAMERA_NAME)
print(
    "Clip start (CCTV clock):",
    datetime.datetime.fromtimestamp(VIDEO_START_EPOCH)
)
print("Results folder:", RUN_DIRECTORY)


# [TEST] HEVC CCTV files give broken grey frames after a hard seek, so
# we decode forward to the start point instead of cap.set(POS_FRAMES).
frame_index = 0

start_frame = int(ARGS.start * VIDEO_FPS)

end_frame = (
    int(ARGS.end * VIDEO_FPS)
    if ARGS.end is not None
    else None
)

if start_frame > 0:

    print(f"Skipping to {ARGS.start:.1f}s ...")

    while frame_index < start_frame:

        if not cap.grab():
            break

        frame_index += 1


# ============================================================
# [TEST] PICK DOOR ZONE (optional)
# ============================================================

if ARGS.pick_zone:

    ok, pick_frame = cap.read()

    if not ok:

        print("Could not read a frame for --pick-zone")

        raise SystemExit(1)

    pick_scale = DISPLAY_WIDTH / VIDEO_WIDTH

    pick_small = cv2.resize(
        pick_frame,
        None,
        fx=pick_scale,
        fy=pick_scale
    )

    clicked = []

    def _on_click(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))

    cv2.namedWindow("Pick door zone")
    cv2.setMouseCallback("Pick door zone", _on_click)

    print("Click the corners of the door area ON THE FLOOR.")
    print("ENTER = done, BACKSPACE = undo, ESC = cancel")

    while True:

        view = pick_small.copy()

        for p in clicked:
            cv2.circle(view, p, 5, (255, 0, 255), -1)

        if len(clicked) >= 2:
            cv2.polylines(
                view,
                [np.array(clicked, np.int32)],
                True,
                (255, 0, 255),
                2
            )

        cv2.imshow("Pick door zone", view)

        k = cv2.waitKey(20) & 0xFF

        if k in (13, 10) and len(clicked) >= 3:
            break

        if k == 8 and clicked:
            clicked.pop()

        if k == 27:
            raise SystemExit(0)

    cv2.destroyAllWindows()

    h_small, w_small = pick_small.shape[:2]

    polygon = [
        (round(x / w_small, 3), round(y / h_small, 3))
        for x, y in clicked
    ]

    print("\nPaste this into CAMERA_PRESETS['" + CAMERA_NAME + "']['door']:")
    print(polygon)

    raise SystemExit(0)


DOOR_POLYGON_PX = np.array(
    [
        (int(x * VIDEO_WIDTH), int(y * VIDEO_HEIGHT))
        for x, y in DOOR_POLYGON_RATIO
    ],
    dtype=np.int32
)


# [TEST] annotated output video
video_writer = None

if not ARGS.no_save_video:

    out_w = DISPLAY_WIDTH
    out_h = int(VIDEO_HEIGHT * DISPLAY_WIDTH / VIDEO_WIDTH)

    video_writer = cv2.VideoWriter(
        ANNOTATED_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS / max(1, ARGS.stride),
        (out_w, out_h)
    )

print(
    "Press Q to quit, SPACE to pause."
)


# ============================================================
# WARM-UP
# ============================================================

print(
    "Warming up models..."
)

try:

    _warm_frame = np.zeros(
        (
            VIDEO_HEIGHT,
            VIDEO_WIDTH,
            3
        ),
        dtype=np.uint8
    )

    model.predict(

        _warm_frame,

        classes=[0],

        device=DEVICE,

        imgsz=YOLO_IMAGE_SIZE,

        half=(GPU_AVAILABLE and YOLO_HALF_PRECISION),

        verbose=False
    )

    face_app.get(
        np.zeros(
            (
                320,
                320,
                3
            ),
            dtype=np.uint8
        )
    )

    print(
        "Warm-up done."
    )

except Exception as e:

    print(
        "Warm-up skipped:",
        e
    )


# ============================================================
# FPS
# ============================================================

previous_wall_time = time.time()   # [TEST] processing speed (PC clock)

fps = 0.0

processed_frames = 0

run_started = time.time()

paused = False


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # [TEST] --stride: skip frames (grab() is cheaper than read())
        skipped_ok = True

        for _ in range(max(1, ARGS.stride) - 1):

            if not cap.grab():
                skipped_ok = False
                break

            frame_index += 1

        if not skipped_ok:
            break

        ret, frame = cap.read()

        if not ret:

            print(
                "End of video."
            )

            break

        frame_index += 1

        if end_frame is not None and frame_index > end_frame:

            print("Reached --end.")

            break

        # [TEST] VIDEO time instead of time.time()
        current_time = (
            VIDEO_START_EPOCH
            +
            (frame_index - 1) / VIDEO_FPS
        )

        CURRENT_VIDEO_TIME = current_time

        processed_frames += 1

        frame_height, frame_width = (
            frame.shape[:2]
        )


        # ====================================================
        # RECOGNITION ZONE
        # ====================================================

        zone_x1 = int(frame_width * ZONE_X1_RATIO)
        zone_y1 = int(frame_height * ZONE_Y1_RATIO)
        zone_x2 = int(frame_width * ZONE_X2_RATIO)
        zone_y2 = int(frame_height * ZONE_Y2_RATIO)


        # ====================================================
        # YOLO PERSON TRACKING
        # ====================================================

        results = model.track(

            frame,

            persist=True,

            tracker=TRACKER_CONFIG,

            classes=[0],

            conf=YOLO_CONFIDENCE,

            iou=YOLO_NMS_IOU,

            device=DEVICE,

            imgsz=YOLO_IMAGE_SIZE,

            half=(GPU_AVAILABLE and YOLO_HALF_PRECISION),

            verbose=False
        )


        result = results[0]

        person_boxes = []


        # ====================================================
        # READ PERSON BOXES
        # ====================================================

        if result.boxes is not None:

            for box in result.boxes:

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                confidence = float(
                    box.conf[0]
                )

                if box.id is not None:

                    track_id = int(
                        box.id[0]
                    )

                else:

                    track_id = -1


                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )


                person = {

                    "id": track_id,

                    "x1": x1,
                    "y1": y1,

                    "x2": x2,
                    "y2": y2,

                    "center_x": center_x,
                    "center_y": center_y,

                    "confidence": confidence
                }


                person_boxes.append(
                    person
                )


                # ============================================
                # CREATE / UPDATE SESSION
                # ============================================

                if track_id != -1:

                    if (
                        track_id
                        not in
                        sessions
                    ):

                        create_session(
                            track_id,
                            current_time
                        )


                    session = sessions[
                        track_id
                    ]


                    session[
                        "last_seen"
                    ] = current_time


                    if session["cached"]:

                        session["cached"] = False

                        entry = reid_cache.get(
                            session["name"]
                        )

                        if (
                            entry is not None
                            and
                            entry.get("track_id") == track_id
                        ):

                            del reid_cache[
                                session["name"]
                            ]


                    previous_position = update_session_position(
                        session,
                        center_x,
                        center_y,
                        current_time
                    )


                    # ========================================
                    # [TEST] DOOR -> ENTRY / EXIT
                    # ========================================

                    door_event = update_entry_state(
                        session,
                        person,
                        frame,
                        current_time,
                        frame_height,
                        DOOR_POLYGON_PX
                    )

                    if door_event is not None:

                        record_entry_event(
                            door_event,
                            session,
                            crop_person(frame, person),
                            current_time
                        )


                    # ========================================
                    # ZONE CHECK
                    # ========================================

                    inside_zone = (

                        zone_x1
                        <=
                        center_x
                        <=
                        zone_x2

                        and

                        zone_y1
                        <=
                        center_y
                        <=
                        zone_y2
                    )


                    if (
                        not inside_zone
                        and
                        INTERPOLATE_ZONE_CROSSING
                        and
                        previous_position is not None
                    ):

                        crossed_zone, _, _ = cv2.clipLine(

                            (
                                zone_x1,
                                zone_y1,
                                zone_x2 - zone_x1,
                                zone_y2 - zone_y1
                            ),

                            previous_position,

                            (center_x, center_y)

                        )

                    else:

                        crossed_zone = False


                    if inside_zone or crossed_zone:

                        if not session["zone_entered"]:

                            session["zone_entered"] = True

                            if ZONE_ENTRY_RECOGNITION:

                                session["recognition_frame_counter"] = (
                                    RECOGNITION_INTERVAL_FRAMES
                                )

                                session["last_recognition_time"] = 0.0
                # ====================================================
        # [HEADCOUNT] HOW MANY PEOPLE ARE INSIDE THE CLASS
        # ====================================================

        head_count_raw = 0

        for p in person_boxes:

            if p["confidence"] < HEADCOUNT_MIN_CONFIDENCE:
                continue

            # feet in the door -> not inside the room yet
            if HEADCOUNT_EXCLUDE_DOOR and point_in_door(
                (p["center_x"], p["y2"]),
                DOOR_POLYGON_PX
            ):
                continue

            head_count_raw += 1

        headcount_history.append((current_time, head_count_raw))

        while (
            headcount_history
            and
            current_time - headcount_history[0][0] > HEADCOUNT_SMOOTH_SECONDS
        ):
            headcount_history.popleft()

        head_count = int(round(float(np.median(
            [c for _, c in headcount_history]
        ))))

        if head_count > peak_head_count:

            peak_head_count = head_count
            peak_head_count_time = current_time

        if (
            last_headcount_log_time is None
            or
            current_time - last_headcount_log_time >= HEADCOUNT_LOG_INTERVAL
        ):

            headcount_log.append(
                {
                    "clock_time": datetime.datetime.fromtimestamp(
                        current_time
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "video_seconds": round(current_time - VIDEO_START_EPOCH, 2),
                    "head_count": head_count,
                    "head_count_raw": head_count_raw
                }
            )

            last_headcount_log_time = current_time


        # ====================================================
        # [TEST] LOST TRACKS -> EXIT / late ENTRY
        # ====================================================

        for session in sessions.values():

            lost_event = check_door_event_on_lost(
                session,
                current_time,
                frame_height
            )

            if lost_event is not None:

                record_entry_event(
                    lost_event,
                    session,
                    session["door_crop"]
                    if lost_event == "EXIT"
                    else session["last_crop"],
                    session["last_seen"]
                )


        # ====================================================
        # LOST TRACKS -> RE-ID CACHE
        # ====================================================

        for (
            track_id,
            session
        ) in sessions.items():

            if (
                session["name"] == "UNKNOWN"
                or
                session["last_embedding"] is None
                or
                session["merged"]
                or
                session["cached"]
            ):

                continue

            cache_delay = (

                LOST_TO_CACHE_SECONDS_FAST
                if session["speed"] >= FAST_SPEED_PX_PER_SEC
                else LOST_TO_CACHE_SECONDS

            )

            if (
                current_time
                -
                session["last_seen"]
                >=
                cache_delay
            ):

                reid_cache[
                    session["name"]
                ] = {

                    "embedding":
                        session[
                            "last_embedding"
                        ],

                    "last_seen":
                        current_time,

                    "track_id":
                        track_id
                }

                session["cached"] = True


        # ====================================================
        # FACE RECOGNITION - SCHEDULED / CACHED
        # ====================================================

        face_checks = 0
        total_faces_this_frame = 0

        recognitions_this_frame = 0

        urgent_recognitions_this_frame = 0

        zone_entry_recognitions_this_frame = 0

        person_boxes.sort(
            key=recognition_priority
        )

        for person in person_boxes:

            track_id = person["id"]

            if track_id == -1:
                continue

            session = sessions.get(track_id)

            if session is None:
                continue

            # [TEST] optional: recognise entrants only
            recognition_allowed = (
                not RECOGNIZE_ONLY_ENTRANTS
                or
                session["entry_state"] == "ENTERED"
            )

            # ------------------------------------------------
            # Recognition scheduler
            # ------------------------------------------------
            session["recognition_frame_counter"] += 1

            if session["status"] == "CONFIRMED":
                interval = CONFIRMED_RECOGNITION_INTERVAL_FRAMES
            else:
                interval = RECOGNITION_INTERVAL_FRAMES

            recognition_due = (
                session["recognition_frame_counter"] >= interval
                or
                (
                    current_time - session["last_recognition_time"]
                    >= MAX_RECOGNITION_GAP_SECONDS
                )
            )

            is_fast = session["speed"] >= FAST_SPEED_PX_PER_SEC

            has_budget = (
                recognitions_this_frame < MAX_RECOGNITIONS_PER_FRAME
            )

            has_urgent_budget = (
                is_fast
                and
                urgent_recognitions_this_frame < MAX_URGENT_RECOGNITIONS_PER_FRAME
            )

            # ====================================================
            # ZONE ENTRY PRIORITY [V5]
            # ====================================================

            zone_entry_priority = (
                ZONE_ENTRY_RECOGNITION
                and
                session["zone_entered"]
                and
                not session["zone_entry_recognition_done"]
            )

            zone_entry_can_run = (
                zone_entry_priority
                and
                zone_entry_recognitions_this_frame
                < MAX_ZONE_ENTRY_RECOGNITIONS_PER_FRAME
            )

            normal_can_run = (
                recognition_due
                and
                (
                    has_budget
                    or
                    has_urgent_budget
                )
            )

            should_recognize = (
                recognition_allowed
                and
                (
                    zone_entry_can_run
                    or
                    normal_can_run
                )
            )

            if should_recognize:

                if zone_entry_can_run:

                    zone_entry_recognitions_this_frame += 1

                elif has_budget:

                    recognitions_this_frame += 1

                elif has_urgent_budget:

                    urgent_recognitions_this_frame += 1

                session["recognition_frame_counter"] = 0

                session["last_recognition_time"] = (
                    current_time
                )

                session["recognition_count"] += 1

                # ============================================
                # PERSON CROP  (from the FULL-resolution frame)
                # ============================================
                cx1, cy1, cx2, cy2 = get_face_crop_box(
                    person,
                    frame_width,
                    frame_height,
                    CROP_MARGIN_RATIO,
                    FAST_CROP_EXTRA_MARGIN_RATIO if is_fast else 0.0
                )

                if cx2 > cx1 and cy2 > cy1:

                    crop = frame[cy1:cy2, cx1:cx2]

                    if crop.size != 0:

                        # ====================================
                        # FACE DETECTION
                        # ====================================
                        faces_in_crop = face_app.get(crop)

                        total_faces_this_frame += len(faces_in_crop)

                        if len(faces_in_crop) > 0:

                            face = max(
                                faces_in_crop,
                                key=lambda f: (
                                    (f.bbox[2] - f.bbox[0])
                                    *
                                    (f.bbox[3] - f.bbox[1])
                                )
                            )

                            fx1, fy1, fx2, fy2 = face.bbox.astype(int)

                            face_width = fx2 - fx1
                            face_height = fy2 - fy1

                            if (
                                face_width >= MIN_FACE_SIZE
                                and face_height >= MIN_FACE_SIZE
                            ):

                                fx1 += cx1
                                fy1 += cy1
                                fx2 += cx1
                                fy2 += cy1

                                face_center_x = int((fx1 + fx2) / 2)
                                face_center_y = int((fy1 + fy2) / 2)

                                # ====================================
                                # FACE MUST BE INSIDE ZONE
                                # ====================================
                                if (
                                    zone_x1 <= face_center_x <= zone_x2
                                    and
                                    zone_y1 <= face_center_y <= zone_y2
                                ):

                                    face_checks += 1
                                    session["last_face_box"] = (
                                        fx1, fy1, fx2, fy2
                                    )

                                    if zone_entry_priority:

                                        session["zone_entry_recognition_done"] = True

                                    # ================================
                                    # EMBEDDING
                                    # ================================
                                    embedding = normalize_embedding(
                                        face.normed_embedding
                                    )

                                    # ================================
                                    # FIND BEST IDENTITY
                                    # ================================
                                    best_name = "UNKNOWN"
                                    best_score = 0.0

                                    for name, known_embedding in known_faces.items():

                                        score = cosine_similarity(
                                            embedding,
                                            known_embedding
                                        )

                                        if score > best_score:
                                            best_score = score
                                            best_name = name

                                    # [TEST] log every face attempt so
                                    # you can tune MATCH_THRESHOLD later
                                    recognition_log.append(
                                        {
                                            "clock_time": datetime.datetime.fromtimestamp(
                                                current_time
                                            ).strftime("%H:%M:%S"),
                                            "video_seconds": round(
                                                current_time - VIDEO_START_EPOCH, 2
                                            ),
                                            "track_id": int(track_id),
                                            "best_name": best_name,
                                            "best_score": round(best_score, 4),
                                            "face_px": int(min(face_width, face_height)),
                                            "det_score": round(float(face.det_score), 3),
                                            "accepted": best_score >= MATCH_THRESHOLD
                                        }
                                    )

                                    # ================================
                                    # CONFIRM
                                    # ================================
                                    if best_score >= MATCH_THRESHOLD:

                                        confirm_identity(
                                            session,
                                            best_name,
                                            best_score,
                                            embedding,
                                            current_time
                                        )

                                    # ================================
                                    # SAVE BEST EMBEDDING
                                    # ================================
                                    if (
                                        session["status"] == "CONFIRMED"
                                        and
                                        session["last_embedding"] is not None
                                    ):

                                        temporary_store.save_embedding(
                                            track_id,
                                            session["last_embedding"]
                                        )

                                    # ================================
                                    # ATTENDANCE
                                    # ================================
                                    if (
                                        session["status"] == "CONFIRMED"
                                        and
                                        not session["attendance_marked"]
                                    ):

                                        attendance_engine.register_presence(
                                            identity=session["name"],
                                            track_id=session["track_id"],
                                            confidence=session["best_score"],
                                            zone_entered=session["zone_entered"]
                                        )

                                        # [TEST] CSV with the CCTV time
                                        mark_attendance(
                                            session["name"],
                                            current_time
                                        )

                                        session["attendance_marked"] = True

                                        print(
                                            f"[ATTENDANCE ENGINE] "
                                            f"{session['name']} marked PRESENT "
                                            f"| Track={session['track_id']} "
                                            f"| Confidence={session['best_score']:.3f}"
                                        )

                                    # ================================
                                    # TEMPORARY PERSON DATA
                                    # ================================
                                    temporary_store.update_session(
                                        track_id,
                                        session
                                    )

                                    # ================================
                                    # SNAPSHOTS
                                    # ================================
                                    last_snapshot = last_snapshot_time.get(
                                        track_id,
                                        0
                                    )

                                    if (
                                        current_time - last_snapshot
                                        >= SNAPSHOT_INTERVAL
                                    ):

                                        temporary_store.save_person_snapshot(
                                            track_id,
                                            frame,
                                            (
                                                person["x1"],
                                                person["y1"],
                                                person["x2"],
                                                person["y2"]
                                            ),
                                            current_time
                                        )

                                        temporary_store.save_face_snapshot(
                                            track_id,
                                            frame,
                                            (fx1, fy1, fx2, fy2),
                                            current_time
                                        )

                                        last_snapshot_time[track_id] = current_time

            # ====================================================
            # DRAW CURRENT TRACK EVERY FRAME
            # ====================================================
            if session["status"] == "CONFIRMED":
                box_color = (0, 255, 0)
            elif session["name"] != "UNKNOWN":
                box_color = (0, 255, 255)
            else:
                box_color = (0, 0, 255)

            # [TEST] entrants get a magenta box until recognised
            if (
                session["entry_state"] == "ENTERED"
                and
                session["status"] != "CONFIRMED"
            ):
                box_color = (255, 0, 255)

            px1 = person["x1"]
            py1 = person["y1"]
            px2 = person["x2"]
            py2 = person["y2"]

            if session["status"] == "CONFIRMED":
                status_text = "CONFIRMED"
            elif session["name"] != "UNKNOWN":
                status_text = (
                    f"CHECK {session['confirmations']}/"
                    f"{REQUIRED_CONFIRMATIONS}"
                )
            else:
                status_text = "UNKNOWN"

            # [TEST]
            if session["entry_state"] == "ENTERED":
                status_text += " | IN"
            elif session["entry_state"] == "EXITED":
                status_text += " | OUT"

            cv2.rectangle(
                frame,
                (px1, py1),
                (px2, py2),
                box_color,
                th(2)
            )

            label = (
                f"{session['name']} | ID {track_id} | "
                f"{session['best_score']:.2f} | {status_text}"
            )

            cv2.putText(
                frame,
                label,
                (px1, max(py1 - th(10), th(25))),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs(0.65),
                box_color,
                th(2)
            )

            cached_face = session.get("last_face_box")

            if cached_face is not None:
                cv2.rectangle(
                    frame,
                    (cached_face[0], cached_face[1]),
                    (cached_face[2], cached_face[3]),
                    box_color,
                    th(1)
                )

        # ====================================================
        # TRAJECTORY
        # ====================================================

        visible_ids = {

            p["id"]

            for p in person_boxes
        }

        if SHOW_TRAJECTORY:

            for (
                track_id,
                session
            ) in sessions.items():

                if track_id not in visible_ids:

                    continue

                trajectory = list(
                    session[
                        "trajectory"
                    ]
                )[
                    -TRAJECTORY_DRAW_POINTS:
                ]


                if len(trajectory) < 2:

                    continue


                n = len(
                    trajectory
                )


                for i in range(
                    1,
                    n
                ):

                    fade = (
                        i / n
                    )


                    color = (

                        int(
                            255 * fade
                        ),

                        int(
                            150 * fade
                        ),

                        0
                    )


                    thickness = (

                        th(1)
                        if fade < 0.5
                        else th(2)
                    )


                    cv2.line(

                        frame,

                        trajectory[
                            i - 1
                        ],

                        trajectory[i],

                        color,

                        thickness
                    )


        # ====================================================
        # UPDATE TEMPORARY STORAGE
        # ====================================================

        for (
            track_id,
            session
        ) in sessions.items():

            temporary_store.update_session(

                track_id,

                session
            )


        # ====================================================
        # EXPIRE SESSIONS
        # ====================================================

        expired_ids = []


        for (
            track_id,
            session
        ) in sessions.items():

            age = (

                current_time
                -
                session[
                    "last_seen"
                ]

            )


            if (
                age
                >
                SESSION_MEMORY_SECONDS
            ):

                expired_ids.append(
                    track_id
                )


        # ====================================================
        # MOVE EXPIRED SESSIONS TO RE-ID CACHE
        # ====================================================

        for track_id in expired_ids:

            session = sessions[
                track_id
            ]


            if (

                session["name"]
                !=
                "UNKNOWN"

                and

                session[
                    "last_embedding"
                ]
                is not None

                and

                not session["merged"]

            ):

                reid_cache[
                    session["name"]
                ] = {

                    "embedding":
                        session[
                            "last_embedding"
                        ],

                    "last_seen":
                        current_time,

                    "track_id":
                        track_id
                }


            temporary_store.update_session(

                track_id,

                session
            )


            temporary_store.save_session(

                track_id
            )


            del sessions[
                track_id
            ]


            if (
                track_id
                in
                last_snapshot_time
            ):

                del last_snapshot_time[
                    track_id
                ]


        # ====================================================
        # PRUNE RE-ID CACHE
        # ====================================================

        stale_names = [

            name

            for name, entry
            in reid_cache.items()

            if (

                current_time
                -
                entry["last_seen"]

            )
            >
            REID_CACHE_SECONDS

        ]


        for name in stale_names:

            del reid_cache[
                name
            ]


        # ====================================================
        # FPS   [TEST] processing speed on the PC clock
        # ====================================================

        wall_now = time.time()

        elapsed = (

            wall_now
            -
            previous_wall_time
        )


        if elapsed > 0:

            instant_fps = (
                1.0 / elapsed
            )

            fps = (

                0.9 * fps
                +
                0.1 * instant_fps
            )


        previous_wall_time = (
            wall_now
        )


        # ====================================================
        # DRAW RECOGNITION ZONE
        # ====================================================

        cv2.rectangle(

            frame,

            (
                zone_x1,
                zone_y1
            ),

            (
                zone_x2,
                zone_y2
            ),

            (
                255,
                255,
                0
            ),

            th(2)
        )


        cv2.putText(

            frame,

            "RECOGNITION ZONE",

            (
                zone_x1 + th(10),
                zone_y2 - th(12)
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            fs(0.8),

            (
                255,
                255,
                0
            ),

            th(2)
        )


        # [TEST] door polygon
        cv2.polylines(
            frame,
            [DOOR_POLYGON_PX],
            True,
            (255, 0, 255),
            th(3)
        )

        cv2.putText(
            frame,
            "DOOR",
            tuple(DOOR_POLYGON_PX[0] + np.array([th(5), -th(10)])),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs(0.9),
            (255, 0, 255),
            th(2)
        )


        # ====================================================
        # INFO PANEL
        # ====================================================

        n_entries = sum(1 for e in entry_events if e["event"] == "ENTRY")
        n_exits = sum(1 for e in entry_events if e["event"] == "EXIT")

        fast_count = sum(
            1
            for s in sessions.values()
            if s["speed"] >= FAST_SPEED_PX_PER_SEC
        )

        panel_lines = [
                        (
                f"HEAD COUNT: {head_count}   "
                f"(raw {head_count_raw} | peak {peak_head_count})",
                (0, 255, 255)
            ),

            (f"FPS: {fps:.1f}", (0, 255, 0)),

            (
                f"CCTV: "
                f"{datetime.datetime.fromtimestamp(current_time).strftime('%H:%M:%S')}  "
                f"({current_time - VIDEO_START_EPOCH:.0f}s)",
                (0, 255, 0)
            ),

            (f"GPU: {GPU_AVAILABLE} | Face: {FACE_DEVICE}", (0, 255, 0)),

            (f"ENTRIES: {n_entries}   EXITS: {n_exits}", (255, 0, 255)),

            (f"Active Sessions: {len(sessions)}", (0, 255, 255)),

            (
                f"Face Check: {face_checks}/{total_faces_this_frame}",
                (255, 255, 0)
            ),

            (f"Re-ID cache: {len(reid_cache)}", (200, 200, 255)),

            (f"Fast movers: {fast_count}", (0, 165, 255)),

        ]

        panel_y = int(frame_height * 0.10)   # below the CCTV clock

        for text, color in panel_lines:

            cv2.putText(
                frame,
                text,
                (th(20), panel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs(0.7),
                (0, 0, 0),
                th(5)
            )

            cv2.putText(
                frame,
                text,
                (th(20), panel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs(0.7),
                color,
                th(2)
            )

            panel_y += th(32)


        # ====================================================
        # [TEST] LAST 6 ENTRY / EXIT EVENTS
        # ====================================================

        for e in entry_events[-6:]:

            cv2.putText(
                frame,
                f"{e['clock_time'][11:]} {e['event']} "
                f"ID {e['track_id']} {e['identity_at_event']}",
                (th(20), panel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs(0.6),
                (255, 255, 255),
                th(2)
            )

            panel_y += th(26)


        # ====================================================
        # DISPLAY  [TEST] downscaled; processing stays full-res
        # ====================================================

        display_frame = cv2.resize(
            frame,
            (
                DISPLAY_WIDTH,
                int(frame_height * DISPLAY_WIDTH / frame_width)
            )
        )

        if video_writer is not None:

            video_writer.write(display_frame)

        if processed_frames % 100 == 0:

            done = (
                f"{100 * frame_index / VIDEO_TOTAL_FRAMES:.0f}%"
                if VIDEO_TOTAL_FRAMES
                else ""
            )

            print(
                f"  frame {frame_index} {done} "
                f"| {fps:.1f} FPS "
                f"| entries {n_entries} exits {n_exits}"
            )

        if not ARGS.no_show:

            cv2.imshow(

                f"AttendAI - Video Test ({CAMERA_NAME})",

                display_frame
            )

            key = cv2.waitKey(0 if paused else 1) & 0xFF

            if key == ord(" "):

                paused = not paused

            if key == ord("q"):

                break


# ============================================================
# SAFE SHUTDOWN
# ============================================================

finally:

    print()
    print(
        "Saving temporary sessions..."
    )

    # [TEST] final update of every still-active session
    for track_id, session in sessions.items():

        temporary_store.update_session(
            track_id,
            session
        )

    temporary_store.save_all()

    cap.release()

    if video_writer is not None:

        video_writer.release()

    attendance_engine.finalize()

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass    # headless OpenCV build


    # ========================================================
    # [TEST] WRITE ENTRY / RECOGNITION RESULTS
    # ========================================================

    # identity can be confirmed AFTER the entry event, so fill in the
    # final identity of each track here
    for e in entry_events:

        stored = temporary_store.memory.get(
            f"track_{e['track_id']}",
            {}
        )

        e["final_identity"] = stored.get("identity", "UNKNOWN")
        e["final_status"] = stored.get("status", "TRACKING")
        e["final_confidence"] = round(
            float(stored.get("best_confidence", 0.0)),
            3
        )

    entry_fields = [
        "event",
        "clock_time",
        "video_seconds",
        "track_id",
        "identity_at_event",
        "final_identity",
        "final_status",
        "final_confidence",
        "snapshot"
    ]

    with open(ENTRIES_CSV_PATH, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=entry_fields,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(entry_events)

    recognition_fields = [
        "clock_time",
        "video_seconds",
        "track_id",
        "best_name",
        "best_score",
        "face_px",
        "det_score",
        "accepted"
    ]

    with open(RECOGNITIONS_CSV_PATH, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=recognition_fields
        )

        writer.writeheader()
        writer.writerows(recognition_log)

    summary = (
        temporary_store.get_summary()
    )

    entries = [e for e in entry_events if e["event"] == "ENTRY"]
    exits = [e for e in entry_events if e["event"] == "EXIT"]

    recognised_entrants = sorted({
        e["final_identity"]
        for e in entries
        if e["final_status"] == "CONFIRMED"
    })

    confirmed_people = sorted({
        item["identity"]
        for item in temporary_store.memory.values()
        if item["status"] == "CONFIRMED"
    })

    run_seconds = time.time() - run_started

    test_summary = {
        "video": VIDEO_NAME,
        "camera": CAMERA_NAME,
        "clip_start": datetime.datetime.fromtimestamp(
            VIDEO_START_EPOCH
        ).strftime("%Y-%m-%d %H:%M:%S"),
        "processed_frames": processed_frames,
        "stride": ARGS.stride,
        "processing_seconds": round(run_seconds, 1),
        "average_fps": round(processed_frames / max(run_seconds, 1e-6), 2),
        "total_tracks": summary["total_sessions"],
        "entries": len(entries),
        "exits": len(exits),
        "unknown_entries": sum(
            1 for e in entries if e["final_status"] != "CONFIRMED"
        ),
        "recognised_entrants": recognised_entrants,
        "confirmed_people": confirmed_people,
        "face_attempts": len(recognition_log),
        "face_matches": sum(1 for r in recognition_log if r["accepted"])
    }
    
        # ========================================================
    # [HEADCOUNT] SAVE CSV + ADD TO SUMMARY
    # ========================================================

    with open(HEADCOUNT_CSV_PATH, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clock_time",
                "video_seconds",
                "head_count",
                "head_count_raw"
            ]
        )

        writer.writeheader()
        writer.writerows(headcount_log)

    test_summary.update({
        "final_head_count": head_count,
        "peak_head_count": peak_head_count,
        "peak_head_count_time": (
            datetime.datetime.fromtimestamp(peak_head_count_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if peak_head_count_time is not None
            else ""
        ),
        "average_head_count": round(
            float(np.mean([r["head_count"] for r in headcount_log])), 1
        ) if headcount_log else 0
    })

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:

        json.dump(test_summary, f, indent=4)

    print()
    print(
        "======================================"
    )

    print(
        "ATTENDAI VIDEO TEST SUMMARY"
    )

    print(
        "======================================"
    )

    for key, value in test_summary.items():

        print(f"{key}: {value}")

    print()
    print("Results folder:", RUN_DIRECTORY)
    print("  entries.csv       - every ENTRY / EXIT with time + identity")
    print("  entry_snapshots/  - photo of each person at the door")
    print("  recognitions.csv  - every face match score (tune thresholds)")
    print("  annotated.mp4     - the video with boxes, door, IDs")
    print("  summary.json")

    print(
        "======================================"
    )

    print(
        "\nAttendAI video test stopped."
    )
    print("  headcount.csv     - people inside the class, every second")
