import os
import sys

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import cv2
import csv
import time
import json
import shutil
import datetime
import logging
import warnings
from collections import deque

import numpy as np
import torch

from app.face.fusion_engine import fuse_identity
# [FIX] Load the CUDA 12 / cuDNN 9 DLLs that onnxruntime-gpu needs
# (cublasLt64_12.dll, cublas64_12.dll, cudart64_12.dll, cufft64_11.dll ...)
# from the pip "nvidia-*-cu12" packages. Your PyTorch is built for CUDA 13,
# so onnxruntime cannot borrow PyTorch's DLLs - without this InsightFace
# silently falls back to CPU. Must run BEFORE InsightFace creates sessions.
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
# [FIX] SILENCE LOG SPAM
# ============================================================
# Ultralytics prints "'half' is deprecated" on EVERY frame. half=True
# still works, so only the message is filtered - behaviour unchanged.

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

# InsightFace's np.linalg.lstsq rcond FutureWarning (harmless)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="insightface"
)

# ============================================================
# ATTENDAI - SMART PERSON + FACE TRACKING
# ============================================================
#
# Pipeline:
#
# Camera
#   ↓
# YOLO Person Detection
#   ↓
# BoT-SORT Tracking
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
# ============================================================
#
# v3 = your exact pipeline, with ONLY tracking-accuracy changes.
# Every changed / added line is marked with:   # [V3]
# Bug fixes for the runtime errors are marked: # [FIX]
#
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 1

YOLO_MODEL = "yolo11n.pt"

TRACKER_CONFIG = "trackers/fast_botsort.yaml"

KNOWN_FACES_DIR = "app/face/known_faces"


# ============================================================
# FACE RECOGNITION
# ============================================================

MATCH_THRESHOLD = 0.50

STRONG_MATCH_THRESHOLD = 0.62

REQUIRED_CONFIRMATIONS = 2

MIN_FACE_SIZE = 45


# ============================================================
# YOLO
# ============================================================

# [V3] Low score on purpose. The tracker needs to SEE the blurred,
# low-confidence boxes of a fast walker to keep the same ID
# (BoT-SORT's 2nd association). New tracks are still only created
# from strong detections (new_track_thresh in the yaml), so this
# does not add false people.
YOLO_CONFIDENCE = 0.20

# [V3] Slightly stricter NMS -> fewer duplicate boxes on one person
# (duplicates are a classic cause of split / swapped IDs).
YOLO_NMS_IOU = 0.50

YOLO_IMAGE_SIZE = 416
YOLO_HALF_PRECISION = True

# [V3] Used automatically ONLY when a GPU is found (see GPU section).
# CPU keeps your original yolo11n @ 416.
YOLO_MODEL_GPU = "yolo11s.pt"
YOLO_IMAGE_SIZE_GPU = 640


# ============================================================
# CAMERA
# ============================================================

FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# [V3] Many USB webcams only give ~10 FPS at 1280x720 in the default
# format. MJPG gives the real 30 FPS. Fewer frames = bigger jump per
# frame = lost tracks, so this alone is a big tracking fix.
CAMERA_USE_MJPG = True
CAMERA_FPS = 30

# [V3] Optional anti-motion-blur. None = leave camera auto-exposure alone.
# If people look smeared when walking, try -6 (then -5, -7...).
# Value scale is camera / driver specific.
CAMERA_EXPOSURE = None


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

# [V3] A confirmed person who has not been seen for this long is put
# into the re-ID cache immediately (before: only after the 10 s session
# expiry). If the tracker gives them a NEW id, they are recognised again
# at once instead of restarting confirmation.
LOST_TO_CACHE_SECONDS = 1.0

# [V4] Same idea but shorter, used ONLY for people who were moving fast
# when last seen - a fast mover who drops out of tracking needs to be
# re-attachable almost instantly, or they'll be gone before the normal
# 1.0s window even elapses.
LOST_TO_CACHE_SECONDS_FAST = 0.35


# ============================================================
# [V4] FAST-MOTION TRACKING
# ============================================================
# Everything below is about people who move/run quickly through the
# frame. Nothing here touches face matching (confirm_identity) or
# storage - only how position/zone/priority are tracked.

# Smoothing for the per-track velocity estimate (0-1). Higher = follows
# sudden speed changes faster; lower = steadier but slower to react.
VELOCITY_SMOOTHING = 0.6

# Speed (pixels/second, on the actual frame resolution) above which a
# person is treated as "fast" and gets tracking priority.
FAST_SPEED_PX_PER_SEC = 550

# A fast person can visibly cross the recognition-zone LINE in the gap
# between two frames, so a single "is the center point inside the zone
# right now" check can miss them entirely. When True, the path they
# travelled since the last frame is also checked against the zone.
INTERPOLATE_ZONE_CROSSING = True

# Extra face-recognition attempts per frame reserved ONLY for people
# flagged "fast" (on top of MAX_RECOGNITIONS_PER_FRAME). This is what
# stops a runner from leaving the zone with zero recognition attempts
# because the normal per-frame budget was spent on someone else.
MAX_URGENT_RECOGNITIONS_PER_FRAME = 2

# [V5] Recognition slots reserved for people entering the zone.
# This gets priority over normal scheduled recognition.
MAX_ZONE_ENTRY_RECOGNITIONS_PER_FRAME = 2

# Fast movement + camera-frame timing can make the detected person box
# lag slightly behind the real head position. A bit of extra crop
# margin, scaled by speed, keeps the face inside the crop instead of
# clipping it off the edge.
FAST_CROP_EXTRA_MARGIN_RATIO = 0.12


# ============================================================
# FACE CROP
# ============================================================

CROP_MARGIN_RATIO = 0.15

# [V3] Face detection now runs on the HEAD / UPPER-BODY part of the
# person box instead of the whole body. A whole-body crop gets shrunk
# to the 320x320 detector input, making the face tiny, so walking
# people were missed. Upper part = face stays big = far better hits.
FACE_CROP_HEIGHT_RATIO = 0.50
FACE_CROP_MIN_HEIGHT = 240


# ============================================================
# ATTENDANCE
# ============================================================

ATTENDANCE_LOG_PATH = "attendance_log.csv"


# ============================================================
# TEMPORARY DAY STORAGE
# ============================================================

TEMP_STORAGE_ROOT = "temporary_day_data"

SAVE_PERSON_SNAPSHOTS = True

SAVE_FACE_SNAPSHOTS = True

SAVE_EMBEDDINGS = True

SNAPSHOT_INTERVAL = 1.5

attendance_engine = AttendanceEngine(
    storage_root=TEMP_STORAGE_ROOT
)

# ============================================================
# FAST RECOGNITION SCHEDULING
# ============================================================
# YOLO/BoT-SORT continues every frame. InsightFace is expensive,
# so only run it periodically for each track.
RECOGNITION_INTERVAL_FRAMES = 8
CONFIRMED_RECOGNITION_INTERVAL_FRAMES = 24
MAX_RECOGNITION_GAP_SECONDS = 2.0

# [V3] At most this many face-recognition runs per frame. When several
# people were "due" on the same frame, the loop stalled for a long
# time, so the tracker saw a big time-jump. Now the work is spread over
# consecutive frames (most-overdue / unconfirmed people first), keeping
# every frame the same short length = smooth tracking.
MAX_RECOGNITIONS_PER_FRAME = 1

# ============================================================
# ZONE ENTRY SNAPSHOT RECOGNITION
# ============================================================

ZONE_ENTRY_RECOGNITION = True


# ============================================================
# RECOGNITION ZONE
# ============================================================

ZONE_X1_RATIO = 0.10
ZONE_Y1_RATIO = 0.10

ZONE_X2_RATIO = 0.90
ZONE_Y2_RATIO = 0.95


# ============================================================
# GPU
# ============================================================

GPU_AVAILABLE = torch.cuda.is_available()

if GPU_AVAILABLE:

    DEVICE = 0

    # [V3] GPU can afford the more accurate detector + bigger input.
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


# ============================================================
# DATE / TIME HELPERS
# ============================================================

def get_today_string():

    return datetime.datetime.now().strftime("%Y-%m-%d")


def get_timestamp_string():

    return datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


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

        os.makedirs(
            self.sessions_directory,
            exist_ok=True
        )

        os.makedirs(
            self.people_directory,
            exist_ok=True
        )

        os.makedirs(
            self.face_directory,
            exist_ok=True
        )

        os.makedirs(
            self.embedding_directory,
            exist_ok=True
        )

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

            # [FIX] duplicate "zone_entry_recognition_done" key removed
            "zone_entry_recognition_done": False,

            "attendance_marked": False,

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

        # [FIX] keep the stored copy in sync
        data["zone_entry_recognition_done"] = bool(
            session["zone_entry_recognition_done"]
        )

        data["attendance_marked"] = bool(
            session["attendance_marked"]
        )

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

# [FIX] Reports which provider InsightFace is REALLY using. Before, the
# code printed "GPU initialized" even when onnxruntime had silently
# fallen back to CPU (the cublasLt64_12.dll errors).
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

    # [FIX] check the real provider instead of assuming GPU
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

    # [FIX] real CPU fallback (was asking for CUDA + ctx_id=0 again)
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
            width < MIN_FACE_SIZE
            or
            height < MIN_FACE_SIZE
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

        # [V3] re-ID hand-over bookkeeping
        "cached": False,
        "merged": False,

        # [V4] fast-motion tracking
        "position_time": None,
        "velocity": (0.0, 0.0),
        "speed": 0.0,

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

    # [V4] capture where/when they were BEFORE this update, to compute
    # velocity and to let the caller check the path just travelled
    # (used for interpolated zone-crossing detection on fast movers).
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

    # [V4] velocity / speed estimate (pixels per second), smoothed so a
    # single noisy frame doesn't flip a person's "fast" status.
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

        # [V3] never hand a track its own cached entry
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

        # [V3] the old (lost) track of this person is now "merged" into
        # this new track: don't re-cache it later, and carry attendance
        # over so it is never counted twice.
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

    # --------------------------------------------------------
    # CHECK DUPLICATE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

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
    extra_margin_ratio=0.0  # [V4] widened for fast movers
):

    px1 = person["x1"]
    py1 = person["y1"]

    px2 = person["x2"]
    py2 = person["y2"]

    pw = px2 - px1
    ph = py2 - py1

    # [V4]
    margin_ratio = margin_ratio + extra_margin_ratio

    # [V3] only the upper part of the body (head + shoulders),
    # never smaller than FACE_CROP_MIN_HEIGHT (or the whole box if
    # the person is very close / the box is small).
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
# RECOGNITION PRIORITY   [V3]
# ============================================================
# Who gets the (limited) face-recognition slot of this frame:
# unconfirmed people first, and among them the one waiting longest.

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
            0.0
        )

    # [V4] among the not-yet-confirmed, a fast-moving person (likely to
    # leave the zone soon) goes ahead of a slow/stationary one.
    is_fast = s["speed"] >= FAST_SPEED_PX_PER_SEC

    return (

        1
        if s["status"] == "CONFIRMED"
        else 0,

        0 if is_fast else 1,

        s["last_recognition_time"]

    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX, cv2.CAP_DSHOW
)

if not cap.isOpened():

    print(
        "ERROR: Camera could not be opened."
    )

    exit()


# [V3] MJPG must be requested BEFORE width/height/fps so the driver
# offers the 30 FPS modes.
if CAMERA_USE_MJPG:

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

# [V3]
cap.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

# [V3] optional manual exposure (less motion blur)
if CAMERA_EXPOSURE is not None:

    cap.set(
        cv2.CAP_PROP_AUTO_EXPOSURE,
        0.25
    )

    cap.set(
        cv2.CAP_PROP_EXPOSURE,
        CAMERA_EXPOSURE
    )


print(
    "Camera started."
)

# [V3] shows what the camera REALLY gives you. If FPS here is ~10,
# the camera/driver is the tracking bottleneck, not the code.
print(
    f"Camera mode: "
    f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
    f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
    f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} FPS"
)

print(
    "Press Q to quit."
)


# ============================================================
# WARM-UP   [V3]
# ============================================================
# The very first inference is slow (GPU kernels / model init) and
# would make the first tracked frames jump. Do it once, up front.

print(
    "Warming up models..."
)

try:

    _warm_frame = np.zeros(
        (
            FRAME_HEIGHT,
            FRAME_WIDTH,
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

previous_time = time.time()

fps = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "Could not read frame."
            )

            break

        current_time = time.time()

        frame_height, frame_width = (
            frame.shape[:2]
        )


        # ====================================================
        # RECOGNITION ZONE
        # ====================================================

        zone_x1 = int(
            frame_width
            *
            ZONE_X1_RATIO
        )

        zone_y1 = int(
            frame_height
            *
            ZONE_Y1_RATIO
        )

        zone_x2 = int(
            frame_width
            *
            ZONE_X2_RATIO
        )

        zone_y2 = int(
            frame_height
            *
            ZONE_Y2_RATIO
        )


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


                    # [V3] the tracker found this person again under
                    # the SAME id -> withdraw their re-ID cache entry.
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
                        current_time  # [V4]
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


                    # [V4] A fast person can hop clean over the zone
                    # between two frames, so the center-point check above
                    # alone can miss them. Also check whether the path
                    # they just walked crosses the zone rectangle.
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

                        # ZONE ENTRY

                        # [FIX] only on the FIRST entry. Before, this
                        # also set zone_entry_recognition_done = True
                        # here, so the V5 zone-entry recognition below
                        # could never trigger. "done" is now set only
                        # after a face is really captured (V5 block).
                        if not session["zone_entered"]:

                            session["zone_entered"] = True

                            if ZONE_ENTRY_RECOGNITION:

                                session["recognition_frame_counter"] = (
                                    RECOGNITION_INTERVAL_FRAMES
                                )

                                session["last_recognition_time"] = 0.0


        # ====================================================
        # LOST TRACKS -> RE-ID CACHE   [V3]
        # ====================================================
        # A confirmed person who just disappeared from the tracker is
        # made available for re-attachment right away, so if they come
        # back with a new track id (fast walk / blur / crossing) the
        # new id inherits the identity immediately.

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

            # [V4] someone who was running when last seen gets handed to
            # the re-ID cache much sooner than a slow/stationary person.
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
        # Person tracking runs every frame, but InsightFace is the
        # expensive part.  We only run face recognition periodically
        # for each track and reuse the last result between checks.

        face_checks = 0
        total_faces_this_frame = 0

        # [V3] limited recognition budget per frame
        recognitions_this_frame = 0

        # [V4] separate, smaller budget spent ONLY on fast movers, on
        # top of the normal budget above.
        urgent_recognitions_this_frame = 0
        # [V5] Separate budget for first recognition when entering the zone.
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

            # [V4] a fast mover gets to use the urgent budget once the
            # normal budget for this frame is spent, instead of just
            # waiting for its next scheduled turn (which might be after
            # it has already left the zone).
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
            # If this person has entered the recognition zone,
            # give them an immediate face-recognition attempt.
            # This is in addition to the normal scheduler.

            zone_entry_priority = (
                ZONE_ENTRY_RECOGNITION
                and
                session["zone_entered"]
                and
                not session["zone_entry_recognition_done"]
            )

            # ============================================================
            # [V5] RECOGNITION DECISION
            # ============================================================

            # Zone-entry recognition gets priority.
            zone_entry_can_run = (
                zone_entry_priority
                and
                zone_entry_recognitions_this_frame
                < MAX_ZONE_ENTRY_RECOGNITIONS_PER_FRAME
            )

            # Normal scheduled recognition.
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
                zone_entry_can_run
                or
                normal_can_run
            )

            if should_recognize:

                # ========================================================
                # [V5] PRIORITY 1 — ZONE ENTRY
                # ========================================================

                if zone_entry_can_run:

                    zone_entry_recognitions_this_frame += 1

                    print(
                        f"[ZONE ENTRY] "
                        f"Recognition triggered "
                        f"| Track={track_id}"
                    )

                # ========================================================
                # [V5] PRIORITY 2 — NORMAL RECOGNITION
                # ========================================================

                elif has_budget:

                    recognitions_this_frame += 1

                # ========================================================
                # [V4] PRIORITY 3 — FAST MOVER
                # ========================================================

                elif has_urgent_budget:

                    urgent_recognitions_this_frame += 1

                # Reset recognition scheduler
                session["recognition_frame_counter"] = 0

                session["last_recognition_time"] = (
                    current_time
                )

                session["recognition_count"] += 1

                # ============================================
                # PERSON CROP
                # ============================================
                cx1, cy1, cx2, cy2 = get_face_crop_box(
                    person,
                    frame_width,
                    frame_height,
                    CROP_MARGIN_RATIO,
                    # [V4] extra margin scaled in for fast movers
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

                                # Map crop coordinates back to full frame.
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

                                    # ========================================================
                                    # [V5] SUCCESSFUL ZONE-ENTRY FACE CAPTURE
                                    # ========================================================
                                    # [FIX] duplicated block removed (printed twice)

                                    if zone_entry_priority:

                                        session["zone_entry_recognition_done"] = True

                                        print(
                                            f"[ZONE ENTRY SNAP] "
                                            f"Face captured successfully "
                                            f"| Track={track_id}"
                                        )

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

                                   # ============================================================
                                    # ADAPTIVE FUSION
                                    # ============================================================

                                    fusion_result = fuse_identity(
                                        track_id=track_id,

                                        # Face evidence
                                        face_identity=best_name,
                                        face_score=best_score,

                                        # Body/Gait will be connected when their engines are ready
                                        body_identity=None,
                                        body_score=None,

                                        gait_identity=None,
                                        gait_score=None
                                    )

                                    # Store fusion result in the current tracking session
                                    session["fusion_result"] = fusion_result

                                    fusion_identity = fusion_result["identity"]
                                    fusion_confidence = fusion_result["confidence"]
                                    fusion_status = fusion_result["status"]

                                    print(
                                        f"[FUSION] "
                                        f"Track={track_id} | "
                                        f"Identity={fusion_identity} | "
                                        f"Confidence={fusion_confidence:.3f} | "
                                        f"Status={fusion_status}"
                                    )

                                    # ============================================================
                                    # FINAL IDENTITY DECISION
                                    # ============================================================

                                    if fusion_status == "CONFIRMED":

                                        confirm_identity(
                                            session,
                                            fusion_identity,
                                            fusion_confidence,
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

                                        # [FIX] was never set, so the
                                        # engine was called (and printed)
                                        # on every recognition.
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

            cv2.rectangle(
                frame,
                (px1, py1),
                (px2, py2),
                box_color,
                2
            )

            label = (
                f"{session['name']} | ID {track_id} | "
                f"{session['best_score']:.2f} | {status_text}"
            )

            cv2.putText(
                frame,
                label,
                (px1, max(py1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2
            )

            # Draw the last known face box even when recognition is skipped.
            cached_face = session.get("last_face_box")

            if cached_face is not None:
                cv2.rectangle(
                    frame,
                    (cached_face[0], cached_face[1]),
                    (cached_face[2], cached_face[3]),
                    box_color,
                    1
                )

        # ====================================================
        # TRAJECTORY
        # ====================================================

        # [V3] only people currently visible get a trail, so old
        # "ghost" sessions no longer leave stray lines on screen.
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

                        1
                        if fade < 0.5
                        else 2
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


            # [V3] "merged" sessions already handed their identity to a
            # newer track, so they must not be cached again.
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


            # Save final session

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


            # Snapshot timing cleanup

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
        # FPS
        # ====================================================

        elapsed = (

            current_time
            -
            previous_time
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


        previous_time = (
            current_time
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

            2
        )


        cv2.putText(

            frame,

            "RECOGNITION ZONE",

            (
                zone_x1,
                zone_y1 - 12
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (
                255,
                255,
                0
            ),

            2
        )


        # ====================================================
        # INFO PANEL
        # ====================================================

        cv2.putText(

            frame,

            f"FPS: {fps:.1f}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"GPU: {GPU_AVAILABLE}",

            (20, 75),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"Face: {FACE_DEVICE}",

            (20, 110),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"Active Sessions: {len(sessions)}",

            (20, 145),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 255),

            2
        )


        cv2.putText(

            frame,

            f"Face Check: "
            f"{face_checks}/"
            f"{total_faces_this_frame}",

            (20, 180),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 0),

            2
        )


        cv2.putText(

            frame,

            f"Re-ID cache: "
            f"{len(reid_cache)}",

            (20, 215),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (200, 200, 255),

            2
        )


        # [V4] how many currently-tracked people are flagged "fast" -
        # quick visual check that fast-motion handling is engaging.
        fast_count = sum(
            1
            for s in sessions.values()
            if s["speed"] >= FAST_SPEED_PX_PER_SEC
        )

        cv2.putText(

            frame,

            f"Fast movers: {fast_count}",

            (20, 250),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 165, 255),

            2
        )


        # ====================================================
        # SESSION LIST
        # ====================================================

        display_y = 285  # [V4] moved down to make room for "Fast movers" line


        for (
            track_id,
            session
        ) in sessions.items():

            if (
                display_y
                >
                frame_height - 30
            ):

                break


            session_text = (

                f"ID {track_id}: "
                f"{session['name']} "
                f"| "
                f"{session['status']} "
                f"| Conf "
                f"{session['confirmations']}"
                # [V4] live speed readout, helps tune FAST_SPEED_PX_PER_SEC
                f"| {int(session['speed'])}px/s"
                f"{'  FAST' if session['speed'] >= FAST_SPEED_PX_PER_SEC else ''}"

            )


            cv2.putText(

                frame,

                session_text,

                (
                    20,
                    display_y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (
                    255,
                    255,
                    255
                ),

                1
            )


            display_y += 25


        # ====================================================
        # STORAGE STATUS
        # ====================================================

        cv2.putText(

            frame,

            "TEMP DAY STORAGE: ACTIVE",

            (
                20,
                frame_height - 45
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (
                0,
                255,
                255
            ),

            1
        )


        cv2.putText(

            frame,

            temporary_store.today,

            (
                20,
                frame_height - 20
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (
                200,
                200,
                200
            ),

            1
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(

            "AttendAI - Smart Recognition",

            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


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

    temporary_store.save_all()

    cap.release()

    # ==============================================
    # FINALIZE ATTENDANCE
    # ==============================================

    attendance_engine.finalize()

    cv2.destroyAllWindows()

    print()
    print(
        "======================================"
    )

    print(
        "ATTENDAI SESSION SUMMARY"
    )

    print(
        "======================================"
    )

    summary = (
        temporary_store.get_summary()
    )

    print(
        "Date:",
        summary["date"]
    )

    print(
        "Total sessions:",
        summary["total_sessions"]
    )

    print(
        "Confirmed people:",
        summary["confirmed_people"]
    )

    print(
        "Temporary storage:",
        temporary_store.day_directory
    )

    print(
        "======================================"
    )

    print(
        "\nAttendAI stopped."
    )