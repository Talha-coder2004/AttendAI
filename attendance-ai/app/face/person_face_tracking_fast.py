import cv2
import os
import csv
import time
import json
import shutil
import datetime
from collections import deque

import numpy as np
import torch

from ultralytics import YOLO
from insightface.app import FaceAnalysis


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


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 2

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

YOLO_CONFIDENCE = 0.45


# ============================================================
# CAMERA
# ============================================================

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


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


# ============================================================
# FACE CROP
# ============================================================

CROP_MARGIN_RATIO = 0.15


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

# ============================================================
# FAST RECOGNITION SCHEDULING
# ============================================================
# YOLO/BoT-SORT continues every frame. InsightFace is expensive,
# so only run it periodically for each track.
RECOGNITION_INTERVAL_FRAMES = 4
CONFIRMED_RECOGNITION_INTERVAL_FRAMES = 30
MAX_RECOGNITION_GAP_SECONDS = 2.0


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

    print("======================================")
    print("GPU AVAILABLE")
    print("GPU:", torch.cuda.get_device_name(0))
    print("======================================")

else:

    DEVICE = "cpu"

    print("======================================")
    print("GPU NOT AVAILABLE")
    print("Using CPU")
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
        det_size=(320, 320)
    )

    FACE_DEVICE = "GPU"

    print(
        "InsightFace GPU initialized."
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
        det_size=(320, 320)
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

        "attendance_marked": False,

        "trajectory": deque(
            maxlen=MAX_TRAJECTORY_POINTS
        ),

        "last_position": None,

        "last_embedding": None,

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
    center_y
):

    session["last_position"] = (
        center_x,
        center_y
    )

    session["trajectory"].append(
        (
            center_x,
            center_y
        )
    )


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
    margin_ratio
):

    px1 = person["x1"]
    py1 = person["y1"]

    px2 = person["x2"]
    py2 = person["y2"]

    pw = px2 - px1
    ph = py2 - py1

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


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


print(
    "Camera started."
)

print(
    "Press Q to quit."
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

            device=DEVICE,

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


                    update_session_position(
                        session,
                        center_x,
                        center_y
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


                    if inside_zone:

                        session[
                            "zone_entered"
                        ] = True


        # ====================================================
        # FACE RECOGNITION
        # ====================================================

        face_checks = 0

        total_faces_this_frame = 0


        for person in person_boxes:

            track_id = person["id"]

            if track_id == -1:
                continue


            session = sessions.get(
                track_id
            )

            if session is None:
                continue


            # ================================================
            # PERSON CROP
            # ================================================

            cx1, cy1, cx2, cy2 = (
                get_face_crop_box(
                    person,
                    frame_width,
                    frame_height,
                    CROP_MARGIN_RATIO
                )
            )


            if (
                cx2 <= cx1
                or
                cy2 <= cy1
            ):

                continue


            crop = frame[
                cy1:cy2,
                cx1:cx2
            ]


            if crop.size == 0:
                continue


            # ================================================
            # FACE DETECTION
            # ================================================

            faces_in_crop = face_app.get(
                crop
            )


            total_faces_this_frame += (
                len(faces_in_crop)
            )


            if len(faces_in_crop) == 0:

                continue


            # ================================================
            # LARGEST FACE
            # ================================================

            face = max(

                faces_in_crop,

                key=lambda f:
                (
                    f.bbox[2]
                    -
                    f.bbox[0]
                )
                *
                (
                    f.bbox[3]
                    -
                    f.bbox[1]
                )
            )


            fx1, fy1, fx2, fy2 = (
                face.bbox
                .astype(int)
            )


            face_width = (
                fx2 - fx1
            )

            face_height = (
                fy2 - fy1
            )


            if (
                face_width
                <
                MIN_FACE_SIZE
                or
                face_height
                <
                MIN_FACE_SIZE
            ):

                continue


            # ================================================
            # MAP TO FULL FRAME
            # ================================================

            fx1 += cx1
            fy1 += cy1

            fx2 += cx1
            fy2 += cy1


            face_center_x = int(
                (fx1 + fx2) / 2
            )

            face_center_y = int(
                (fy1 + fy2) / 2
            )


            # ================================================
            # FACE MUST BE INSIDE ZONE
            # ================================================

            if not (

                zone_x1
                <=
                face_center_x
                <=
                zone_x2

                and

                zone_y1
                <=
                face_center_y
                <=
                zone_y2

            ):

                continue


            face_checks += 1


            # ================================================
            # EMBEDDING
            # ================================================

            embedding = normalize_embedding(
                face.normed_embedding
            )


            # ================================================
            # FIND BEST IDENTITY
            # ================================================

            best_name = "UNKNOWN"

            best_score = 0.0


            for name, known_embedding in (
                known_faces.items()
            ):

                score = cosine_similarity(

                    embedding,

                    known_embedding
                )


                if score > best_score:

                    best_score = score

                    best_name = name


            # ================================================
            # CONFIRM
            # ================================================

            if (
                best_score
                >=
                MATCH_THRESHOLD
            ):

                confirm_identity(

                    session,

                    best_name,

                    best_score,

                    embedding,

                    current_time
                )


            # ================================================
            # SAVE BEST EMBEDDING
            # ================================================

            if (
                session["status"]
                ==
                "CONFIRMED"
            ):

                if (
                    session[
                        "last_embedding"
                    ]
                    is not None
                ):

                    temporary_store.save_embedding(

                        track_id,

                        session[
                            "last_embedding"
                        ]
                    )


            # ================================================
            # ATTENDANCE
            # ================================================

            if (

                session["status"]
                ==
                "CONFIRMED"

                and

                not
                session[
                    "attendance_marked"
                ]

            ):

                mark_attendance(

                    session["name"],

                    current_time
                )


                session[
                    "attendance_marked"
                ] = True


            # ================================================
            # TEMPORARY PERSON DATA
            # ================================================

            temporary_store.update_session(

                track_id,

                session
            )


            # ================================================
            # SAVE PERSON SNAPSHOT
            # ================================================

            last_snapshot = (
                last_snapshot_time.get(
                    track_id,
                    0
                )
            )


            if (
                current_time
                -
                last_snapshot
                >=
                SNAPSHOT_INTERVAL
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

                    (
                        fx1,
                        fy1,
                        fx2,
                        fy2
                    ),

                    current_time
                )


                last_snapshot_time[
                    track_id
                ] = current_time


            # ================================================
            # DRAWING
            # ================================================

            if (
                session["status"]
                ==
                "CONFIRMED"
            ):

                box_color = (
                    0,
                    255,
                    0
                )

            elif (
                session["name"]
                !=
                "UNKNOWN"
            ):

                box_color = (
                    0,
                    255,
                    255
                )

            else:

                box_color = (
                    0,
                    0,
                    255
                )


            px1 = person["x1"]
            py1 = person["y1"]

            px2 = person["x2"]
            py2 = person["y2"]


            # ================================================
            # STATUS
            # ================================================

            if (
                session["status"]
                ==
                "CONFIRMED"
            ):

                status_text = (
                    "CONFIRMED"
                )

            elif (
                session["name"]
                !=
                "UNKNOWN"
            ):

                status_text = (

                    f"CHECK "
                    f"{session['confirmations']}/"
                    f"{REQUIRED_CONFIRMATIONS}"

                )

            else:

                status_text = (
                    "UNKNOWN"
                )


            # ================================================
            # PERSON BOX
            # ================================================

            cv2.rectangle(

                frame,

                (px1, py1),

                (px2, py2),

                box_color,

                2
            )


            label = (

                f"{session['name']} "
                f"| ID {track_id} "
                f"| "
                f"{session['best_score']:.2f} "
                f"| "
                f"{status_text}"

            )


            cv2.putText(

                frame,

                label,

                (
                    px1,
                    max(
                        py1 - 10,
                        25
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                box_color,

                2
            )


            # ================================================
            # FACE BOX
            # ================================================

            cv2.rectangle(

                frame,

                (fx1, fy1),

                (fx2, fy2),

                box_color,

                1
            )


        # ====================================================
        # TRAJECTORY
        # ====================================================

        if SHOW_TRAJECTORY:

            for (
                track_id,
                session
            ) in sessions.items():

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


            if (

                session["name"]
                !=
                "UNKNOWN"

                and

                session[
                    "last_embedding"
                ]
                is not None

            ):

                reid_cache[
                    session["name"]
                ] = {

                    "embedding":
                        session[
                            "last_embedding"
                        ],

                    "last_seen":
                        current_time
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


        # ====================================================
        # SESSION LIST
        # ====================================================

        display_y = 250


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