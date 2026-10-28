import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis


# ==========================================
# SETTINGS
# ==========================================

CAMERA_INDEX = 0

KNOWN_FACES_DIR = "app/face/known_faces"

MATCH_THRESHOLD = 0.45
MIN_FACE_SIZE = 60


# ==========================================
# LOAD INSIGHTFACE
# ==========================================

print("Loading InsightFace...")

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("InsightFace loaded.")


# ==========================================
# COSINE SIMILARITY
# ==========================================

def cosine_similarity(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return np.dot(a, b) / denominator


# ==========================================
# NORMALIZE
# ==========================================

def normalize_embedding(embedding):

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


# ==========================================
# LOAD KNOWN FACES
# ==========================================

known_faces = {}

print("\nScanning known faces...\n")

for filename in os.listdir(KNOWN_FACES_DIR):

    # Only images
    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    path = os.path.join(
        KNOWN_FACES_DIR,
        filename
    )

    print("Processing:", path)

    image = cv2.imread(path)

    if image is None:
        print("Could not read:", filename)
        continue

    # Detect face
    faces = face_app.get(image)

    if len(faces) == 0:
        print("No face found:", filename)
        continue

    # Select largest face
    face = max(
        faces,
        key=lambda f:
        (f.bbox[2] - f.bbox[0]) *
        (f.bbox[3] - f.bbox[1])
    )

    # Face size
    x1, y1, x2, y2 = face.bbox.astype(int)

    width = x2 - x1
    height = y2 - y1

    if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:

        print("Face too small:", filename)
        continue

    # Get normalized embedding
    embedding = face.normed_embedding

    embedding = normalize_embedding(
        embedding
    )

    # Filename becomes person's name
    person_name = os.path.splitext(filename)[0]

    known_faces[person_name] = embedding

    print(
        f"Loaded person: {person_name}"
    )


# ==========================================
# CHECK DATABASE
# ==========================================

print(
    f"\nKnown people loaded: {len(known_faces)}"
)

if len(known_faces) == 0:

    print("\nERROR: No known faces loaded!")

    print(
        "Make sure your image is inside:"
    )

    print(
        "app/face/known_faces/"
    )

    exit()


# ==========================================
# OPEN CAMERA
# ==========================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")
    exit()


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


print("\nCamera started.")
print("Press Q to quit.")


# ==========================================
# LIVE FACE RECOGNITION
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not read camera frame.")
        break


    # Detect faces
    faces = face_app.get(frame)


    for face in faces:

        # ==================================
        # FACE BOX
        # ==================================

        x1, y1, x2, y2 = face.bbox.astype(int)

        width = x2 - x1
        height = y2 - y1

        if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
            continue


        # ==================================
        # LIVE EMBEDDING
        # ==================================

        embedding = face.normed_embedding

        embedding = normalize_embedding(
            embedding
        )


        # ==================================
        # FIND BEST MATCH
        # ==================================

        best_name = "UNKNOWN"
        best_score = -1.0


        for name, known_embedding in known_faces.items():

            score = cosine_similarity(
                embedding,
                known_embedding
            )

            if score > best_score:

                best_score = score
                best_name = name


        # ==================================
        # RECOGNITION DECISION
        # ==================================

        if best_score >= MATCH_THRESHOLD:

            label = (
                f"{best_name} "
                f"{best_score:.2f}"
            )

            box_color = (0, 255, 0)

        else:

            label = (
                f"UNKNOWN "
                f"{best_score:.2f}"
            )

            box_color = (0, 0, 255)


        # ==================================
        # DRAW FACE BOX
        # ==================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )


        # ==================================
        # DRAW NAME
        # ==================================

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            box_color,
            2
        )


        # ==================================
        # DRAW SCORE
        # ==================================

        cv2.putText(
            frame,
            f"Match: {best_score:.2f}",
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2
        )


    # ==================================
    # DISPLAY
    # ==================================

    cv2.imshow(
        "AttendAI - Face Recognition",
        frame
    )


    # ==================================
    # QUIT
    # ==================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()