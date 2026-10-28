import cv2
from ultralytics import YOLO

# ==========================================
# SETTINGS
# ==========================================

CAMERA_INDEX = 0
MODEL_NAME = "yolo11n.pt"

# ==========================================
# LOAD YOLO
# ==========================================

print("Loading YOLO...")

model = YOLO(MODEL_NAME)

print("YOLO loaded.")
print("Starting person tracker...")

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera started.")
print("Press Q to quit.")

# ==========================================
# PERSON TRACKING
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # YOLO TRACKING
    results = model.track(
        frame,
        persist=True,
        classes=[0],       # Person only
        conf=0.5,
        verbose=False
    )

    # ======================================
    # PROCESS DETECTIONS
    # ======================================

    if results[0].boxes is not None:

        boxes = results[0].boxes

        for box in boxes:

            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Confidence
            confidence = float(box.conf[0])

            # Track ID
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            # ==================================
            # DRAW
            # ==================================

            label = f"PERSON ID {track_id} {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "AttendAI - Person Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()