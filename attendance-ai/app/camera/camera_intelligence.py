import cv2
import time
from ultralytics import YOLO


# ==========================================
# CONFIGURATION
# ==========================================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

MODEL_NAME = "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.45


# ==========================================
# LOAD YOLO
# ==========================================

print("Loading YOLO...")

model = YOLO(MODEL_NAME)

print("YOLO loaded successfully.")


# ==========================================
# OPEN CAMERA
# ==========================================

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()


# ==========================================
# CAMERA INFORMATION
# ==========================================

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

camera_fps = cap.get(cv2.CAP_PROP_FPS)

print("--------------------------------")
print("CAMERA INFORMATION")
print("--------------------------------")
print(f"Resolution : {actual_width} x {actual_height}")
print(f"Camera FPS : {camera_fps}")
print("--------------------------------")


# ==========================================
# FPS VARIABLES
# ==========================================

previous_time = time.time()

fps = 0


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read frame.")
        break


    # --------------------------------------
    # YOLO PERSON DETECTION
    # --------------------------------------

    results = model(
    frame,
    verbose=False,
    conf=CONFIDENCE_THRESHOLD,
    device=0
)

    person_count = 0


    for result in results:

        boxes = result.boxes


        for box in boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])


            # COCO class 0 = person
            if class_id != 0:
                continue


            person_count += 1


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            label = f"PERSON {confidence * 100:.1f}%"


            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    # --------------------------------------
    # FPS CALCULATION
    # --------------------------------------

    current_time = time.time()

    fps = 1 / (current_time - previous_time)

    previous_time = current_time


    # --------------------------------------
    # INFORMATION PANEL
    # --------------------------------------

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Resolution: {actual_width}x{actual_height}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"People detected: {person_count}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Human Detection: ACTIVE",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "Camera: CENTER",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # --------------------------------------
    # DISPLAY
    # --------------------------------------

    cv2.imshow(
        "AttendAI - Camera Intelligence",
        frame
    )


    # ESC to exit
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("Camera test completed.")