# ============================================================
# ATTENDAI - BODY RE-ID + PERSON TRACKING TEST
# ============================================================

import os
import sys

# Add AttendAI project root to Python path
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
from ultralytics import YOLO

from app.person.body_reid import BodyReID


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 1
MODEL_NAME = "yolo11n.pt"

YOLO_CONFIDENCE = 0.35

BODY_MATCH_THRESHOLD = 0.80


# ============================================================
# INITIALIZE
# ============================================================

print("================================")
print("AttendAI Body Re-ID + Tracking")
print("================================")

print(f"Camera Index : {CAMERA_INDEX}")
print(f"YOLO Model   : {MODEL_NAME}")

print()
print("Loading YOLO...")

model = YOLO(MODEL_NAME)

print("YOLO loaded.")

print()
print("Initializing Body Re-ID...")

body_reid = BodyReID(
    match_threshold=BODY_MATCH_THRESHOLD
)

print("Body Re-ID ready.")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    print()
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


print()
print("Camera started.")
print()
print("Controls:")
print("S = Register visible person as TALHA")
print("Q = Quit")
print()


# ============================================================
# TRACKING LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not read frame.")

        break


    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=YOLO_CONFIDENCE,
        verbose=False
    )


    if (
        results
        and
        results[0].boxes is not None
    ):

        boxes = results[0].boxes


        for box in boxes:

            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = float(
                box.conf[0]
            )


            # ------------------------------------------------
            # TRACK ID
            # ------------------------------------------------

            if box.id is not None:

                track_id = int(
                    box.id[0]
                )

            else:

                track_id = -1


            # ------------------------------------------------
            # SAFETY
            # ------------------------------------------------

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                frame.shape[1],
                x2
            )

            y2 = min(
                frame.shape[0],
                y2
            )


            if x2 <= x1 or y2 <= y1:

                continue


            # ------------------------------------------------
            # BODY CROP
            # ------------------------------------------------

            body_crop = frame[
                y1:y2,
                x1:x2
            ]


            # ------------------------------------------------
            # BODY MATCH
            # ------------------------------------------------

            body_result = body_reid.match(
                body_crop
            )


            body_identity = body_result[
                "identity"
            ]

            body_score = body_result[
                "score"
            ]


            # ------------------------------------------------
            # DRAW
            # ------------------------------------------------

            if body_identity != "UNKNOWN":

                label = (
                    f"ID {track_id} | "
                    f"{body_identity} | "
                    f"Body {body_score:.2f}"
                )

            else:

                label = (
                    f"ID {track_id} | "
                    f"Body UNKNOWN | "
                    f"{body_score:.2f}"
                )


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
                0.65,
                (0, 255, 0),
                2
            )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "AttendAI - Body Re-ID Tracking",
        frame
    )


    key = (
        cv2.waitKey(1)
        &
        0xFF
    )


    # --------------------------------------------------------
    # REGISTER BODY
    # --------------------------------------------------------

    if key == ord("s"):

        print()
        print("S pressed.")

        # Find the largest visible person
        # and register that person's body.

        if (
            results
            and
            results[0].boxes is not None
            and
            len(results[0].boxes) > 0
        ):

            largest_box = None
            largest_area = 0


            for box in results[0].boxes:

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )


                area = (
                    x2 - x1
                ) * (
                    y2 - y1
                )


                if area > largest_area:

                    largest_area = area
                    largest_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )


            if largest_box is not None:

                x1, y1, x2, y2 = (
                    largest_box
                )


                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)


                body_crop = frame[
                    y1:y2,
                    x1:x2
                ]


                success = body_reid.register_identity(
                    "talha",
                    body_crop
                )


                if success:

                    print(
                        "[BODY RE-ID] "
                        "Successfully registered TALHA."
                    )

                else:

                    print(
                        "[BODY RE-ID] "
                        "Registration failed."
                    )

        else:

            print(
                "[BODY RE-ID] "
                "No person detected."
            )


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("AttendAI Body Re-ID Tracking stopped.")