import cv2
import time

CAMERA_INDEX = 2

print("Starting iVCam test...")
print("Opening camera index:", CAMERA_INDEX)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ FAILED: iVCam could not be opened")
    print("Try camera index 0 or 1.")
    input("Press Enter to exit...")
    exit()

print("✅ iVCam OPENED successfully!")

# Request 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Frame could not be read")
        break

    cv2.putText(
        frame,
        "AttendAI - iVCam",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("AttendAI - iVCam Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Camera test finished.")