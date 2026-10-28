import cv2
import time

print("=" * 50)
print("       ATTENDAI CAMERA TEST")
print("=" * 50)

# We know:
# Camera 0 = another camera
# Camera 1 = Lenovo Virtual Camera / Camo
CAMERA_INDEX = 2

print(f"\nTrying Camera {CAMERA_INDEX}...")
print("Backend: DirectShow")
print("Please wait...\n")

# Use Windows DirectShow backend
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

print("Opened:", cap.isOpened())

if not cap.isOpened():
    print("\n❌ ERROR: Camera could not be opened.")
    print("Check that Camo Studio is running and the iPhone is connected.")
    exit()

print("✅ Camera device opened.")
print("Waiting for video frame...")

# Give the virtual camera a little time to initialize
time.sleep(2)

# Try to read a frame
ret, frame = cap.read()

print("Frame received:", ret)

if not ret or frame is None:
    print("\n❌ CAMERA OPENED BUT NO FRAME RECEIVED")
    print("The Lenovo Virtual Camera is detected,")
    print("but OpenCV is not receiving the Camo video stream.")

    cap.release()
    exit()

print("✅ FRAME RECEIVED!")
print("Frame size:", frame.shape)

print("\nOpening camera window...")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        print("⚠️ Frame lost!")
        break

    cv2.imshow("Camo Camera Test", frame)

    # Press Q to exit
    key = cv2.waitKey(2) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n✅ Camera test finished.")