import cv2

cap = cv2.VideoCapture(1)

print("Camera 1 opened:", cap.isOpened())
print("Press Q to close")

if not cap.isOpened():
    print("ERROR: Camera 1 could not be opened")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    cv2.imshow("LAPTOP CAMERA", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()