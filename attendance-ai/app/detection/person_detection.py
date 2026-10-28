import cv2
from ultralytics import YOLO


def main():

    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:

        success, frame = cap.read()

        if not success:
            print("Failed to read frame")
            break

        # Persistent multi-object tracking
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=0.5,
            tracker="bytetrack.yaml",
            verbose=False
        )

        annotated_frame = results[0].plot()

        cv2.imshow(
            "AttendAI - Person Tracking",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()