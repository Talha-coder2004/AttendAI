import requests
from datetime import datetime


BACKEND_URL = "http://localhost:8080"

IDENTITY_EVENT_URL = (
    f"{BACKEND_URL}/api/events/identity"
)


def send_identity_event(
    track_id,
    candidate_id,
    camera_id="CAM01",
    confidence=0.0,
    direction="IN",
    event_type="IDENTITY_CONFIRMED"
):

    payload = {

        "trackId": str(track_id),

        "candidateId": str(candidate_id),

        "cameraId": str(camera_id),

        "eventType": event_type,

        "confidence": float(confidence),

        "direction": direction,

        "timestamp": datetime.now().isoformat()
    }

    try:

        response = requests.post(
            IDENTITY_EVENT_URL,
            json=payload,
            timeout=2
        )

        if response.status_code in (200, 201):

            print(
                "[BACKEND] Identity event sent | "
                f"Track={track_id} | "
                f"Identity={candidate_id} | "
                f"Confidence={confidence:.3f}"
            )

            return True

        print(
            "[BACKEND ERROR] "
            f"HTTP {response.status_code} | "
            f"{response.text}"
        )

        return False

    except requests.RequestException as e:

        print(
            "[BACKEND ERROR] "
            f"Could not connect to Spring Boot: {e}"
        )

        return False