import json
import os
from datetime import datetime
from threading import Lock


class TemporarySessionStore:

    def __init__(self, base_dir="data/temporary"):
        self.base_dir = base_dir
        self.lock = Lock()

        os.makedirs(self.base_dir, exist_ok=True)

        self.sessions = {}

    # ---------------------------------------------------------
    # Create / get session
    # ---------------------------------------------------------

    def get_or_create(self, track_id):

        track_id = int(track_id)

        with self.lock:

            if track_id not in self.sessions:

                now = datetime.now().isoformat()

                self.sessions[track_id] = {

                    "track_id": track_id,

                    "first_seen": now,
                    "last_seen": now,

                    "frame_count": 0,

                    "identity": None,
                    "identity_id": None,
                    "confidence": 0.0,

                    "face_confirmed": False,

                    "positions": [],

                    "appearance": {

                        "clothing": None,
                        "bag": None,
                        "glasses": None,
                        "watch": None,
                        "other": []
                    },

                    "recognition_history": []
                }

            return self.sessions[track_id]

    # ---------------------------------------------------------
    # Update tracking information
    # ---------------------------------------------------------

    def update_track(
        self,
        track_id,
        center=None,
        identity=None,
        identity_id=None,
        confidence=None,
        face_confirmed=None
    ):

        session = self.get_or_create(track_id)

        now = datetime.now().isoformat()

        with self.lock:

            session["last_seen"] = now

            session["frame_count"] += 1

            if identity is not None:
                session["identity"] = identity

            if identity_id is not None:
                session["identity_id"] = identity_id

            if confidence is not None:
                session["confidence"] = float(confidence)

            if face_confirmed is not None:
                session["face_confirmed"] = bool(face_confirmed)

            # Store movement position
            if center is not None:

                x, y = center

                session["positions"].append({
                    "x": int(x),
                    "y": int(y),
                    "time": now
                })

                # Keep only recent positions
                if len(session["positions"]) > 30:
                    session["positions"] = session["positions"][-30:]

            return session

    # ---------------------------------------------------------
    # Add recognition observation
    # ---------------------------------------------------------

    def add_recognition(
        self,
        track_id,
        identity,
        confidence
    ):

        session = self.get_or_create(track_id)

        now = datetime.now().isoformat()

        with self.lock:

            session["recognition_history"].append({

                "identity": identity,

                "confidence": float(confidence),

                "time": now
            })

            # Keep temporary history small
            if len(session["recognition_history"]) > 20:

                session["recognition_history"] = \
                    session["recognition_history"][-20:]

            # Update current identity
            if identity is not None:

                session["identity"] = identity

                session["confidence"] = float(confidence)

                session["face_confirmed"] = True

    # ---------------------------------------------------------
    # Appearance information
    # ---------------------------------------------------------

    def update_appearance(
        self,
        track_id,
        clothing=None,
        bag=None,
        glasses=None,
        watch=None,
        other=None
    ):

        session = self.get_or_create(track_id)

        with self.lock:

            appearance = session["appearance"]

            if clothing is not None:
                appearance["clothing"] = clothing

            if bag is not None:
                appearance["bag"] = bag

            if glasses is not None:
                appearance["glasses"] = glasses

            if watch is not None:
                appearance["watch"] = watch

            if other is not None:
                appearance["other"] = other

    # ---------------------------------------------------------
    # Remove disappeared track
    # ---------------------------------------------------------

    def remove_track(self, track_id):

        with self.lock:

            return self.sessions.pop(int(track_id), None)

    # ---------------------------------------------------------
    # Get active sessions
    # ---------------------------------------------------------

    def get_active_sessions(self):

        with self.lock:

            return dict(self.sessions)

    # ---------------------------------------------------------
    # Save today's temporary data
    # ---------------------------------------------------------

    def save_daily_data(self):

        today = datetime.now().strftime("%Y-%m-%d")

        filename = os.path.join(
            self.base_dir,
            f"{today}.json"
        )

        with self.lock:

            data = {

                "date": today,

                "saved_at": datetime.now().isoformat(),

                "session_count": len(self.sessions),

                "sessions": self.sessions
            }

            with open(
                filename,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    data,
                    f,
                    indent=4,
                    ensure_ascii=False
                )

        return filename

    # ---------------------------------------------------------
    # Clear memory
    # ---------------------------------------------------------

    def clear(self):

        with self.lock:
            self.sessions.clear()