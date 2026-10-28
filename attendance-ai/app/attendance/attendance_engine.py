import os
import json
import threading
from datetime import datetime


class AttendanceEngine:
    """
    Attendance decision layer for AttendAI.

    Responsibilities:
    - Keep one attendance record per identity per day
    - Record first entry
    - Update last seen
    - Preserve confidence
    - Preserve track information
    - Save temporary daily attendance data
    """

    def __init__(self, storage_root="temporary_day_data"):
        self.storage_root = storage_root
        self.lock = threading.Lock()

        self.today = datetime.now().strftime("%Y-%m-%d")

        self.day_dir = os.path.join(
            self.storage_root,
            self.today
        )

        self.attendance_file = os.path.join(
            self.day_dir,
            "attendance.json"
        )

        os.makedirs(self.day_dir, exist_ok=True)

        self.records = {}

        self._load_existing()

        print("======================================")
        print("ATTENDANCE ENGINE")
        print("======================================")
        print("Date:", self.today)
        print("Storage:", self.day_dir)
        print("Existing records:", len(self.records))
        print("======================================")

    # =========================================================
    # LOAD EXISTING DATA
    # =========================================================

    def _load_existing(self):

        if not os.path.exists(self.attendance_file):
            return

        try:

            with open(
                self.attendance_file,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            self.records = data.get("records", {})

        except Exception as e:

            print(
                "[ATTENDANCE] Could not load existing data:",
                e
            )

            self.records = {}

    # =========================================================
    # SAVE DATA
    # =========================================================

    def _save(self):

        data = {
            "date": self.today,
            "saved_at": datetime.now().isoformat(),
            "total_present": len(self.records),
            "records": self.records
        }

        temporary_file = self.attendance_file + ".tmp"

        with open(
            temporary_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False
            )

        os.replace(
            temporary_file,
            self.attendance_file
        )

    # =========================================================
    # REGISTER PERSON
    # =========================================================

    def register_presence(
        self,
        identity,
        track_id,
        confidence,
        zone_entered=False
    ):

        if not identity:
            return None

        if identity == "UNKNOWN":
            return None

        now = datetime.now()

        timestamp = now.isoformat()

        with self.lock:

            # -------------------------------------------------
            # FIRST APPEARANCE
            # -------------------------------------------------

            if identity not in self.records:

                self.records[identity] = {

                    "identity": identity,

                    "track_ids": [
                        int(track_id)
                    ],

                    "status": "PRESENT",

                    "first_seen": timestamp,

                    "last_seen": timestamp,

                    "best_confidence": float(confidence),

                    "recognition_count": 1,

                    "zone_entered": bool(zone_entered),

                    "attendance_marked": True
                }

                self._save()

                print(
                    f"[ATTENDANCE] "
                    f"{identity} PRESENT "
                    f"| confidence={confidence:.3f}"
                )

                return self.records[identity]

            # -------------------------------------------------
            # EXISTING PERSON
            # -------------------------------------------------

            record = self.records[identity]

            record["last_seen"] = timestamp

            record["recognition_count"] += 1

            # Keep the best confidence
            if confidence > record["best_confidence"]:

                record["best_confidence"] = float(
                    confidence
                )

            # Keep different tracker IDs
            if track_id not in record["track_ids"]:

                record["track_ids"].append(
                    int(track_id)
                )

            # Once entered, always preserve it
            if zone_entered:

                record["zone_entered"] = True

            self._save()

            return record

    # =========================================================
    # GET PERSON
    # =========================================================

    def get_person(self, identity):

        with self.lock:

            return self.records.get(identity)

    # =========================================================
    # GET ALL PRESENT PEOPLE
    # =========================================================

    def get_present_people(self):

        with self.lock:

            return list(
                self.records.values()
            )

    # =========================================================
    # CHECK ATTENDANCE
    # =========================================================

    def is_present(self, identity):

        with self.lock:

            return identity in self.records

    # =========================================================
    # FINALIZE
    # =========================================================

    def finalize(self):

        with self.lock:

            self._save()

        print()
        print("======================================")
        print("ATTENDANCE SUMMARY")
        print("======================================")
        print("Date:", self.today)
        print(
            "Total Present:",
            len(self.records)
        )
        print(
            "Storage:",
            self.attendance_file
        )
        print("======================================")

        return self.attendance_file