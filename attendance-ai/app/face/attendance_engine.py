from datetime import datetime


class AttendanceEngine:

    def __init__(self, confidence_threshold=0.70):
        self.confidence_threshold = confidence_threshold

        # Prevent duplicate attendance
        self.marked_students = set()

        print("================================")
        print("AttendAI Attendance Engine")
        print("================================")
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print("Attendance Engine Ready")
        print()


    def process_fusion_result(self, fusion_result):
        """
        Receives the output of the Fusion Engine.

        Expected format:

        {
            "identity": "talha",
            "confidence": 0.799,
            "status": "CONFIRMED",
            "face_score": 0.79,
            "body_score": 0.84,
            "gait_score": 0.76,
            "modalities_used": 3
        }
        """

        identity = fusion_result.get("identity")
        confidence = fusion_result.get("confidence", 0.0)
        status = fusion_result.get("status", "UNKNOWN")

        # -----------------------------------------
        # 1. Unknown person
        # -----------------------------------------

        if identity is None or identity == "unknown":
            return {
                "attendance": False,
                "reason": "UNKNOWN_PERSON"
            }

        # -----------------------------------------
        # 2. Fusion confidence check
        # -----------------------------------------

        if confidence < self.confidence_threshold:
            return {
                "attendance": False,
                "reason": "LOW_CONFIDENCE",
                "identity": identity,
                "confidence": confidence
            }

        # -----------------------------------------
        # 3. Fusion status check
        # -----------------------------------------

        if status != "CONFIRMED":
            return {
                "attendance": False,
                "reason": "NOT_CONFIRMED",
                "identity": identity,
                "confidence": confidence
            }

        # -----------------------------------------
        # 4. Duplicate attendance prevention
        # -----------------------------------------

        if identity in self.marked_students:
            return {
                "attendance": False,
                "reason": "ALREADY_MARKED",
                "identity": identity,
                "confidence": confidence
            }

        # -----------------------------------------
        # 5. Mark attendance
        # -----------------------------------------

        self.marked_students.add(identity)

        timestamp = datetime.now().isoformat(timespec="seconds")

        attendance_record = {
            "attendance": True,
            "identity": identity,
            "status": "PRESENT",
            "confidence": confidence,
            "timestamp": timestamp,
            "face_score": fusion_result.get("face_score"),
            "body_score": fusion_result.get("body_score"),
            "gait_score": fusion_result.get("gait_score"),
            "modalities_used": fusion_result.get("modalities_used")
        }

        print()
        print("================================")
        print("ATTENDANCE MARKED")
        print("================================")
        print(f"Student     : {identity}")
        print(f"Status      : PRESENT")
        print(f"Confidence  : {confidence:.3f}")
        print(f"Timestamp   : {timestamp}")
        print("================================")
        print()

        return attendance_record


# ============================================================
# TEST - FUSION → ATTENDANCE
# ============================================================

if __name__ == "__main__":

    from fusion_engine import FusionEngine

    print()
    print("================================")
    print("AttendAI Fusion → Attendance Test")
    print("================================")

    # --------------------------------------------------------
    # STEP 1: Create Fusion Engine
    # --------------------------------------------------------

    fusion = FusionEngine()

    # --------------------------------------------------------
    # STEP 2: Generate Fusion Result
    # --------------------------------------------------------

    fusion_result = fusion.fuse(

        face_identity="talha",
        face_score=0.79,

        body_identity="talha",
        body_score=0.84,

        gait_identity="talha",
        gait_score=0.76
    )

    # Convert FusionResult object to dictionary
    fusion_data = fusion_result.to_dict()

    print()
    print("FUSION RESULT")
    print("--------------------------------")
    print(fusion_data)

    # --------------------------------------------------------
    # STEP 3: Send Fusion Result to Attendance Engine
    # --------------------------------------------------------

    engine = AttendanceEngine(
        confidence_threshold=0.70
    )

    attendance_result = engine.process_fusion_result(
        fusion_data
    )

    print()
    print("ATTENDANCE RESULT")
    print("--------------------------------")
    print(attendance_result)

    # --------------------------------------------------------
    # STEP 4: Test Duplicate Prevention
    # --------------------------------------------------------

    print()
    print("TESTING DUPLICATE RECOGNITION")
    print("--------------------------------")

    duplicate_result = engine.process_fusion_result(
        fusion_data
    )

    print(duplicate_result)