# ============================================================
# ATTENDAI - ADAPTIVE MULTIMODAL FUSION ENGINE
# ============================================================

from collections import defaultdict, deque


# ============================================================
# CONFIGURATION
# ============================================================

FUSION_CONFIRM_THRESHOLD = 0.65
FUSION_UNKNOWN_THRESHOLD = 0.45

# Initial weights.
# These can later be learned/calibrated from validation data.
FACE_WEIGHT = 0.50
BODY_WEIGHT = 0.25
GAIT_WEIGHT = 0.25


# ============================================================
# TEMPORAL EVIDENCE
# ============================================================

TEMPORAL_HISTORY_SIZE = 5

identity_history = defaultdict(
    lambda: deque(
        maxlen=TEMPORAL_HISTORY_SIZE
    )
)


# ============================================================
# HELPER
# ============================================================

def valid_score(score):
    return (
        score is not None
        and
        isinstance(score, (int, float))
        and
        score >= 0.0
    )


# ============================================================
# FUSION ENGINE
# ============================================================

def fuse_identity(
    track_id,

    face_identity=None,
    face_score=None,

    body_identity=None,
    body_score=None,

    gait_identity=None,
    gait_score=None
):

    evidence = []

    # --------------------------------------------------------
    # FACE
    # --------------------------------------------------------

    if (
        face_identity
        and
        face_identity != "UNKNOWN"
        and
        valid_score(face_score)
    ):

        evidence.append(
            {
                "identity": face_identity,
                "score": float(face_score),
                "weight": FACE_WEIGHT,
                "modality": "face"
            }
        )


    # --------------------------------------------------------
    # BODY
    # --------------------------------------------------------

    if (
        body_identity
        and
        body_identity != "UNKNOWN"
        and
        valid_score(body_score)
    ):

        evidence.append(
            {
                "identity": body_identity,
                "score": float(body_score),
                "weight": BODY_WEIGHT,
                "modality": "body"
            }
        )


    # --------------------------------------------------------
    # GAIT
    # --------------------------------------------------------

    if (
        gait_identity
        and
        gait_identity != "UNKNOWN"
        and
        valid_score(gait_score)
    ):

        evidence.append(
            {
                "identity": gait_identity,
                "score": float(gait_score),
                "weight": GAIT_WEIGHT,
                "modality": "gait"
            }
        )


    # --------------------------------------------------------
    # NO EVIDENCE
    # --------------------------------------------------------

    if not evidence:

        return {
            "track_id": track_id,
            "identity": "UNKNOWN",
            "confidence": 0.0,
            "status": "UNKNOWN",
            "face_score": 0.0,
            "body_score": 0.0,
            "gait_score": 0.0,
            "modalities_used": 0
        }


    # --------------------------------------------------------
    # WEIGHTED IDENTITY SCORES
    # --------------------------------------------------------

    identity_scores = defaultdict(float)
    identity_weights = defaultdict(float)

    for item in evidence:

        identity = item["identity"]

        identity_scores[identity] += (
            item["score"]
            *
            item["weight"]
        )

        identity_weights[identity] += (
            item["weight"]
        )


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    normalized_scores = {}

    for identity in identity_scores:

        weight = identity_weights[identity]

        if weight > 0:

            normalized_scores[identity] = (
                identity_scores[identity]
                /
                weight
            )


    # --------------------------------------------------------
    # BEST IDENTITY
    # --------------------------------------------------------

    best_identity = max(
        normalized_scores,
        key=normalized_scores.get
    )

    best_confidence = normalized_scores[
        best_identity
    ]


    # --------------------------------------------------------
    # TEMPORAL AGGREGATION
    # --------------------------------------------------------

    identity_history[track_id].append(
        (
            best_identity,
            best_confidence
        )
    )

    history = identity_history[track_id]

    same_identity = [
        score
        for identity, score in history
        if identity == best_identity
    ]

    if same_identity:

        temporal_score = (
            sum(same_identity)
            /
            len(same_identity)
        )

        # Small temporal stabilization.
        best_confidence = (
            0.70 * best_confidence
            +
            0.30 * temporal_score
        )


    # --------------------------------------------------------
    # MODALITY SCORES
    # --------------------------------------------------------

    face_score_output = 0.0
    body_score_output = 0.0
    gait_score_output = 0.0

    for item in evidence:

        if item["identity"] != best_identity:
            continue

        if item["modality"] == "face":
            face_score_output = item["score"]

        elif item["modality"] == "body":
            body_score_output = item["score"]

        elif item["modality"] == "gait":
            gait_score_output = item["score"]


    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    if best_confidence >= FUSION_CONFIRM_THRESHOLD:

        status = "CONFIRMED"

    elif best_confidence >= FUSION_UNKNOWN_THRESHOLD:

        status = "REVIEW"

    else:

        status = "UNKNOWN"


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {

        "track_id": track_id,

        "identity": best_identity,

        "confidence": round(
            float(best_confidence),
            4
        ),

        "status": status,

        "face_score": round(
            float(face_score_output),
            4
        ),

        "body_score": round(
            float(body_score_output),
            4
        ),

        "gait_score": round(
            float(gait_score_output),
            4
        ),

        "modalities_used": len(evidence)
    }


# ============================================================
# TEST
# ============================================================


if __name__ == "__main__":

    print()
    print("================================")
    print("AttendAI Fusion Engine Test")
    print("================================")

    result = fuse_identity(

        track_id=6,

        face_identity="talha",
        face_score=0.79,

        body_identity="talha",
        body_score=0.84,

        gait_identity="talha",
        gait_score=0.76
    )

    print(result)
