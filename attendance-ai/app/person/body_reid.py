# ============================================================
# ATTENDAI - BODY RE-ID MODULE
# ============================================================

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BODY_MATCH_THRESHOLD = 0.80

FEATURE_WIDTH = 64
FEATURE_HEIGHT = 128


# ============================================================
# BODY FEATURE EXTRACTION
# ============================================================

def extract_body_feature(crop):
    """
    Extract a lightweight body appearance feature.

    Current implementation:
    HSV spatial color histogram.

    This is a prototype Body Re-ID representation.
    """

    if crop is None or crop.size == 0:
        return None

    crop = cv2.resize(
        crop,
        (FEATURE_WIDTH, FEATURE_HEIGHT)
    )

    # BGR -> HSV
    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV
    )

    # Spatial color histogram
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 16],
        [0, 180, 0, 256]
    )

    hist = cv2.normalize(
        hist,
        hist
    ).flatten()

    return hist.astype(
        np.float32
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    if a is None or b is None:
        return 0.0

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b)
        /
        denominator
    )


# ============================================================
# BODY RE-ID ENGINE
# ============================================================

class BodyReID:

    def __init__(
        self,
        match_threshold=BODY_MATCH_THRESHOLD
    ):

        self.match_threshold = (
            match_threshold
        )

        # Identity -> body feature
        self.reference_features = {}

        print(
            "[BODY RE-ID] Engine initialized"
        )

    # --------------------------------------------------------
    # REGISTER BODY
    # --------------------------------------------------------

    def register_identity(
        self,
        identity,
        body_crop
    ):
        """
        Store a body feature for an identity.
        """

        if (
            identity is None
            or
            identity == "UNKNOWN"
        ):
            return False

        feature = extract_body_feature(
            body_crop
        )

        if feature is None:
            return False

        self.reference_features[
            identity
        ] = feature

        print(
            f"[BODY RE-ID] "
            f"Registered: {identity}"
        )

        return True

    # --------------------------------------------------------
    # MATCH BODY
    # --------------------------------------------------------

    def match(
        self,
        body_crop
    ):
        """
        Compare a detected body against
        all registered identities.

        Returns:

        {
            "identity": "...",
            "score": 0.xx
        }
        """

        if not self.reference_features:
            return {
                "identity": "UNKNOWN",
                "score": 0.0
            }

        feature = extract_body_feature(
            body_crop
        )

        if feature is None:
            return {
                "identity": "UNKNOWN",
                "score": 0.0
            }

        best_identity = "UNKNOWN"
        best_score = 0.0

        for (
            identity,
            reference
        ) in self.reference_features.items():

            score = cosine_similarity(
                reference,
                feature
            )

            if score > best_score:

                best_score = score

                best_identity = identity

        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        if (
            best_score
            <
            self.match_threshold
        ):

            best_identity = "UNKNOWN"

        return {
            "identity": best_identity,
            "score": round(
                float(best_score),
                4
            )
        }

    # --------------------------------------------------------
    # REGISTER FROM FACE
    # --------------------------------------------------------

    def register_from_face(
        self,
        identity,
        body_crop
    ):
        """
        Convenience method used when Face
        Recognition has already identified
        the person.
        """

        return self.register_identity(
            identity,
            body_crop
        )