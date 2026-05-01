from __future__ import annotations
import numpy as np
from names import Face, KnownFaces

_L_SHOULDER, _R_SHOULDER = 5, 6
_L_WRIST,    _R_WRIST    = 9, 10
_CONF_THRESH = 0.4


def _salience_score(keypoints: np.ndarray, frame_height: int) -> float:
    """
    Compute hand-raise score for one person.
    keypoints: shape (17, 3) — (x, y, confidence) per keypoint.
    Returns value in [0, 1]; higher means wrist further above shoulder.
    """
    score = 0.0
    for shoulder_idx, wrist_idx in ((_L_SHOULDER, _L_WRIST), (_R_SHOULDER, _R_WRIST)):
        sy, sc = keypoints[shoulder_idx, 1], keypoints[shoulder_idx, 2]
        wy, wc = keypoints[wrist_idx,    1], keypoints[wrist_idx,    2]
        if sc < _CONF_THRESH or wc < _CONF_THRESH:
            continue
        # y increases downward, so wrist above shoulder means sy > wy
        lift = (sy - wy) / frame_height
        score = max(score, lift)
    return max(0.0, score)


class GestureDetector:
    RAISED_HAND_THRESHOLD = 0.10  # wrist must be >= 10% of frame height above shoulder

    def update_scores(
        self,
        pose_results,
        known_faces: KnownFaces,
        frame_height: int,
    ) -> None:
        """
        Recompute gesture_score on every Face in known_faces from the latest
        pose results. Faces with no matching detection get their score reset to 0.
        """
        for face in known_faces.faces:
            face.gesture_score = 0.0

        if not pose_results or pose_results[0].keypoints is None:
            return

        kpts_data = pose_results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)

        for idx, kpts in enumerate(kpts_data, start=1):
            face = known_faces.get_by_id(idx)
            if face is None:
                continue
            face.gesture_score = _salience_score(kpts, frame_height)

    def most_salient(self, known_faces: KnownFaces) -> Face | None:
        """
        Return the Face with the highest gesture_score above threshold,
        or None if no one clears it.
        """
        candidates = [
            f for f in known_faces.faces
            if f.gesture_score >= self.RAISED_HAND_THRESHOLD
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda f: f.gesture_score)