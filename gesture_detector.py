from __future__ import annotations
import numpy as np
from names import Face, KnownFaces

_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW,    _R_ELBOW    = 7, 8
_L_WRIST,    _R_WRIST    = 9, 10

# Wrist must be clearly detected — high confidence is the proxy for "close to camera"
_WRIST_CONF_MIN = 0.3
# Elbow must be BELOW this — not visible means arm is extended toward the camera.
# yolov8n-pose predicts occluded joints via heatmap interpolation and assigns them
# moderate confidence (~0.4-0.5) even when truly hidden, so 0.6 is the practical cutoff.
_ELBOW_CONF_MAX = 0.6
# Above this confidence the shoulder is considered "visible"
_SHOULDER_CONF_MIN = 0.4
# If shoulder is visible, shoulder_y and wrist_y must be within this fraction of frame height
_SHOULDER_WRIST_MAX_Y_FRAC = 0.10


def _wave_score(keypoints: np.ndarray, frame_height: int) -> float:
    """
    Return wrist confidence if the arm satisfies a "wave toward camera" gesture,
    otherwise 0.0. Evaluated per arm; returns the max across both arms.

    Criteria for each arm:
      1. Wrist is clearly visible (wrist confidence >= _WRIST_CONF_MIN) — the hand
         is large/close to the camera.
      2. Elbow is NOT visible (elbow confidence < _ELBOW_CONF_MAX) — arm is
         extended toward the camera so the elbow is hidden behind the hand.
      3. Shoulder is not visible (confidence < _SHOULDER_CONF_MIN) OR, if the
         shoulder is visible, |shoulder_y - wrist_y| / frame_height < 0.10 —
         the arm is foreshortened, so both joints appear at nearly the same height.
    """
    score = 0.0
    for shoulder_idx, elbow_idx, wrist_idx in (
        (_L_SHOULDER, _L_ELBOW, _L_WRIST),
        (_R_SHOULDER, _R_ELBOW, _R_WRIST),
    ):
        wc = keypoints[wrist_idx,    2]
        ec = keypoints[elbow_idx,    2]
        sc = keypoints[shoulder_idx, 2]
        wy = keypoints[wrist_idx,    1]
        sy = keypoints[shoulder_idx, 1]

        # criterion 1: wrist clearly visible
        if wc < _WRIST_CONF_MIN:
            continue

        # criterion 2: elbow NOT visible
        if ec >= _ELBOW_CONF_MAX:
            continue

        # criterion 3: shoulder not visible, or shoulder and wrist at similar height
        if sc >= _SHOULDER_CONF_MIN:
            if abs(sy - wy) / frame_height > _SHOULDER_WRIST_MAX_Y_FRAC:
                continue

        score = max(score, wc)

    return score


class GestureDetector:
    WAVE_THRESHOLD = 0.3  # minimum wrist confidence to count as a wave gesture

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
            face.gesture_score = _wave_score(kpts, frame_height)

    def most_salient(self, known_faces: KnownFaces) -> Face | None:
        """
        Return the Face with the highest gesture_score above the wave threshold,
        or None if no one clears it.
        """
        candidates = [
            f for f in known_faces.faces
            if f.gesture_score >= self.WAVE_THRESHOLD
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda f: f.gesture_score)
