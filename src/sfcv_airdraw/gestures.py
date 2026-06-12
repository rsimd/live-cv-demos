"""Gesture classification helpers for MediaPipe Hands landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


PoseName = Literal["index_up", "open_palm", "fist", "other"]


@dataclass(frozen=True, slots=True)
class Landmark:
    """Normalized hand landmark.

    Attributes:
        x: Horizontal coordinate in image-normalized coordinates.
        y: Vertical coordinate in image-normalized coordinates.
        z: Relative depth coordinate.
    """

    x: float
    y: float
    z: float = 0.0


FINGER_TIP_AND_PIP = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}


def extended_fingers(
    landmarks: Sequence[Landmark],
    *,
    margin: float = 0.025,
) -> dict[str, bool]:
    """Return extension state for the four non-thumb fingers.

    Args:
        landmarks: Sequence of 21 MediaPipe hand landmarks.
        margin: Minimum normalized y distance between tip and PIP joint.

    Returns:
        Mapping from finger name to whether the finger is extended upward.

    Raises:
        ValueError: If fewer than 21 landmarks are provided.
    """

    if len(landmarks) < 21:
        raise ValueError("MediaPipe Hands must provide 21 landmarks.")

    states: dict[str, bool] = {}
    for name, (tip_index, pip_index) in FINGER_TIP_AND_PIP.items():
        tip = landmarks[tip_index]
        pip = landmarks[pip_index]
        states[name] = tip.y < pip.y - margin
    return states


def is_index_up_pose(landmarks: Sequence[Landmark]) -> bool:
    """Return whether the hand is making an index-only start gesture."""

    states = extended_fingers(landmarks)
    return (
        states["index"]
        and not states["middle"]
        and not states["ring"]
        and not states["pinky"]
    )


def is_open_palm_pose(landmarks: Sequence[Landmark]) -> bool:
    """Return whether the hand is making an open-palm stop gesture."""

    states = extended_fingers(landmarks)
    return all(states.values())


def is_fist_pose(landmarks: Sequence[Landmark]) -> bool:
    """Return whether the hand is making a closed-fist stop gesture."""

    states = extended_fingers(landmarks)
    return not any(states.values())


def classify_pose(landmarks: Sequence[Landmark]) -> PoseName:
    """Classify the demo gestures plus the fallback state."""

    if is_open_palm_pose(landmarks):
        return "open_palm"
    if is_index_up_pose(landmarks):
        return "index_up"
    if is_fist_pose(landmarks):
        return "fist"
    return "other"
