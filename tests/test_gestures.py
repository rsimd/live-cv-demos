from __future__ import annotations

import unittest

from sfcv_airdraw.gestures import Landmark, classify_pose, extended_fingers


def hand_with_fingers(*, index: bool, middle: bool, ring: bool, pinky: bool) -> list[Landmark]:
    landmarks = [Landmark(0.5, 0.5, 0.0) for _ in range(21)]
    for is_extended, tip_index, pip_index in (
        (index, 8, 6),
        (middle, 12, 10),
        (ring, 16, 14),
        (pinky, 20, 18),
    ):
        landmarks[pip_index] = Landmark(0.5, 0.50, 0.0)
        landmarks[tip_index] = Landmark(0.5, 0.25 if is_extended else 0.72, 0.0)
    return landmarks


class GestureTests(unittest.TestCase):
    def test_index_only_starts_tracking(self) -> None:
        landmarks = hand_with_fingers(index=True, middle=False, ring=False, pinky=False)

        self.assertEqual(classify_pose(landmarks), "index_up")

    def test_open_palm_stops_tracking(self) -> None:
        landmarks = hand_with_fingers(index=True, middle=True, ring=True, pinky=True)

        self.assertEqual(classify_pose(landmarks), "open_palm")

    def test_fist_stops_tracking(self) -> None:
        landmarks = hand_with_fingers(index=False, middle=False, ring=False, pinky=False)

        self.assertEqual(classify_pose(landmarks), "fist")

    def test_extended_fingers_uses_margin(self) -> None:
        landmarks = hand_with_fingers(index=False, middle=False, ring=False, pinky=False)
        landmarks[8] = Landmark(0.5, 0.49, 0.0)
        landmarks[6] = Landmark(0.5, 0.50, 0.0)

        self.assertFalse(extended_fingers(landmarks)["index"])


if __name__ == "__main__":
    unittest.main()
