from __future__ import annotations

import unittest

import numpy as np

from sfcv_airdraw.app import FOOTER_HEIGHT, AirDrawApp


def make_app() -> AirDrawApp:
    return AirDrawApp(
        camera=0,
        width=640,
        height=480,
        lifetime_seconds=60.0,
        min_distance=5.0,
        mirror=True,
        fullscreen=False,
        show_hud=True,
    )


class AirDrawStateTests(unittest.TestCase):
    def test_index_up_starts_recording(self) -> None:
        app = make_app()
        state = app._hand_state("Left")

        app._update_recording_state(state, "index_up")

        self.assertTrue(state.recording)
        self.assertEqual(app.stroke_id, 1)
        self.assertEqual(state.stroke_id, 1)

    def test_open_palm_stops_recording(self) -> None:
        app = make_app()
        state = app._hand_state("Left")
        app._update_recording_state(state, "index_up")
        state.last_point = (10, 20)

        app._update_recording_state(state, "open_palm")

        self.assertFalse(state.recording)
        self.assertIsNone(state.last_point)

    def test_fist_stops_recording(self) -> None:
        app = make_app()
        state = app._hand_state("Left")
        app._update_recording_state(state, "index_up")
        state.last_point = (10, 20)

        app._update_recording_state(state, "fist")

        self.assertFalse(state.recording)
        self.assertIsNone(state.last_point)

    def test_fist_separates_the_next_curve(self) -> None:
        app = make_app()
        state = app._hand_state("Left")
        app._update_recording_state(state, "index_up")
        app._add_point(state, (10, 20), now=0.0)

        app._update_recording_state(state, "fist")
        app._update_recording_state(state, "index_up")
        app._add_point(state, (100, 120), now=1.0)

        self.assertEqual([point.stroke_id for point in app.points], [1, 2])
        self.assertEqual(state.last_point, (100, 120))

    def test_two_hands_draw_independent_curves(self) -> None:
        app = make_app()
        left = app._hand_state("Left")
        right = app._hand_state("Right")

        app._update_recording_state(left, "index_up")
        app._add_point(left, (10, 20), now=0.0)
        app._update_recording_state(right, "index_up")
        app._add_point(right, (120, 140), now=0.0)

        self.assertEqual([point.stroke_id for point in app.points], [1, 2])
        self.assertEqual(left.last_point, (10, 20))
        self.assertEqual(right.last_point, (120, 140))

    def test_interleaved_two_hand_points_still_create_two_segments(self) -> None:
        app = make_app()
        left = app._hand_state("Left")
        right = app._hand_state("Right")

        app._update_recording_state(left, "index_up")
        app._update_recording_state(right, "index_up")
        app._add_point(left, (10, 20), now=0.0)
        app._add_point(right, (120, 140), now=0.0)
        app._add_point(left, (20, 30), now=1.0)
        app._add_point(right, (130, 150), now=1.0)

        segments = app._stroke_segments()

        self.assertEqual([(a.stroke_id, b.stroke_id) for a, b in segments], [(1, 1), (2, 2)])
        self.assertEqual([(b.x, b.y) for _, b in segments], [(20, 30), (130, 150)])

    def test_missing_hand_does_not_reconnect_to_previous_point(self) -> None:
        app = make_app()
        left = app._hand_state("Left")
        right = app._hand_state("Right")
        left.last_point = (10, 20)
        right.last_point = (120, 140)

        app._disconnect_missing_hands({"Right"})

        self.assertIsNone(left.last_point)
        self.assertEqual(right.last_point, (120, 140))

    def test_instruction_footer_is_appended_below_frame(self) -> None:
        app = make_app()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        output = app._append_instruction_footer(frame)

        self.assertEqual(output.shape, (480 + FOOTER_HEIGHT, 640, 3))
        self.assertTrue(output[480:].any())


if __name__ == "__main__":
    unittest.main()
