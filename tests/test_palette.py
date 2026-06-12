from __future__ import annotations

import unittest

from sfcv_airdraw.palette import DRAWING_PALETTE, brighten_color, color_for_stroke


class PaletteTests(unittest.TestCase):
    def test_palette_has_30_unique_colors(self) -> None:
        self.assertEqual(len(DRAWING_PALETTE), 30)
        self.assertEqual(len(set(DRAWING_PALETTE)), 30)

    def test_stroke_colors_cycle_through_palette(self) -> None:
        self.assertEqual(color_for_stroke(1), DRAWING_PALETTE[0])
        self.assertEqual(color_for_stroke(30), DRAWING_PALETTE[29])
        self.assertEqual(color_for_stroke(31), DRAWING_PALETTE[0])

    def test_brighten_color_moves_toward_white(self) -> None:
        self.assertEqual(brighten_color((100, 150, 200), 0.0), (100, 150, 200))
        self.assertEqual(brighten_color((100, 150, 200), 1.0), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
