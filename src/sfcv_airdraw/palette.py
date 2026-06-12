"""Color palette for air drawing strokes."""

from __future__ import annotations

Color = tuple[int, int, int]


# OpenCV uses BGR channel order. Bright colors remain visible over camera frames.
DRAWING_PALETTE: tuple[Color, ...] = (
    (255, 64, 64),
    (255, 128, 64),
    (255, 192, 64),
    (255, 240, 64),
    (192, 255, 64),
    (128, 255, 64),
    (64, 255, 64),
    (64, 255, 128),
    (64, 255, 192),
    (64, 255, 240),
    (64, 192, 255),
    (64, 128, 255),
    (64, 64, 255),
    (128, 64, 255),
    (192, 64, 255),
    (240, 64, 255),
    (255, 64, 192),
    (255, 64, 128),
    (255, 255, 255),
    (220, 220, 255),
    (255, 220, 220),
    (220, 255, 220),
    (180, 255, 255),
    (255, 180, 255),
    (255, 255, 180),
    (180, 180, 255),
    (180, 255, 180),
    (255, 180, 180),
    (120, 240, 255),
    (255, 240, 120),
)


def color_for_stroke(stroke_id: int) -> Color:
    """Return the display color for a stroke id.

    Args:
        stroke_id: One-based stroke id used by the drawing app.

    Returns:
        BGR color tuple selected from the 30-color palette.
    """

    return DRAWING_PALETTE[(stroke_id - 1) % len(DRAWING_PALETTE)]


def brighten_color(color: Color, factor: float) -> Color:
    """Brighten a BGR color toward white.

    Args:
        color: Base BGR color.
        factor: Blend factor toward white in the range 0.0 to 1.0.

    Returns:
        Brightened BGR color.
    """

    clamped = max(0.0, min(factor, 1.0))
    return tuple(int(channel + (255 - channel) * clamped) for channel in color)
