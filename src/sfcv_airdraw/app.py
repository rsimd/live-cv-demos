"""OpenCV webcam app for glowing air drawing with finger gestures."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sfcv_airdraw.gestures import Landmark, classify_pose, extended_fingers
from sfcv_airdraw.palette import Color, brighten_color, color_for_stroke


_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "live-cv-airdraw-matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import mediapipe.python.solutions.hands as mp_hands  # noqa: E402

FOOTER_HEIGHT = 146
FOOTER_SLOGAN = "深層学習の力でお絵描きをしてみよう！"
INSTRUCTION_ITEMS: tuple[tuple[str, str, str, Color], ...] = (
    ("GU", "グー", "区切る / 次の線へ", (140, 180, 255)),
    ("INDEX", "人差し指", "描く", (80, 255, 190)),
    ("PA", "パー", "止める", (80, 180, 255)),
)
JAPANESE_FONT_CANDIDATES = (
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


@dataclass(slots=True)
class TrailPoint:
    """A point in a drawn stroke."""

    x: int
    y: int
    created_at: float
    stroke_id: int


@dataclass(slots=True)
class Particle:
    """Short-lived sparkle or burst particle."""

    x: float
    y: float
    vx: float
    vy: float
    radius: float
    born_at: float
    ttl: float
    color: Color


@dataclass(slots=True)
class HandObservation:
    """Detected hand state for one video frame."""

    hand_id: str
    pose: str
    tip_xy: tuple[int, int]
    index_extended: bool


@dataclass(slots=True)
class HandDrawState:
    """Per-hand drawing state."""

    recording: bool = False
    stroke_id: int = 0
    last_point: tuple[int, int] | None = None


def clamp(value: int, lower: int, upper: int) -> int:
    """Clamp an integer to an inclusive interval."""

    return max(lower, min(value, upper))


def bgr_to_rgb(color: Color) -> Color:
    """Convert a BGR color tuple to RGB."""

    return color[2], color[1], color[0]


def load_japanese_font(size: int) -> ImageFont.ImageFont:
    """Load a Japanese-capable font with a safe fallback."""

    for font_path in JAPANESE_FONT_CANDIDATES:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


class AirDrawApp:
    """CPU-only hand-tracking air drawing demo."""

    def __init__(
        self,
        *,
        camera: int,
        width: int,
        height: int,
        lifetime_seconds: float,
        min_distance: float,
        mirror: bool,
        fullscreen: bool,
        show_hud: bool,
    ) -> None:
        self.camera = camera
        self.width = width
        self.height = height
        self.lifetime_seconds = lifetime_seconds
        self.min_distance = min_distance
        self.mirror = mirror
        self.fullscreen = fullscreen
        self.show_hud = show_hud

        self.window_name = "Live CV Air Draw"
        self.points: list[TrailPoint] = []
        self.particles: list[Particle] = []
        self.rng = np.random.default_rng()
        self.stroke_id = 0
        self.hand_states: dict[str, HandDrawState] = {}
        self.last_frame_at = time.monotonic()
        self.footer_label_font = load_japanese_font(26)
        self.footer_action_font = load_japanese_font(20)
        self.footer_roman_font = load_japanese_font(15)
        self.footer_slogan_font = load_japanese_font(24)

    def run(self) -> None:
        """Run the camera loop."""

        cap = cv2.VideoCapture(self.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera}.")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(
                self.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:
            try:
                self._run_loop(cap, hands)
            finally:
                cap.release()
                cv2.destroyAllWindows()

    def _run_loop(self, cap: cv2.VideoCapture, hands: object) -> None:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if self.mirror:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()
            dt = min(max(now - self.last_frame_at, 1.0 / 120.0), 1.0 / 15.0)
            self.last_frame_at = now

            observations = self._read_hands(frame, hands)
            visible_hand_ids = {observation.hand_id for observation in observations}
            self._disconnect_missing_hands(visible_hand_ids)
            for observation in observations:
                state = self._hand_state(observation.hand_id)
                self._update_recording_state(state, observation.pose)
                if state.recording and observation.index_extended:
                    self._add_point(state, observation.tip_xy, now)
                    self._spawn_sparkles(
                        observation.tip_xy[0],
                        observation.tip_xy[1],
                        state.stroke_id,
                        now,
                        amount=2,
                    )

            self._expire_points(now)
            self._update_particles(now, dt)

            output = self._render(frame, now, observations)
            cv2.imshow(self.window_name, output)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                self._clear()

    def _read_hands(
        self,
        frame: np.ndarray,
        hands: object,
    ) -> list[HandObservation]:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            return []

        observations: list[HandObservation] = []
        for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = [
                Landmark(point.x, point.y, point.z)
                for point in hand_landmarks.landmark
            ]
            pose = classify_pose(landmarks)
            fingers = extended_fingers(landmarks)

            index_tip = hand_landmarks.landmark[8]
            tip_xy = (
                clamp(int(index_tip.x * width), 0, width - 1),
                clamp(int(index_tip.y * height), 0, height - 1),
            )
            observations.append(
                HandObservation(
                    hand_id=self._hand_id(results, index),
                    pose=pose,
                    tip_xy=tip_xy,
                    index_extended=fingers["index"],
                )
            )
        return observations

    def _hand_id(self, results: object, index: int) -> str:
        if not getattr(results, "multi_handedness", None):
            return f"hand-{index}"

        handedness = results.multi_handedness[index]
        classifications = handedness.classification
        if not classifications:
            return f"hand-{index}"
        return classifications[0].label

    def _hand_state(self, hand_id: str) -> HandDrawState:
        if hand_id not in self.hand_states:
            self.hand_states[hand_id] = HandDrawState()
        return self.hand_states[hand_id]

    def _disconnect_missing_hands(self, visible_hand_ids: set[str]) -> None:
        for hand_id, state in self.hand_states.items():
            if hand_id not in visible_hand_ids:
                state.last_point = None

    def _update_recording_state(self, state: HandDrawState, pose: str) -> None:
        if pose in {"open_palm", "fist"}:
            state.recording = False
            state.last_point = None
            return

        if pose == "index_up" and not state.recording:
            state.recording = True
            self.stroke_id += 1
            state.stroke_id = self.stroke_id
            state.last_point = None

    def _add_point(self, state: HandDrawState, point: tuple[int, int], now: float) -> None:
        if state.last_point is not None:
            distance = math.dist(state.last_point, point)
            if distance < self.min_distance:
                return

        self.points.append(
            TrailPoint(
                x=point[0],
                y=point[1],
                created_at=now,
                stroke_id=state.stroke_id,
            )
        )
        state.last_point = point

    def _expire_points(self, now: float) -> None:
        fresh_points: list[TrailPoint] = []
        for point in self.points:
            if now - point.created_at >= self.lifetime_seconds:
                self._spawn_burst(point.x, point.y, point.stroke_id, now)
            else:
                fresh_points.append(point)
        self.points = fresh_points

    def _spawn_sparkles(
        self,
        x: int,
        y: int,
        stroke_id: int,
        now: float,
        *,
        amount: int,
    ) -> None:
        base_color = brighten_color(color_for_stroke(stroke_id), 0.45)
        for _ in range(amount):
            angle = float(self.rng.uniform(0.0, math.tau))
            speed = float(self.rng.uniform(15.0, 75.0))
            radius = float(self.rng.uniform(1.0, 2.6))
            color = brighten_color(base_color, float(self.rng.uniform(0.0, 0.45)))
            self.particles.append(
                Particle(
                    x=x + float(self.rng.normal(0.0, 4.0)),
                    y=y + float(self.rng.normal(0.0, 4.0)),
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    radius=radius,
                    born_at=now,
                    ttl=float(self.rng.uniform(0.18, 0.45)),
                    color=color,
                )
            )

    def _spawn_burst(self, x: int, y: int, stroke_id: int, now: float) -> None:
        base_color = color_for_stroke(stroke_id)
        for _ in range(8):
            angle = float(self.rng.uniform(0.0, math.tau))
            speed = float(self.rng.uniform(90.0, 280.0))
            color = brighten_color(base_color, float(self.rng.uniform(0.0, 0.6)))
            self.particles.append(
                Particle(
                    x=float(x),
                    y=float(y),
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed - float(self.rng.uniform(20.0, 90.0)),
                    radius=float(self.rng.uniform(1.5, 4.0)),
                    born_at=now,
                    ttl=float(self.rng.uniform(0.55, 1.25)),
                    color=color,
                )
            )

        if len(self.particles) > 2200:
            self.particles = self.particles[-2200:]

    def _update_particles(self, now: float, dt: float) -> None:
        alive: list[Particle] = []
        for particle in self.particles:
            if now - particle.born_at >= particle.ttl:
                continue
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.vy += 160.0 * dt
            alive.append(particle)
        self.particles = alive

    def _render(
        self,
        frame: np.ndarray,
        now: float,
        observations: Sequence[HandObservation],
    ) -> np.ndarray:
        stroke_layer = np.zeros_like(frame)
        glow_layer = np.zeros_like(frame)

        for previous, current in self._stroke_segments():
            color = color_for_stroke(current.stroke_id)
            start = (previous.x, previous.y)
            end = (current.x, current.y)
            cv2.line(glow_layer, start, end, color, 18, cv2.LINE_AA)
            cv2.line(stroke_layer, start, end, color, 5, cv2.LINE_AA)

        glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=9)
        output = cv2.addWeighted(frame, 1.0, glow_layer, 0.62, 0.0)
        output = cv2.addWeighted(output, 1.0, stroke_layer, 0.96, 0.0)
        output = self._draw_particles(output, now)

        for observation in observations:
            state = self._hand_state(observation.hand_id)
            self._draw_pointer(output, observation.tip_xy, state.recording)

        if self.show_hud:
            self._draw_hud(output, now, observations)

        return self._append_instruction_footer(output)

    def _stroke_segments(self) -> list[tuple[TrailPoint, TrailPoint]]:
        previous_by_stroke: dict[int, TrailPoint] = {}
        segments: list[tuple[TrailPoint, TrailPoint]] = []
        for point in self.points:
            previous = previous_by_stroke.get(point.stroke_id)
            if previous is not None:
                segments.append((previous, point))
            previous_by_stroke[point.stroke_id] = point
        return segments

    def _draw_particles(self, frame: np.ndarray, now: float) -> np.ndarray:
        sparkle_layer = np.zeros_like(frame)
        sparkle_glow = np.zeros_like(frame)

        for particle in self.particles:
            age = now - particle.born_at
            fade = max(0.0, 1.0 - age / particle.ttl)
            center = (int(particle.x), int(particle.y))
            radius = max(1, int(particle.radius * fade))
            color = tuple(int(channel * fade) for channel in particle.color)
            cv2.circle(sparkle_glow, center, radius * 5, color, -1, cv2.LINE_AA)
            cv2.circle(sparkle_layer, center, radius, color, -1, cv2.LINE_AA)
            arm = radius * 4
            cv2.line(
                sparkle_layer,
                (center[0] - arm, center[1]),
                (center[0] + arm, center[1]),
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                sparkle_layer,
                (center[0], center[1] - arm),
                (center[0], center[1] + arm),
                color,
                1,
                cv2.LINE_AA,
            )

        sparkle_glow = cv2.GaussianBlur(sparkle_glow, (0, 0), sigmaX=5)
        frame = cv2.addWeighted(frame, 1.0, sparkle_glow, 0.75, 0.0)
        return cv2.addWeighted(frame, 1.0, sparkle_layer, 1.0, 0.0)

    def _draw_pointer(
        self,
        frame: np.ndarray,
        point: tuple[int, int],
        recording: bool,
    ) -> None:
        color = (80, 255, 190) if recording else (40, 210, 255)
        glow = np.zeros_like(frame)
        cv2.circle(glow, point, 26, color, -1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=8)
        cv2.addWeighted(frame, 1.0, glow, 0.55, 0.0, dst=frame)
        cv2.circle(frame, point, 9, color, 2, cv2.LINE_AA)
        cv2.circle(frame, point, 2, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_hud(
        self,
        frame: np.ndarray,
        now: float,
        observations: Sequence[HandObservation],
    ) -> None:
        poses = {observation.pose for observation in observations}
        active_hands = sum(state.recording for state in self.hand_states.values())

        if active_hands > 0:
            status = "DRAWING"
            color = (80, 255, 190)
        elif "open_palm" in poses:
            status = "STOPPED"
            color = (80, 180, 255)
        elif "fist" in poses:
            status = "PAUSED"
            color = (140, 180, 255)
        elif not observations:
            status = "NO HAND"
            color = (180, 180, 180)
        else:
            status = "READY"
            color = (60, 220, 255)

        if self.points:
            oldest = min(point.created_at for point in self.points)
            remaining = max(0.0, self.lifetime_seconds - (now - oldest))
        else:
            remaining = self.lifetime_seconds

        label = (
            f"{status}  hands {active_hands}/2  vanish {remaining:04.1f}s  "
            f"points {len(self.points)}"
        )
        cv2.rectangle(frame, (14, 14), (760, 54), (15, 18, 24), -1)
        cv2.putText(
            frame,
            label,
            (28, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )

    def _append_instruction_footer(self, frame: np.ndarray) -> np.ndarray:
        width = frame.shape[1]
        footer = np.full((FOOTER_HEIGHT, width, 3), (18, 20, 26), dtype=np.uint8)
        cv2.line(footer, (0, 0), (width, 0), (70, 76, 92), 1, cv2.LINE_AA)

        section_width = max(width // len(INSTRUCTION_ITEMS), 1)
        for index, (_, _, _, color) in enumerate(INSTRUCTION_ITEMS):
            x0 = index * section_width

            cv2.circle(footer, (x0 + 38, 54), 18, color, -1, cv2.LINE_AA)
            cv2.circle(footer, (x0 + 38, 54), 22, brighten_color(color, 0.45), 2, cv2.LINE_AA)
            if index > 0:
                cv2.line(footer, (x0, 20), (x0, 88), (55, 60, 74), 1, cv2.LINE_AA)

        self._draw_footer_text(footer)
        return np.vstack((frame, footer))

    def _draw_footer_text(self, footer: np.ndarray) -> None:
        width = footer.shape[1]
        section_width = max(width // len(INSTRUCTION_ITEMS), 1)
        footer_rgb = cv2.cvtColor(footer, cv2.COLOR_BGR2RGB)
        pil_footer = Image.fromarray(footer_rgb)
        draw = ImageDraw.Draw(pil_footer)

        for index, (roman, label, action, color) in enumerate(INSTRUCTION_ITEMS):
            x0 = index * section_width
            text_x = x0 + 72
            draw.text(
                (text_x, 18),
                label,
                font=self.footer_label_font,
                fill=bgr_to_rgb(color),
            )
            draw.text(
                (text_x, 52),
                action,
                font=self.footer_action_font,
                fill=(232, 236, 242),
            )
            draw.text(
                (text_x, 78),
                roman,
                font=self.footer_roman_font,
                fill=(154, 162, 176),
            )

        draw.line((22, 106, width - 22, 106), fill=(70, 76, 92), width=1)
        slogan_bbox = draw.textbbox((0, 0), FOOTER_SLOGAN, font=self.footer_slogan_font)
        slogan_width = slogan_bbox[2] - slogan_bbox[0]
        draw.text(
            ((width - slogan_width) // 2, 113),
            FOOTER_SLOGAN,
            font=self.footer_slogan_font,
            fill=(248, 242, 210),
        )

        footer[:] = cv2.cvtColor(np.asarray(pil_footer), cv2.COLOR_RGB2BGR)

    def _clear(self) -> None:
        self.points.clear()
        self.particles.clear()
        for state in self.hand_states.values():
            state.recording = False
            state.last_point = None
        self.stroke_id += 1


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Live CV air drawing demo.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
    parser.add_argument(
        "--lifetime",
        type=float,
        default=60.0,
        help="Seconds before drawn points burst and disappear.",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=5.0,
        help="Minimum pixel distance between stored stroke points.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable selfie-style horizontal mirroring.",
    )
    parser.add_argument("--fullscreen", action="store_true", help="Run fullscreen.")
    parser.add_argument("--no-hud", action="store_true", help="Hide status overlay.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the demo application."""

    args = build_parser().parse_args(argv)
    app = AirDrawApp(
        camera=args.camera,
        width=args.width,
        height=args.height,
        lifetime_seconds=args.lifetime,
        min_distance=args.min_distance,
        mirror=not args.no_mirror,
        fullscreen=args.fullscreen,
        show_hud=not args.no_hud,
    )
    app.run()


if __name__ == "__main__":
    main()
