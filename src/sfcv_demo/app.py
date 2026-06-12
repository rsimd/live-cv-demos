from __future__ import annotations

import argparse
import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor, nn
from torchvision.models import resnet50
from ultralytics import YOLO

Frame = UInt8[NDArray[np.uint8], "height width 3"]
BBox = tuple[int, int, int, int]

EMOTION_LABELS = (
    "Neutral",
    "Happiness",
    "Sadness",
    "Surprise",
    "Fear",
    "Disgust",
    "Anger",
)

EMOTION_MEAN_BGR = np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMOTION_GRAPH_COLORS_BGR = (
    (220, 220, 220),  # Neutral
    (60, 220, 60),  # Happiness
    (255, 120, 40),  # Sadness
    (0, 220, 255),  # Surprise
    (180, 90, 255),  # Fear
    (80, 180, 180),  # Disgust
    (60, 60, 255),  # Anger
)


@dataclass(frozen=True, slots=True)
class DemoConfig:
    camera: int | str
    width: int
    height: int
    imgsz: int
    conf: float
    device: str
    detect: bool
    pose: bool
    segment: bool
    emotion: bool
    detect_every: int
    pose_every: int
    segment_every: int
    emotion_every: int
    detect_model: str
    detect_fallback_model: str
    pose_model: str
    pose_fallback_model: str
    segment_model: str
    segment_fallback_model: str
    yolo_cache_dir: Path
    emotion_model_id: str
    emotion_filename: str
    emotion_model_path: Path | None
    emotion_cache_dir: Path
    emotion_graph: bool
    emotion_graph_height: int
    emotion_history: int
    max_faces: int
    min_face_size: int
    torch_threads: int


@dataclass(frozen=True, slots=True)
class EmotionPrediction:
    box: BBox
    label: str
    confidence: float
    probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PosePrediction:
    box: BBox
    track_id: int
    label: str
    speed_px_s: float


@dataclass(slots=True)
class TrackState:
    track_id: int
    center: tuple[float, float]
    timestamp: float
    speed_px_s: float = 0.0


class VggFaceResNetBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        stride: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.i_downsample: nn.Sequential | None = None
        if downsample:
            self.i_downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.batch_norm1(self.conv1(x)))
        out = self.relu(self.batch_norm2(self.conv2(out)))
        out = self.batch_norm3(self.conv3(out))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        return self.relu(out + identity)


class EmoAffectNetStaticModel(nn.Module):
    def __init__(self, classes: int = len(EMOTION_LABELS)) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv_layer_s2_same = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.layer1 = self._make_layer(64, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, classes)

    def _make_layer(self, mid_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [
            VggFaceResNetBlock(
                self.in_channels,
                mid_channels,
                stride=stride,
                downsample=True,
            )
        ]
        self.in_channels = mid_channels * VggFaceResNetBlock.expansion
        for _ in range(1, blocks):
            layers.append(VggFaceResNetBlock(self.in_channels, mid_channels))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.batch_norm1(self.conv_layer_s2_same(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class SimplePersonTracker:
    def __init__(self, max_distance_px: float = 120.0, stale_after_s: float = 1.5) -> None:
        self._tracks: list[TrackState] = []
        self._next_id = 1
        self._max_distance_px = max_distance_px
        self._stale_after_s = stale_after_s

    def assign(self, boxes: list[BBox], timestamp: float) -> list[TrackState]:
        assignments: list[TrackState] = []
        available_tracks = [track for track in self._tracks if timestamp - track.timestamp <= self._stale_after_s]

        for box in boxes:
            center = box_center(box)
            best_track: TrackState | None = None
            best_distance = self._max_distance_px

            for track in available_tracks:
                distance = point_distance(center, track.center)
                if distance < best_distance:
                    best_track = track
                    best_distance = distance

            if best_track is None:
                best_track = TrackState(self._next_id, center, timestamp)
                self._next_id += 1
            else:
                dt = max(timestamp - best_track.timestamp, 1e-3)
                best_track.speed_px_s = point_distance(center, best_track.center) / dt
                best_track.center = center
                best_track.timestamp = timestamp
                available_tracks.remove(best_track)

            assignments.append(best_track)

        self._tracks = assignments + available_tracks
        return assignments


class EmotionRecognizer:
    def __init__(
        self,
        *,
        model_id: str,
        filename: str,
        model_path: Path | None,
        cache_dir: Path,
        device: str,
        max_faces: int,
        min_face_size: int,
    ) -> None:
        self._device = torch.device(device)
        self._max_faces = max_faces
        self._min_face_size = min_face_size

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._face_detector = cv2.CascadeClassifier(str(cascade_path))
        if self._face_detector.empty():
            raise RuntimeError(f"OpenCV face cascade was not loaded: {cascade_path}")

        resolved_path = model_path or self._download_model(model_id, filename, cache_dir)
        self._model = self._load_model(resolved_path).to(self._device)
        self._model.eval()

    @staticmethod
    def _download_model(model_id: str, filename: str, cache_dir: Path) -> Path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_dir = cache_dir / model_id.replace("/", "__")
        local_path = local_dir / filename
        if local_path.exists():
            return local_path
        local_path = hf_hub_download(repo_id=model_id, filename=filename, local_dir=local_dir)
        return Path(local_path)

    def _load_model(self, model_path: Path) -> nn.Module:
        try:
            model = torch.jit.load(str(model_path), map_location=self._device)
            if isinstance(model, nn.Module):
                return model
        except Exception:
            pass

        state = self._load_state_dict(model_path)
        state = normalize_state_dict(state)
        if "conv_layer_s2_same.weight" in state and "fc2.weight" in state:
            model = EmoAffectNetStaticModel(classes=len(EMOTION_LABELS))
            model.load_state_dict(state, strict=True)
            return model

        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(EMOTION_LABELS))

        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded_count = len(state) - len(unexpected)
        if loaded_count < max(10, len(state) // 2):
            raise RuntimeError(
                "The Hugging Face emotion model is neither TorchScript nor a compatible "
                f"ResNet50 state_dict. Missing={len(missing)} unexpected={len(unexpected)}"
            )
        return model

    @staticmethod
    def _load_state_dict(model_path: Path) -> dict[str, Tensor]:
        try:
            obj = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(model_path, map_location="cpu")

        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        if isinstance(obj, dict) and "model_state_dict" in obj and isinstance(
            obj["model_state_dict"], dict
        ):
            obj = obj["model_state_dict"]
        if not isinstance(obj, dict):
            raise RuntimeError(f"Unsupported emotion model payload: {type(obj)!r}")
        return obj

    def predict_frame(self, frame: Frame) -> list[EmotionPrediction]:
        face_boxes = self._detect_faces(frame)
        predictions: list[EmotionPrediction] = []
        for box in face_boxes:
            x1, y1, x2, y2 = box
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            label, confidence, probabilities = self._predict_face(face)
            predictions.append(
                EmotionPrediction(
                    box=box,
                    label=label,
                    confidence=confidence,
                    probabilities=probabilities,
                )
            )
        return predictions

    def _detect_faces(self, frame: Frame) -> list[BBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self._min_face_size, self._min_face_size),
        )
        boxes = [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in faces]
        boxes.sort(key=lambda box: (box[1], box[0]))
        return boxes[: self._max_faces]

    def _predict_face(self, face_bgr: Frame) -> tuple[str, float, tuple[float, ...]]:
        tensor = preprocess_emotion_face(face_bgr).to(self._device)
        with torch.inference_mode():
            logits = self._model(tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs: Float[Tensor, "batch emotion"] = torch.softmax(logits, dim=1)
            confidence, index = torch.max(probs, dim=1)
            probabilities = tuple(float(value) for value in probs[0].cpu().tolist())
        label = EMOTION_LABELS[int(index.item())]
        return label, float(confidence.item()), probabilities


def normalize_state_dict(state: dict[str, Any]) -> dict[str, Tensor]:
    normalized: dict[str, Tensor] = {}
    prefixes = ("module.", "model.", "backbone.")
    for key, value in state.items():
        if not isinstance(value, Tensor):
            continue
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        normalized[new_key] = value
    return normalized


def preprocess_emotion_face(face_bgr: Frame) -> Float[Tensor, "batch channel height width"]:
    resized = cv2.resize(face_bgr, (224, 224), interpolation=cv2.INTER_NEAREST)
    chw = resized.astype(np.float32).transpose(2, 0, 1)
    chw -= EMOTION_MEAN_BGR[:, None, None]
    return torch.from_numpy(chw).unsqueeze(0)


def resolve_yolo_weight(model_name: str, cache_dir: Path) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.is_absolute() and model_path.exists():
        return str(model_path)

    if model_path.name == model_name:
        candidates = [cache_dir / model_name, PROJECT_ROOT / model_name, Path.cwd() / model_name]
    else:
        candidates = [model_path, PROJECT_ROOT / model_path]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return model_name


def cache_yolo_weight(model_name: str, cache_dir: Path, model: YOLO) -> None:
    model_path = Path(model_name)
    if model_path.name != model_name:
        return

    target = cache_dir / model_name
    if target.exists():
        return

    candidate_paths: list[Path] = []
    for attr in ("ckpt_path", "pt_path"):
        value = getattr(model, attr, None)
        if value:
            candidate_paths.append(Path(value))
    candidate_paths.extend([PROJECT_ROOT / model_name, Path.cwd() / model_name])

    for source in candidate_paths:
        if not source.exists():
            continue
        if source.resolve() == target.resolve():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return


def load_yolo_model(
    model_name: str,
    fallback_model_name: str,
    label: str,
    cache_dir: Path,
) -> YOLO:
    try:
        model = YOLO(resolve_yolo_weight(model_name, cache_dir))
        cache_yolo_weight(model_name, cache_dir, model)
        return model
    except Exception as exc:
        if fallback_model_name and fallback_model_name != model_name:
            print(f"[{label}] failed to load {model_name}: {exc}")
            print(f"[{label}] falling back to {fallback_model_name}")
            model = YOLO(resolve_yolo_weight(fallback_model_name, cache_dir))
            cache_yolo_weight(fallback_model_name, cache_dir, model)
            return model
        raise


def predict_yolo(model: YOLO, frame: Frame, config: DemoConfig):
    return model.predict(
        source=frame,
        imgsz=config.imgsz,
        conf=config.conf,
        device=config.device,
        verbose=False,
    )[0]


def plot_yolo_result(
    result: Any,
    frame: Frame,
    *,
    boxes: bool = True,
    masks: bool = True,
    labels: bool = True,
    conf: bool = True,
) -> Frame:
    if result is None:
        return frame
    try:
        return result.plot(
            img=frame,
            line_width=2,
            boxes=boxes,
            masks=masks,
            labels=labels,
            conf=conf,
        )
    except TypeError:
        return result.plot(img=frame)


def extract_pose_predictions(result: Any, tracker: SimplePersonTracker, timestamp: float) -> list[PosePrediction]:
    if result is None or result.boxes is None or result.keypoints is None:
        return []

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    keypoints_xy = result.keypoints.xy.cpu().numpy()
    keypoints_conf = result.keypoints.conf
    conf_np = keypoints_conf.cpu().numpy() if keypoints_conf is not None else None

    boxes: list[BBox] = []
    for xyxy in boxes_xyxy:
        x1, y1, x2, y2 = xyxy.astype(int).tolist()
        boxes.append((x1, y1, x2, y2))

    tracks = tracker.assign(boxes, timestamp)
    predictions: list[PosePrediction] = []
    for idx, box in enumerate(boxes):
        conf = conf_np[idx] if conf_np is not None and idx < len(conf_np) else None
        label = classify_body_pose(keypoints_xy[idx], conf, box, tracks[idx].speed_px_s)
        predictions.append(
            PosePrediction(
                box=box,
                track_id=tracks[idx].track_id,
                label=label,
                speed_px_s=tracks[idx].speed_px_s,
            )
        )
    return predictions


def classify_body_pose(
    keypoints: NDArray[np.float32],
    confidence: NDArray[np.float32] | None,
    box: BBox,
    speed_px_s: float,
) -> str:
    if speed_px_s > 55:
        return "walking/moving"

    shoulder_y = mean_keypoint_y(keypoints, confidence, [5, 6])
    hip_y = mean_keypoint_y(keypoints, confidence, [11, 12])
    knee_y = mean_keypoint_y(keypoints, confidence, [13, 14])
    ankle_y = mean_keypoint_y(keypoints, confidence, [15, 16])

    if shoulder_y is not None and hip_y is not None and ankle_y is not None:
        torso_h = max(hip_y - shoulder_y, 1.0)
        leg_h = ankle_y - hip_y
        if knee_y is not None and (knee_y - hip_y) < 0.45 * torso_h:
            return "sitting"
        if leg_h < 0.85 * torso_h:
            return "sitting"
        return "standing"

    x1, y1, x2, y2 = box
    aspect = (y2 - y1) / max(x2 - x1, 1)
    if aspect < 1.35:
        return "sitting?"
    return "standing?"


def mean_keypoint_y(
    keypoints: NDArray[np.float32],
    confidence: NDArray[np.float32] | None,
    indices: list[int],
    threshold: float = 0.25,
) -> float | None:
    values: list[float] = []
    for index in indices:
        if index >= len(keypoints):
            continue
        if confidence is not None and index < len(confidence) and confidence[index] < threshold:
            continue
        x, y = keypoints[index]
        if x <= 0 and y <= 0:
            continue
        values.append(float(y))
    if not values:
        return None
    return float(np.mean(values))


class EmotionProbabilityHistory:
    def __init__(self, max_samples: int) -> None:
        self._samples: deque[tuple[float, ...]] = deque(maxlen=max_samples)
        self.last_face_count = 0

    def append(self, predictions: list[EmotionPrediction]) -> None:
        self.last_face_count = len(predictions)
        if not predictions:
            self._samples.append(tuple(0.0 for _ in EMOTION_LABELS))
            return

        values = np.array([prediction.probabilities for prediction in predictions], dtype=np.float32)
        self._samples.append(tuple(float(value) for value in values.mean(axis=0).tolist()))

    @property
    def samples(self) -> list[tuple[float, ...]]:
        return list(self._samples)

    @property
    def latest(self) -> tuple[float, ...]:
        if not self._samples:
            return tuple(0.0 for _ in EMOTION_LABELS)
        return self._samples[-1]


def draw_pose_predictions(frame: Frame, predictions: list[PosePrediction]) -> None:
    for prediction in predictions:
        x1, y1, _, _ = prediction.box
        label = f"P{prediction.track_id}: {prediction.label}"
        draw_label(frame, label, (x1, max(y1 - 8, 18)), color=(50, 220, 90))


def draw_emotion_predictions(frame: Frame, predictions: list[EmotionPrediction]) -> None:
    for prediction in predictions:
        x1, y1, x2, y2 = prediction.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        label = f"{prediction.label} {prediction.confidence:.2f}"
        draw_label(frame, label, (x1, max(y1 - 8, 18)), color=(255, 0, 255))


def append_emotion_graph(
    frame: Frame,
    history: EmotionProbabilityHistory,
    graph_height: int,
) -> Frame:
    if graph_height <= 0:
        return frame

    _, frame_width = frame.shape[:2]
    panel = np.zeros((graph_height, frame_width, 3), dtype=np.uint8)
    panel[:] = (12, 12, 12)

    left = 58
    right = 190
    top = 24
    bottom = 28
    plot_right = max(left + 1, frame_width - right)
    plot_bottom = graph_height - bottom
    plot_width = max(plot_right - left, 1)
    plot_height = max(plot_bottom - top, 1)

    cv2.putText(
        panel,
        f"Emotion probabilities  faces:{history.last_face_count}",
        (12, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    for value in (0.0, 0.5, 1.0):
        y = int(plot_bottom - value * plot_height)
        cv2.line(panel, (left, y), (plot_right, y), (45, 45, 45), 1)
        cv2.putText(
            panel,
            f"{value:.1f}",
            (10, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

    samples = history.samples
    if samples:
        point_count = len(samples)
        x_denominator = max(point_count - 1, 1)
        for emotion_index, color in enumerate(EMOTION_GRAPH_COLORS_BGR):
            points = []
            for sample_index, sample in enumerate(samples):
                x = int(left + (sample_index / x_denominator) * plot_width)
                y = int(plot_bottom - np.clip(sample[emotion_index], 0.0, 1.0) * plot_height)
                points.append((x, y))
            if len(points) == 1:
                cv2.circle(panel, points[0], 3, color, -1)
            else:
                cv2.polylines(panel, [np.array(points, dtype=np.int32)], False, color, 2)

    legend_x = plot_right + 12
    legend_y = top + 6
    latest = history.latest
    for index, (label, color) in enumerate(zip(EMOTION_LABELS, EMOTION_GRAPH_COLORS_BGR, strict=True)):
        y = legend_y + index * 18
        cv2.line(panel, (legend_x, y - 4), (legend_x + 18, y - 4), color, 2)
        cv2.putText(
            panel,
            f"{label[:8]:<8} {latest[index]:.2f}",
            (legend_x + 24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )

    return np.vstack([frame, panel])


def draw_status(frame: Frame, fps: float, config: DemoConfig) -> None:
    active = []
    if config.detect:
        active.append("detect")
    if config.pose:
        active.append("pose")
    if config.segment:
        active.append("segment")
    if config.emotion:
        active.append("emotion")
    text = f"FPS {fps:.1f} | {'/'.join(active)} | q: quit"
    draw_label(frame, text, (10, 24), color=(40, 180, 255))


def draw_label(frame: Frame, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    x, y = origin
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    w, h = text_size
    cv2.rectangle(frame, (x - 3, y - h - baseline - 3), (x + w + 3, y + baseline + 3), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def box_center(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def parse_camera(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


def positive_interval(value: str) -> int:
    interval = int(value)
    if interval < 1:
        raise argparse.ArgumentTypeError("interval must be >= 1")
    return interval


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-friendly live computer vision demo.")
    parser.add_argument("--camera", default="0", help="Camera index or video path.")
    parser.add_argument("--width", type=int, default=960, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=540, help="Requested camera height.")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--device", default="cpu", help="Ultralytics/PyTorch device.")

    parser.add_argument("--no-detect", action="store_true", help="Disable object detection.")
    parser.add_argument("--no-pose", action="store_true", help="Disable pose estimation.")
    parser.add_argument("--segment", action="store_true", help="Enable instance segmentation.")
    parser.add_argument("--no-emotion", action="store_true", help="Disable facial emotion recognition.")

    parser.add_argument("--detect-every", type=positive_interval, default=3)
    parser.add_argument("--pose-every", type=positive_interval, default=3)
    parser.add_argument("--segment-every", type=positive_interval, default=8)
    parser.add_argument("--emotion-every", type=positive_interval, default=5)
    parser.add_argument("--no-emotion-graph", action="store_true", help="Hide emotion line graph.")
    parser.add_argument("--emotion-graph-height", type=int, default=180)
    parser.add_argument("--emotion-history", type=positive_interval, default=90)

    parser.add_argument("--detect-model", default="yolo11n.pt")
    parser.add_argument("--detect-fallback-model", default="yolov8n.pt")
    parser.add_argument("--pose-model", default="yolo11n-pose.pt")
    parser.add_argument("--pose-fallback-model", default="yolov8n-pose.pt")
    parser.add_argument("--segment-model", default="yolo11n-seg.pt")
    parser.add_argument("--segment-fallback-model", default="yolov8n-seg.pt")
    parser.add_argument("--yolo-cache-dir", type=Path, default=Path("models/yolo"))

    parser.add_argument("--emotion-model-id", default="ElenaRyumina/face_emotion_recognition")
    parser.add_argument("--emotion-filename", default="FER_static_ResNet50_AffectNet.pt")
    parser.add_argument("--emotion-model-path", type=Path, default=None)
    parser.add_argument("--emotion-cache-dir", type=Path, default=Path("models/hf"))
    parser.add_argument("--max-faces", type=int, default=8)
    parser.add_argument("--min-face-size", type=int, default=48)
    parser.add_argument("--torch-threads", type=int, default=0)
    return parser


def config_from_args(args: argparse.Namespace) -> DemoConfig:
    return DemoConfig(
        camera=parse_camera(args.camera),
        width=args.width,
        height=args.height,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        detect=not args.no_detect,
        pose=not args.no_pose,
        segment=args.segment,
        emotion=not args.no_emotion,
        detect_every=args.detect_every,
        pose_every=args.pose_every,
        segment_every=args.segment_every,
        emotion_every=args.emotion_every,
        detect_model=args.detect_model,
        detect_fallback_model=args.detect_fallback_model,
        pose_model=args.pose_model,
        pose_fallback_model=args.pose_fallback_model,
        segment_model=args.segment_model,
        segment_fallback_model=args.segment_fallback_model,
        yolo_cache_dir=args.yolo_cache_dir,
        emotion_model_id=args.emotion_model_id,
        emotion_filename=args.emotion_filename,
        emotion_model_path=args.emotion_model_path,
        emotion_cache_dir=args.emotion_cache_dir,
        emotion_graph=not args.no_emotion_graph,
        emotion_graph_height=args.emotion_graph_height,
        emotion_history=args.emotion_history,
        max_faces=args.max_faces,
        min_face_size=args.min_face_size,
        torch_threads=args.torch_threads,
    )


def run_demo(config: DemoConfig) -> None:
    if config.torch_threads > 0:
        torch.set_num_threads(config.torch_threads)

    detector = (
        load_yolo_model(
            config.detect_model,
            config.detect_fallback_model,
            "detect",
            config.yolo_cache_dir,
        )
        if config.detect
        else None
    )
    pose_model = (
        load_yolo_model(
            config.pose_model,
            config.pose_fallback_model,
            "pose",
            config.yolo_cache_dir,
        )
        if config.pose
        else None
    )
    segment_model = (
        load_yolo_model(
            config.segment_model,
            config.segment_fallback_model,
            "segment",
            config.yolo_cache_dir,
        )
        if config.segment
        else None
    )

    emotion_recognizer: EmotionRecognizer | None = None
    if config.emotion:
        try:
            emotion_recognizer = EmotionRecognizer(
                model_id=config.emotion_model_id,
                filename=config.emotion_filename,
                model_path=config.emotion_model_path,
                cache_dir=config.emotion_cache_dir,
                device=config.device,
                max_faces=config.max_faces,
                min_face_size=config.min_face_size,
            )
        except Exception as exc:
            print(f"[emotion] disabled: {exc}")

    cap = cv2.VideoCapture(config.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera or video source: {config.camera!r}")

    tracker = SimplePersonTracker()
    emotion_history = (
        EmotionProbabilityHistory(config.emotion_history)
        if emotion_recognizer is not None and config.emotion_graph
        else None
    )
    last_detect = None
    last_pose = None
    last_segment = None
    last_emotions: list[EmotionPrediction] = []
    last_pose_predictions: list[PosePrediction] = []
    frame_index = 0
    fps = 0.0
    previous_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            now = time.perf_counter()

            if detector is not None and frame_index % config.detect_every == 0:
                last_detect = predict_yolo(detector, frame, config)
            if segment_model is not None and frame_index % config.segment_every == 0:
                last_segment = predict_yolo(segment_model, frame, config)
            if pose_model is not None and frame_index % config.pose_every == 0:
                last_pose = predict_yolo(pose_model, frame, config)
                last_pose_predictions = extract_pose_predictions(last_pose, tracker, now)
            if emotion_recognizer is not None and frame_index % config.emotion_every == 0:
                last_emotions = emotion_recognizer.predict_frame(frame)
                if emotion_history is not None:
                    emotion_history.append(last_emotions)

            display = frame.copy()
            display = plot_yolo_result(last_detect, display)
            display = plot_yolo_result(
                last_segment,
                display,
                boxes=False,
                labels=False,
                conf=False,
            )
            display = plot_yolo_result(
                last_pose,
                display,
                boxes=False,
                masks=False,
                labels=False,
                conf=False,
            )
            draw_pose_predictions(display, last_pose_predictions)
            draw_emotion_predictions(display, last_emotions)

            dt = max(now - previous_time, 1e-3)
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            previous_time = now
            draw_status(display, fps, config)
            if emotion_history is not None:
                display = append_emotion_graph(display, emotion_history, config.emotion_graph_height)

            cv2.imshow("Live CV Demo", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = build_arg_parser()
    config = config_from_args(parser.parse_args())
    run_demo(config)


if __name__ == "__main__":
    main()
