# live-cv-demos

Lightweight CPU-first live computer vision demos for:

- multi-class object detection with Ultralytics YOLO
- multi-person pose estimation with simple standing/sitting/moving labels
- multi-face facial emotion recognition using `ElenaRyumina/face_emotion_recognition`
- optional instance segmentation

The app uses OpenCV for camera capture and drawing. It intentionally runs each model only every
few frames so that a normal laptop CPU has a chance to keep up.

To keep the demo readable, object detection draws boxes, pose draws keypoints and action labels
without pose boxes, and segmentation draws masks without segmentation boxes.

When emotion recognition is enabled, a real-time line graph for all seven emotion classes is shown
below the camera view. If multiple faces are detected, the graph uses the average class probability
across faces.

## Setup

```bash
uv sync
```

This project pins Python to `3.12.2` through `.python-version` because PyTorch, torchvision,
OpenCV, and Ultralytics wheels are more reliable on Python 3.11/3.12 than on bleeding-edge
Python versions.

## Run

```bash
uv run live-cv-demo
```

The old command is kept as a compatibility alias:

```bash
uv run sfcv-demo
```

Press `q` or `Esc` to quit.

First launch downloads YOLO weights and the Hugging Face emotion model. After that, local files are
used first and the app does not download them again when they already exist.

YOLO weights are searched in this order:

- `models/yolo/<model-name>.pt`
- repository root, such as `yolo11n.pt`
- current working directory

The default emotion model file is:

```text
ElenaRyumina/face_emotion_recognition::FER_static_ResNet50_AffectNet.pt
```

Downloaded Hugging Face files are stored under `models/hf/` in this repository. Downloaded or
locally found YOLO weights are stored under `models/yolo/`.

## Lighter CPU Modes

Disable expensive tasks when the laptop is slow:

```bash
uv run live-cv-demo --no-detect
uv run live-cv-demo --no-emotion
uv run live-cv-demo --no-detect --imgsz 256
```

Increase frame skipping:

```bash
uv run live-cv-demo --detect-every 6 --pose-every 4 --emotion-every 10
```

Limit PyTorch CPU threads:

```bash
uv run live-cv-demo --torch-threads 4
```

Use a different local YOLO cache directory:

```bash
uv run live-cv-demo --yolo-cache-dir /path/to/yolo-weights
```

Adjust or hide the emotion graph:

```bash
uv run live-cv-demo --emotion-graph-height 140 --emotion-history 120
uv run live-cv-demo --no-emotion-graph
```

## Air Drawing Demo

`live-cv-airdraw` is a separate open campus app for drawing glowing strokes in the air with an
index finger.

- Raise only the index finger to start drawing.
- Both hands can draw at the same time.
- Show an open palm to stop drawing.
- Make a fist to pause drawing and separate the next curve from the current one.
- Each new stroke cycles through a bright 30-color palette.
- Drawn points burst into sparkles and disappear after 60 seconds.
- The app always shows a bottom instruction bar for `グー`, `人差し指`, and `パー`.
- Press `c` to clear the canvas.
- Press `q` or `Esc` to quit.

Run it with:

```bash
uv run live-cv-airdraw
```

The old command is kept as a compatibility alias:

```bash
uv run sfcv-airdraw
```

Useful options:

```bash
uv run live-cv-airdraw --camera 0 --width 1280 --height 720
uv run live-cv-airdraw --fullscreen
uv run live-cv-airdraw --lifetime 45
```

The air drawing app uses MediaPipe Hands for 21-point hand landmarks and OpenCV for camera
capture, drawing, glow, sparkle, and burst effects. Ultralytics YOLO is kept in this project for
the main camera demo; its pretrained pose models are useful for body keypoints, but they do not
provide finger landmarks out of the box.

## Optional Segmentation

Instance segmentation is disabled by default because it is expensive on CPU.

```bash
uv run live-cv-demo --segment --segment-every 12
```

## Model Notes

Defaults use stable nano-size YOLO models:

- `yolo11n.pt`
- `yolo11n-pose.pt`
- `yolo11n-seg.pt`

If your installed Ultralytics version supports newer model families, you can switch them:

```bash
uv run live-cv-demo \
  --detect-model yolo26n.pt \
  --pose-model yolo26n-pose.pt \
  --segment-model yolo26n-seg.pt
```

The emotion model loader first tries TorchScript. If the file is a state dict, it supports the
VGGFace-style ResNet-50 used by `FER_static_ResNet50_AffectNet.pt`, then falls back to a standard
ResNet-50 classifier head with seven emotion classes. If the model cannot be loaded, the app prints
a warning and keeps the rest of the demo running.

## Privacy

The app does not save camera frames by default. Facial emotion labels are only rough demo
predictions and should not be presented as ground truth.
