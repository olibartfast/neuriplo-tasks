# Pose Estimation Export

Export tools for pose estimation models to ONNX format.

## Supported Models

| Model | Source | Input Size | Output Shape | vision-core type |
|-------|--------|------------|--------------|-----------------|
| YOLOv8-pose | Ultralytics | 640×640 | `[1, 56, 8400]` | `yolov8pose` |
| YOLO11-pose | Ultralytics | 640×640 | `[1, 56, 8400]` | `yolo11pose` |
| YOLO26-pose | Ultralytics | 640×640 | `[1, 56, 8400]` | `yolo26pose` |
| YOLOv5-pose | Ultralytics yolov5 repo | 640×640 | `[1, 8400, 56]` | `yolov5pose` |
| ViTPose | HuggingFace `VitPoseForPoseEstimation` | 256×192 | `[1, 17, 64, 48]` | `vitpose` |

All models detect 17 COCO keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.

---

## YOLO Pose (YOLOv5 / YOLOv8 / YOLO11 / YOLO26)

YOLO pose models are **bottom-up** detectors: they detect all persons and their keypoints in a single forward pass, returning per-person bounding boxes alongside keypoints.

### Output Format

| Model Family | Shape | Channels |
|---|---|---|
| YOLOv8 / YOLO11 / YOLO26 | `[batch, 56, 8400]` | `4 bbox + 1 conf + 51 kpts` (no objectness) |
| YOLOv5 | `[batch, 8400, 56]` | `4 bbox + 1 obj + 51 kpts` (with objectness) |

The 51 keypoint channels are laid out as `kpt_x, kpt_y, kpt_conf` × 17.

### Quick Export (Ultralytics — YOLOv8 / YOLO11 / YOLO26)

```bash
cd export/pose_estimation/yolo_pose

# Download weights and export
python export.py --model yolov8n-pose --download-weights --format onnx

# Local weights
python export.py --model yolov8s-pose.pt --format onnx

# YOLO11 pose
python export.py --model yolo11s-pose.pt --format onnx

# Export and test with ONNX Runtime
python export.py --model yolov8n-pose.pt --format onnx --test
```

### Quick Export (YOLOv5 — requires repo clone)

```bash
git clone https://github.com/ultralytics/yolov5.git repositories/yolov5
pip install -r repositories/yolov5/requirements.txt

python export.py \
    --model yolov5s-pose.pt \
    --version v5 \
    --repo-dir ./repositories/yolov5 \
    --format onnx
```

### Manual Export (Alternative)

```bash
# YOLOv8 / YOLO11 / YOLO26 via ultralytics CLI
pip install ultralytics>=8.3.0
yolo export model=yolov8n-pose.pt format=onnx imgsz=640 opset=12
```

### Available Weights

| Model | Size |
|-------|------|
| `yolov8n-pose` | nano |
| `yolov8s-pose` | small |
| `yolov8m-pose` | medium |
| `yolov8l-pose` | large |
| `yolov8x-pose` | extra-large |
| `yolov8x-pose-p6` | extra-large (1280px) |
| `yolo11n-pose` | nano |
| `yolo11s-pose` | small |
| `yolo11m-pose` | medium |
| `yolo11l-pose` | large |
| `yolo11x-pose` | extra-large |
| `yolo26n-pose` | nano |
| `yolo26s-pose` | small |
| `yolo26m-pose` | medium |
| `yolo26l-pose` | large |
| `yolo26x-pose` | extra-large |

### Export Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to `.pt` weights or model name | required |
| `--version` | `auto`, `v5`, `v8`, `v11`, `26` | `auto` |
| `--format` | `onnx` or `tensorrt` | `onnx` |
| `--output-dir` | Output directory | `./exported_models` |
| `--imgsz` | Input image size | `640` |
| `--batch-size` | Batch size | `1` |
| `--opset` | ONNX opset version | `12` |
| `--dynamic` | Enable dynamic batch size | off |
| `--no-simplify` | Skip ONNX simplification | off |
| `--repo-dir` | yolov5 repo path (v5 only) | — |
| `--download-weights` | Auto-download weights | off |
| `--test` | Run ONNX Runtime inference test | off |

### TensorRT

```bash
# Convert the ONNX model to a TensorRT engine
trtexec --onnx=yolov8n-pose.onnx --saveEngine=yolov8n-pose.engine --fp16
```

### Preprocessing (vision-core C++)

YOLO pose uses the same **letterbox** preprocessing as YOLO detection:
- Resize with aspect ratio preserved, pad to square with grey (114, 114, 114)
- Normalize to `[0, 1]` (divide by 255)
- BGR → RGB
- NCHW format (float32)

---

## ViTPose

ViTPose is a **top-down** pose estimator: it requires a person bounding box from a preceding detector (e.g., YOLO) and outputs heatmaps for a single cropped person image.

### Usage

```bash
# Default export
python export/pose_estimation/vitpose/export_vitpose_to_onnx.py

# Custom model and settings
python export/pose_estimation/vitpose/export_vitpose_to_onnx.py \
    --model usyd-community/vitpose-base-simple \
    --output vitpose.onnx \
    --image-height 256 \
    --image-width 192 \
    --test
```

### Available Models

- `usyd-community/vitpose-base-simple` (default)
- `usyd-community/vitpose-base`
- `usyd-community/vitpose-plus-base`

### Export Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | HuggingFace model name | `usyd-community/vitpose-base-simple` |
| `--output` | Output ONNX file path | `vitpose.onnx` |
| `--image-height` | Input image height | `256` |
| `--image-width` | Input image width | `192` |
| `--opset` | ONNX opset version | `14` |
| `--static-batch` | Disable dynamic batch size | off |
| `--test` | Run ONNX Runtime inference test | off |

### Preprocessing (vision-core C++)

ViTPose uses standard ImageNet preprocessing:
- Resize to 256×192 (or model-specific size)
- BGR → RGB conversion
- Normalize to `[0, 1]`
- ImageNet normalization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- NCHW format (float32)

### Requirements

```
torch
onnx
numpy
transformers
onnxruntime  # optional, for --test
```

---

## Choosing a Model

| Use case | Recommended model |
|----------|------------------|
| Fast single-stage detection + pose | `yolov8n-pose` or `yolo11n-pose` |
| Highest accuracy single-stage | `yolov8x-pose` or `yolo11x-pose` |
| Top-down accuracy (person already cropped) | `vitpose-base` or `vitpose-plus-base` |
