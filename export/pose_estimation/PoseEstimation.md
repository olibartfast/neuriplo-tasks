# Pose Estimation Export

Export tools for HuggingFace pose estimation models to ONNX format.

## Supported Models

| Model | HuggingFace Class | Input Size | Output |
|-------|-------------------|------------|--------|
| ViTPose | `VitPoseForPoseEstimation` | 256x192 | Heatmaps [batch, 17, 64, 48] |

ViTPose expects a cropped person image as input (from a preceding object detector). The output is per-keypoint heatmaps for 17 COCO keypoints.

## Usage

### ViTPose

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

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | HuggingFace model name | `usyd-community/vitpose-base-simple` |
| `--output` | Output ONNX file path | `vitpose.onnx` |
| `--image-height` | Input image height | 256 |
| `--image-width` | Input image width | 192 |
| `--opset` | ONNX opset version | 14 |
| `--static-batch` | Disable dynamic batch size | Off |
| `--test` | Run ONNX Runtime inference test | Off |

## Requirements

```
torch
onnx
numpy
transformers
onnxruntime  # optional, for --test
```

## Preprocessing (vision-core C++)

ViTPose uses standard ImageNet preprocessing:
- Resize to 256x192 (or model-specific size)
- BGR to RGB conversion
- Rescale to [0, 1]
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- NCHW format

## Notes

- ViTPose is a top-down pose estimator: it requires a person bounding box from a detector (e.g., YOLO) as input
- The model outputs heatmaps that are decoded into keypoint coordinates during postprocessing
- 17 COCO keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
