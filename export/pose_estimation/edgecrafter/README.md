# EdgeCrafter Pose Estimation ONNX Export

Export an EdgeCrafter human pose estimation checkpoint to ONNX for use with C++ inference.

## Prerequisites

- Python 3.11
- Git

## Steps

```bash
# 1. Clone EdgeCrafter
git clone --depth 1 https://github.com/Intellindust-AI-Lab/EdgeCrafter
cd EdgeCrafter

# 2. Create virtual environment and install dependencies
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r ecpose/requirements.txt
pip install onnx onnxsim onnxscript

# 3. Download a pose checkpoint
# Supported sizes: ecpose_s, ecpose_m
curl -L -o ecpose_s.pth \
  https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_s.pth

# 4. Export to ONNX
python tools/deployment/export_onnx.py \
  -c configs/ecpose/ecpose_s_coco.yml \
  -r ecpose_s.pth \
  --check \
  --simplify
```

The exported `ecpose_s.onnx` (and optional `ecpose_s.onnx.data`) will be written next to the checkpoint.

## ONNX I/O Contract

| Role    | Name               | Dtype  | Shape              | Description                          |
|---------|--------------------|--------|--------------------|--------------------------------------|
| Input   | `images`           | float  | `[1, 3, H, W]`     | NCHW preprocessed image              |
| Input   | `orig_target_sizes`| int64  | `[1, 2]`           | Original image `[width, height]`     |
| Output  | `labels`           | int64  | `[1, N]`           | Class IDs (1 = person)               |
| Output  | `scores`           | float  | `[1, N]`           | Confidence scores                    |
| Output  | `keypoints`        | float  | `[1, N, 17, K]`    | COCO 17-keypoints; K=2 `[x,y]` or K=3 `[x,y,conf]` |

Preprocessing: resize to `[H, W]` (direct, no letterbox) → BGR to RGB → scale to `[0,1]` → ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

The ONNX graph handles top-k selection and keypoint coordinate scaling internally. The C++ postprocessor applies a `label_offset` of -1 (person `1` → `0`), derives bounding boxes from visible keypoints, and uses the standard COCO 17-keypoint skeleton.

Keypoints are drawn with the COCO skeleton: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.
