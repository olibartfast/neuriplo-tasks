# EdgeCrafter Detection ONNX Export

Export an EdgeCrafter detection checkpoint to ONNX for use with C++ inference.

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
pip install -r ecdetseg/requirements.txt
pip install onnx onnxsim onnxscript

# 3. Download a detection checkpoint
# Supported sizes: ecdet_s, ecdet_m, ecdet_l, ecdet_x
curl -L -o ecdet_s.pth \
  https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_s.pth

# 4. Export to ONNX
python tools/deployment/export_onnx.py \
  -c configs/ecdet/ecdet_s.yml \
  -r ecdet_s.pth \
  --check \
  --simplify
```

The exported `ecdet_s.onnx` (and optional `ecdet_s.onnx.data`) will be written next to the checkpoint.

## ONNX I/O Contract

| Role    | Name               | Dtype  | Shape            | Description                          |
|---------|--------------------|--------|------------------|--------------------------------------|
| Input   | `images`           | float  | `[1, 3, H, W]`   | NCHW preprocessed image              |
| Input   | `orig_target_sizes`| int64  | `[1, 2]`         | Original image `[width, height]`     |
| Output  | `labels`           | int64  | `[1, N]`         | Class IDs (0-indexed COCO)           |
| Output  | `boxes`            | float  | `[1, N, 4]`      | `[x1, y1, x2, y2]` in orig coords   |
| Output  | `scores`           | float  | `[1, N]`         | Confidence scores                    |

Preprocessing: resize to `[H, W]` (direct, no letterbox) → BGR to RGB → scale to `[0,1]` → ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

The ONNX graph handles top-k selection and box coordinate scaling internally via `orig_target_sizes`. If `ModelInfo::output_names` is provided, the C++ postprocessor resolves tensors by exact output name (`labels`, `boxes`, `scores`); omitted names keep the documented default order.
