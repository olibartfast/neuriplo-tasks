# Depth Estimation Export

Export tools for HuggingFace depth estimation models to ONNX format.

## Supported Models

| Model | HuggingFace Class | Default Model |
|-------|-------------------|---------------|
| Depth Anything V2 | `AutoModelForDepthEstimation` | `depth-anything/Depth-Anything-V2-Small-hf` |

The exporter produces:
- Input tensor: `pixel_values` with shape `[batch, 3, height, width]`
- Output tensor: `predicted_depth` with shape `[batch, height, width]` (or model-equivalent depth output)

## Usage

```bash
# Default export
python export/depth_estimation/depth_anything_v2/export_depth_anything_v2_to_onnx.py

# Custom model and settings
python export/depth_estimation/depth_anything_v2/export_depth_anything_v2_to_onnx.py \
  --model depth-anything/Depth-Anything-V2-Base-hf \
  --output depth_anything_v2_base.onnx \
  --image-height 518 \
  --image-width 518 \
  --test
```

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | HuggingFace model name | `depth-anything/Depth-Anything-V2-Small-hf` |
| `--output` | Output ONNX file path | `depth_anything_v2.onnx` |
| `--image-height` | Input height | 518 |
| `--image-width` | Input width | 518 |
| `--opset` | ONNX opset version | 14 |
| `--static-batch` | Disable dynamic batch size | Off |
| `--test` | Run ONNX Runtime inference test | Off |

## Requirements

```text
torch
onnx
numpy
transformers
onnxruntime  # optional, for --test
```

## Notes

- Keep export input size aligned with your deployment model config for best runtime compatibility.
- The C++ `neuriplo-tasks` depth postprocessor accepts common depth output layouts, including `[B,H,W]` and `[B,1,H,W]`.
