# Gaussian Splatting Export

Export tools for feed-forward Gaussian Splatting models to ONNX format.

## Supported Models

| Model | HuggingFace ID | Default input size | Output layout |
|-------|----------------|--------------------|---------------|
| LGM / LGM-mini | `dylanebert/LGM` | 256 × 256 | `[N, G, 14]` |
| GRM | `GaussianWorld/GRM` | 256 × 256 | `[N, G, 14]` |

### Output tensor layout — `[N, G, 14]`

Each of the G Gaussians is described by 14 values in this order:

| Index | Field | Description |
|-------|-------|-------------|
| 0–2   | x, y, z | 3D position |
| 3–5   | scale_x/y/z | Log-scale per axis |
| 6–9   | rot_w/x/y/z | Rotation quaternion |
| 10    | opacity | Raw logit (apply sigmoid for αblending) |
| 11–13 | sh_r/g/b | Spherical-harmonic DC colour (degree-0) |

## Usage

```bash
# Default export (LGM-mini, 256×256)
python export/gaussian_splatting/lgm/export_lgm_to_onnx.py

# Custom model and settings
python export/gaussian_splatting/lgm/export_lgm_to_onnx.py \
  --model dylanebert/LGM \
  --output lgm.onnx \
  --image-size 256 \
  --test
```

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | HuggingFace model ID | `dylanebert/LGM` |
| `--output` | Output ONNX file path | `lgm.onnx` |
| `--image-size` | Square input resolution | `256` |
| `--opset` | ONNX opset version | `17` |
| `--static-batch` | Disable dynamic batch axis | Off |
| `--test` | Run ONNX Runtime inference test after export | Off |

## Requirements

```text
torch
onnx
numpy
transformers
diffusers
onnxruntime  # optional, for --test
```

## Notes

- LGM is designed for **4-view** inputs produced by a multi-view diffusion stage
  (e.g. `dylanebert/multi-view-diffusion`). Pass 4 images as separate calls to
  `GaussianSplattingTask::preprocess()` then concatenate before inference.
- The C++ `neuriplo-tasks` postprocessor accepts both `[G, 14]` (single-batch) and
  `[N, G, 14]` (batched) layouts.
- Rendering (rasterisation to a novel-view image) is outside the scope of this
  library. Use `diff_gaussian_rasterization` or a similar renderer to display
  the output splat.
