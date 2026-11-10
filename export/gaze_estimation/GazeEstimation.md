# Gaze Estimation Export

Export tools for appearance-based gaze estimation models to ONNX format.

## Supported Models

| Model | Export API | Input Size | Output Shape | neuriplo-tasks type |
|-------|------------|------------|--------------|-----------------|
| ETH-XGaze resnet18 (ptgaze) | timm + safetensors | 224×224 | `[B, 2]` pitch/yaw | `ethxgaze` |
| GazeTR-style | timm / torch | 224×224 | `[B, 3]` gaze vector | `gazetr` |

The ETH-XGaze exporter produces:
- Input tensor: `pixel_values` with shape `[batch, 3, 224, 224]` (ImageNet-normalized face crop)
- Output tensor: `gaze` with shape `[batch, 2]` (`[pitch, yaw]` radians, ETH-XGaze order)

`GazePostprocessor` also accepts `[B, 3]` xyz gaze vectors (GazeTR-style); see the [ETH-XGaze family guide](eth_xgaze/README.md) for the angle convention.

## Usage

For ETH-XGaze setup, ONNX export, and tensor contracts, see the [ETH-XGaze guide](eth_xgaze/README.md).

```bash
# Default export (downloads ptgaze resnet18 weights, verifies with ONNX Runtime)
python export/gaze_estimation/eth_xgaze/export.py --test

# Export and run on a face photo
python export/gaze_estimation/eth_xgaze/export.py --image face.jpg --test
```

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--weights-dir` | Where to download the safetensors | `weights` |
| `--output` | Output ONNX file path | `ethxgaze_resnet18.onnx` |
| `--opset` | ONNX opset version | 17 |
| `--image` | Optional face photo for `--test` | — |
| `--test` | Run ONNX Runtime inference test | Off |

## Requirements

```text
torch
timm
safetensors
huggingface_hub
onnx
onnxscript
numpy
opencv-python  # optional, for --image
onnxruntime    # optional, for --test
```

## Notes

- The model expects an ETH-XGaze-normalized face crop (face detection, landmarks, and head-pose normalization stay outside this library — see [docs/gaze_estimation.md](../../docs/gaze_estimation.md)). A naive resize runs end to end but its angles are only meaningful on normalized inputs.
- The C++ `neuriplo-tasks` gaze postprocessor decodes `[B, 2]` pitch/yaw and `[B, 3]` vectors behind one `GazeEstimation` result; verified against a real `hysts/ptgaze-eth-xgaze-resnet18` export.
