# Gaze Estimation Export

Export tools for appearance-based gaze estimation models to ONNX format.

Gaze estimation predicts the visual gaze direction of a person from eye or facial appearance. This task is **distinct from 2D human pose estimation**: in `neuriplo-tasks`, **Pose Estimation means 2D human skeletal keypoint estimation**, while gaze describes **where a person's eyes are directed**. It should not be interpreted as 3D object pose, 6-DoF pose, head pose, or human skeletal pose.

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

## Task contract

The common result type is `GazeEstimation` (`include/neuriplo/tasks/core/result_types.hpp`):

```cpp
GazeEstimation {
    pitch,                  // vertical gaze angle in radians, when provided or derivable
    yaw,                    // horizontal gaze angle in radians, when provided or derivable
    direction = {x, y, z},  // normalized 3D gaze direction
    confidence              // optional model confidence in [0,1]
}
```

The normalized 3D direction is the task-level representation that makes models with different native outputs interoperable. Models that directly regress pitch/yaw convert those angles during postprocessing; models that directly regress a gaze vector populate `direction` natively.

`TaskFactory` routes `gaze`, `ethxgaze*` (e.g. `ethxgaze`, `eth-xgaze`), `gazetr*` (e.g. `gazetr`, `gaze-tr`), and any string containing `gaze` to `GazeEstimationTask` (`TaskType::GazeEstimation`, `Result::GazeEstimation`).

## Input scope

The reference implementation targets appearance-based gaze estimation from a normalized face or eye crop. Face detection, facial landmarks, head-pose estimation, temporal filtering, fixation classification, blink detection, and calibration are related but separate concerns and are not folded into this task contract.

A higher-level application may compose these stages, for example:

```text
face detection -> face normalization -> gaze estimation -> temporal filtering
```

For XR/eye-tracking applications, a downstream system may additionally combine the predicted direction with an eye/camera origin to form a 3D gaze ray. The generic task contract intentionally does not assume a headset-specific coordinate system or calibration model.

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

- The model expects an ETH-XGaze-normalized face crop (face detection, landmarks, and head-pose normalization stay outside this library). A naive resize runs end to end but its angles are only meaningful on normalized inputs.
- The C++ `neuriplo-tasks` gaze postprocessor decodes `[B, 2]` pitch/yaw and `[B, 3]` vectors behind one `GazeEstimation` result; verified against a real `hysts/ptgaze-eth-xgaze-resnet18` export.
