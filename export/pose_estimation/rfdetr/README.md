# RF-DETR Keypoint Pose Estimation Model Export

Export an RF-DETR keypoint pose estimation model to ONNX format for use with C++ inference.

The keypoint models extend the RF-DETR detection architecture with keypoint prediction heads, producing detections, class labels, and keypoint coordinates in a single forward pass.

## Requirements

```bash
pip install rfdetr
```

## Quick Export

```bash
# Export the medium keypoint model
python export_keypoint.py --model_type medium

# Export with ONNX simplification
python export_keypoint.py --model_type medium --simplify

# Export with custom output directory and batch size
python export_keypoint.py --model_type large --output_dir ./models --batch_size 4
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model_type` | Model size: `nano`, `small`, `medium`, `large`, `xlarge` | `medium` |
| `--output_dir` | Path to save exported model | current directory |
| `--opset_version` | ONNX opset version | `17` |
| `--simplify` | Simplify ONNX model using onnxsim | off |
| `--batch_size` | Batch size for export | `1` |
| `--input_size` | Input image size | `640` |

## ONNX I/O Contract

| Role   | Name        | Dtype  | Shape                    | Description                                    |
|--------|-------------|--------|--------------------------|------------------------------------------------|
| Input  | `images`    | float  | `[B, 3, H, W]`          | NCHW preprocessed image                        |
| Output | `dets`      | float  | `[1, N, 4]`             | Bounding boxes `[cx, cy, w, h]` (normalized)   |
| Output | `labels`    | float  | `[1, N, C]`             | Class logits                                    |
| Output | `keypoints` | float  | `[1, N, K_max, 3]`      | Keypoints `[x, y, confidence]` (normalized)    |

## Preprocessing

Same as RF-DETR detection:

- Resize with aspect ratio preserved, pad to square (letterbox) with constant value
- BGR to RGB conversion
- Scale to `[0, 1]`
- ImageNet normalization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- NCHW format (float32)

## References

- [neuriplo-tasks](https://github.com/olibartfast/neuriplo-tasks)
- [rf-detr-cpp-inference](https://github.com/olibartfast/rf-detr-cpp-inference)
