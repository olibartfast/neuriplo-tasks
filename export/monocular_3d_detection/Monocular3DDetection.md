# Monocular 3D Detection Export

Export tools for UrbanOmniDetect-style monocular 3D cuboid models to ONNX format.

## Supported Models

| Model | Export API | Input Size | Output Shape | neuriplo-tasks type |
|-------|------------|------------|--------------|-----------------|
| UrbanOmniDetect v1 (YOLOv8/v9/v11/v12) | Ultralytics `YOLO` | 640/1280/1920 | `[B, 4+nc+8kd, N]` pose-style | `urbanomnidetect` |
| UrbanOmniDetect-2 (YOLO26 hybrid) | Ultralytics `YOLO` | 640 | `[B, N, 30]` end-to-end rows | `urbanomnidetect2` |

The v1 exporter produces (verified on `urbanomnidetect_yolo11n-p2_640`, nc=3, kd=2):
- Input tensor: `images` with shape `[1, 3, H, W]` (YOLO letterbox, `[0, 1]`, RGB)
- Output tensor: `output0` with shape `[1, 23, 34100]` — `cx, cy, w, h` (input pixels), 3 class scores, 8× `(x, y)` cuboid corners in input pixels

`Monocular3DPostprocessor` also accepts kd=3 corner triples and end-to-end rows; see the [UrbanOmniDetect family guide](urbanomnidetect/README.md) for the full layout table and the corner convention.

## Usage

For checkpoint download, ONNX export, and tensor contracts, see the [UrbanOmniDetect guide](urbanomnidetect/README.md).

```bash
# Default export (downloads the v1 nano checkpoint, verifies with ONNX Runtime)
python export/monocular_3d_detection/urbanomnidetect/export.py --test

# Export and run on a street photo
python export/monocular_3d_detection/urbanomnidetect/export.py --image street.jpg --test
```

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Checkpoint name or local `.pt` path | `urbanomnidetect_yolo11n-p2_640` |
| `--weights-dir` | Where to download the checkpoint | `weights` |
| `--imgsz` | Export image size (match the checkpoint) | 640 |
| `--image` | Optional road photo for `--test` | — |
| `--conf` | Detection threshold for `--test` | 0.25 |
| `--test` | Run ONNX Runtime inference test | Off |

## Requirements

```text
ultralytics
huggingface_hub
onnx
numpy
opencv-python  # optional, for --image
onnxruntime    # optional, for --test
```

## Notes

- Match `imgsz` to the checkpoint's training size and to `ModelInfo::input_shapes`. v1 covers YOLOv8/v9/v11/v12 (`n/s/m/l/x`) at 640/1280/1920; UrbanOmniDetect-2 checkpoints live on the [model hub](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2).
- Checkpoints are distributed under AGPL-3.0; that governs the weights, not this MIT library, but be aware before redistributing them. Do not commit downloaded checkpoints or exported models.
- The C++ `neuriplo-tasks` cuboid postprocessor decodes channels-first/rows/end-to-end rows behind one `Monocular3DDetection` result; verified against the real v1 export above.
