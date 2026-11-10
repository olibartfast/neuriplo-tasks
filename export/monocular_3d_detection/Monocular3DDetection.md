# Monocular 3D Detection Export

Export tools for UrbanOmniDetect-style monocular 3D cuboid models to ONNX format.

Monocular 3D object detection estimates an object's 3D extent from a single image, represented here as a projected 3D cuboid (eight 2D corners) plus a 2D box. It is intentionally separate from `PoseEstimation`, which is 2D human skeletal keypoints — see [pose task scope](../../docs/pose_estimation_scope.md). The eight points are rigid-object geometry (not human joints), so they live in `Monocular3DDetection`, not `PoseEstimation`.

The reference models are [UrbanOmniDetect](https://github.com/mkturkcan/urbanomnidetect) (v1, CVPR 2026 DriveX workshop) and [UrbanOmniDetect-2](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2) (YOLO26 hybrid): calibration-free detectors that emit the eight projected cuboid corners as a YOLO-pose-style keypoint head, with no camera intrinsics at inference time.

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

## Task contract

- Task: `Monocular3DDetectionTask` (`TaskType::Monocular3DDetection`, `Result::Monocular3DDetection`).
- Input: road-scene image, YOLO letterbox to the checkpoint size (640/1280/1920), `[0, 1]`, BGR→RGB, NCHW float32.
- Output: one YOLO-pose-style tensor — v1 `[B, 4+nc+8kd, N]` channels-first, v1 rows `[B, N, 5+nc+8kd]` with objectness, or v2 `[B, N, 30]` end-to-end rows. Coordinates are input-pixel space; letterbox padding is inverted for boxes and corners.
- Postprocessing drops rows below `TaskConfig::confidence_threshold`, then runs class-agnostic NMS (`TaskConfig::nms_threshold`).

The common result type is `Monocular3DDetection` (`include/neuriplo/tasks/monocular_3d_detection/monocular_3d_detection_types.hpp`):

```cpp
Monocular3DDetection {
    bbox,                  // 2D box in original-frame pixels
    class_id,
    class_confidence,
    projected_corners[8],  // 2D cuboid corners in original-frame pixels
    geometry_confidence    // mean corner confidence
}
```

Canonical corner convention (ground-first):

- corners 0..3: ground-contact corners
- corners 4..7: roof corners
- corner `i+4` lies above ground corner `i`
- corners 0 and 3 belong to the front face

Released checkpoints (UrbanOmniDetect v1 and UrbanOmniDetect-2) both emit ground-first corners — verified empirically on the released weights and in the upstream runtime (the paper text states the opposite order). The postprocessor preserves model order by default and offers an opt-in remap for top-first checkpoints.

`TaskFactory` routes `urbanomni*` (incl. `urbanomnidetect`, `urbanomnidetect2`), `monocular3d*`, and `mono3d` here; names containing `v2`/`26` select the v2 layout.

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
