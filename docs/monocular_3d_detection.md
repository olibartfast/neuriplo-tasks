# Monocular 3D Detection

Monocular 3D object detection estimates an object's 3D extent from a single image, represented here as a projected 3D cuboid (eight 2D corners) plus a 2D box. It is intentionally separate from `PoseEstimation`, which is 2D human skeletal keypoints — see [pose task scope](pose_estimation_scope.md).

The reference models are [UrbanOmniDetect](https://github.com/mkturkcan/urbanomnidetect) (v1, CVPR 2026 DriveX workshop) and [UrbanOmniDetect-2](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2) (YOLO26 hybrid): calibration-free detectors that emit the eight projected cuboid corners as a YOLO-pose-style keypoint head, with no camera intrinsics at inference time.

## Result contract

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

Released checkpoints (UrbanOmniDetect v1 and UrbanOmniDetect-2) both emit ground-first corners — verified empirically on the released weights and in the upstream runtime (the `GroundIndexResolver` there auto-detects the same; the paper text states the opposite order). The postprocessor preserves model order by default and offers an opt-in remap for top-first checkpoints.

## Task contract

- Task: `Monocular3DDetectionTask` (`TaskType::Monocular3DDetection`, `Result::Monocular3DDetection`).
- Input: road-scene image, YOLO letterbox to the checkpoint size (640/1280/1920), `[0,1]`, BGR→RGB, NCHW float32.
- Output: one YOLO-pose-style tensor — v1 `[B, 4+nc+24, N]` channels-first, v1 rows `[B, N, 5+nc+24]` with objectness, or v2 `[B, N, 30]` end-to-end rows. Coordinates are input-pixel space; letterbox padding is inverted for boxes and corners.
- Postprocessing drops rows below `TaskConfig::confidence_threshold`, then runs class-agnostic NMS (`TaskConfig::nms_threshold`).

`TaskFactory` routes `urbanomni*` (incl. `urbanomnidetect`, `urbanomnidetect2`), `monocular3d*`, and `mono3d` here; names containing `v2`/`26` select the v2 layout. See [export/monocular_3d_detection/urbanomnidetect/README.md](../export/monocular_3d_detection/urbanomnidetect/README.md).
