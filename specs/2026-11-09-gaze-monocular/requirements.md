# Gaze + Monocular 3D — Requirements

## Goal

Implement the first runnable models for the two contract-only tasks:

- `GazeEstimation` (`Result` + `TaskType::GazeEstimation` already exist; no task/factory).
- `Monocular3DDetection` (standalone header only; not in `Result`/`TaskType`/factory).

## Decisions (with user, 2026-11-09)

- Gaze decoder supports both `[B,2]` pitch/yaw (radians) and `[B,3]` gaze-vector outputs. `[B,2]` order is `[pitch, yaw]`; direction follows the ETH-XGaze / OpenGaze normalized camera (cf. ptgaze): `x=-cos(pitch)sin(yaw)`, `y=-sin(pitch)`, `z=-cos(pitch)cos(yaw)`; inverse `pitch=asin(-y)`, `yaw=atan2(-x,-z)`. Optional second tensor `[B,1]`/`[B]` is confidence, else `0`.
- Mono3D v1 uses a DETR-style multi-tensor contract (not single YOLO-style): boxes `[B,Q,4]` cxcywh normalized, labels `[B,Q,C]` logits or `[B,Q]` scores, corners `[B,Q,16]` or `[B,Q,8,2]` normalized, geom-conf `[B,Q]`/`[B,Q,1]`. Letterbox inverse (YOLO preprocessor) maps boxes+corners to the original frame.
- Clean worktree from `develop` as `feature/gaze-monocular-tasks`.
- Full scope: code + tests + factory + README block + `export/` guides + docs status, CI-green.

## Scope

In:

- `GazeEstimationTask` (`ethxgaze` / `gazetr` / `gaze`) with ImageNet 224 preprocessing, dual-layout postprocessor, angle/vector conversion helpers.
- `Monocular3DDetectionTask` (`urbanomni` / `urbanomnidetect` / `urbanomnidetect2` / `monocular3d` / `mono3d`) with YOLO letterbox preprocessing and a YOLO-pose-style single-tensor cuboid postprocessor (v1 channels-first/rows, v2 YOLO26 end-to-end).
- `Monocular3DDetection` appended to `Result` variant and `TaskType` (index 12, no renumbering).
- Factory routing (both matchers), CMake wiring, unit tests, README block, export guides, docs updates.

Out:

- Real ONNX weights/exports (guides only, no binaries).
- GazeTR/L2CS bin heads, temporal filtering, face detection, calibration, XR ray origin.
- 6-DoF pose, 3D human pose, BEV occupancy — cuboid corners only (BEV/homography stays downstream).
- Runtime/plugin registry changes.

## Contracts

### Gaze

- Input: face/eye crop RGB float32 NCHW `[B,3,224,224]` (from `ModelInfo`), BGR→RGB, resize, `[0,1]`, ImageNet mean/std. UInt8 raw supported via `applyImageInputType`.
- Output tensor 0: `[B,2]`/`[2]` pitch,yaw rad or `[B,3]`/`[3]` xyz (any `TensorElement` scalar type via `tensorElementToFloat`).
- Output tensor 1 (optional): confidence.
- Result: one `GazeEstimation` per batch row.

### Mono3D (revised 2026-11-09: YOLO-pose-style, matching real UrbanOmniDetect exports)

- Input: road-scene image, YOLO letterbox to the checkpoint size (640/1280/1920), `[0,1]`, BGR→RGB, NCHW float32.
- Output: one `output0` tensor — v1 `[B,4+nc+24,N]` channels-first or `[B,N,5+nc+24]` rows with objectness; v2 `[B,N,30]` end-to-end rows. Box + 8 corner triples in input-pixel space.
- Corners preserved in model order; released checkpoints emit ground-first (0-3 ground, verified empirically on the released weights and in the upstream runtime — the paper text states the opposite). `geometry_confidence` is the mean corner confidence (v1 exports carry no per-corner confidence, so it equals the detection score).
- Result: `Monocular3DDetection{bbox, class_id, class_confidence, projected_corners[8], geometry_confidence}` in original-frame pixels; threshold on `confidence_threshold` then class-agnostic NMS.

## Validation

See `validation.md`.
