# Gaze Estimation

Gaze estimation predicts the visual gaze direction of a person from eye or facial appearance.

> **Status:** implemented. `GazeEstimationTask` + `GazePostprocessor` decode ETH-XGaze-style `[B,2]` pitch/yaw and GazeTR-style `[B,3]` vectors; `TaskFactory` routes `gaze`, `ethxgaze*`, `gazetr*` (and any string containing `gaze`).

## Relation to pose estimation

This task is **distinct from 2D human pose estimation**. In `neuriplo-tasks`, **Pose Estimation means 2D human skeletal keypoint estimation** (for example COCO body keypoints produced by YOLO Pose, ViTPose, RF-DETR keypoint pose, or EdgeCrafter pose).

Gaze estimation instead describes **where a person's eyes are directed**. It should not be interpreted as 3D object pose, 6-DoF pose, head pose, or human skeletal pose.

## Public task contract

The common result type is declared in:

```text
include/neuriplo/tasks/core/result_types.hpp
```

`GazeEstimation` exposes:

- `pitch`: vertical gaze angle in radians, when provided or derivable by the model.
- `yaw`: horizontal gaze angle in radians, when provided or derivable by the model.
- `direction`: normalized 3D gaze direction `[x, y, z]`.
- `confidence`: optional model confidence in `[0, 1]`.

The normalized 3D direction is the task-level representation intended to make models with different native outputs interoperable. Models that directly regress pitch/yaw can convert those angles during postprocessing; models that directly regress a gaze vector can populate `direction` natively.

At task level the intended output is therefore conceptually:

```cpp
GazeEstimation {
    pitch,
    yaw,
    direction = {x, y, z},
    confidence
}
```

## Input scope

The first implementation targets appearance-based gaze estimation from a normalized face or eye crop. Face detection, facial landmarks, head-pose estimation, temporal filtering, fixation classification, blink detection, and calibration are related but separate concerns and should not be silently folded into this task contract.

A higher-level application may compose these stages, for example:

```text
face detection -> face normalization -> gaze estimation -> temporal filtering
```

For XR/eye-tracking applications, a downstream system may additionally combine the predicted direction with an eye/camera origin to form a 3D gaze ray. The generic task contract intentionally does not assume a headset-specific coordinate system or calibration model.

## Initial model plan

The reference implementation covers an **ETH-XGaze-style** appearance-based model contract and a **GazeTR-style** vector contract behind one decoder (`GazePostprocessor`):

- input: face/eye crop RGB NCHW `[B,3,224,224]`, `[0,1]` + ImageNet mean/std, BGR→RGB (`GazePreprocessor`);
- output 0: `[B,2]`/`[2]` pitch/yaw radians `[pitch, yaw]` or `[B,3]`/`[3]` xyz vector;
- output 1 (optional): confidence;
- direction convention follows the ETH-XGaze / OpenGaze normalized camera (cf. ptgaze): `direction=[-cos(pitch)sin(yaw), -sin(pitch), -cos(pitch)cos(yaw)]`, `pitch=asin(-y)`, `yaw=atan2(-x,-z)`, forward `-Z`. A horizontal flip of the input negates yaw and leaves pitch unchanged.

## TaskFactory status

`TaskFactory` routes `gaze`, `ethxgaze*` (e.g. `ethxgaze`, `eth-xgaze`), `gazetr*` (e.g. `gazetr`, `gaze-tr`), and any string containing `gaze` to `GazeEstimationTask` (`TaskType::GazeEstimation`, `Result::GazeEstimation`). See [export/gaze_estimation/eth_xgaze/README.md](../export/gaze_estimation/eth_xgaze/README.md).
