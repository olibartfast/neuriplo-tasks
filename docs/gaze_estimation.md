# Gaze Estimation

Gaze estimation predicts the visual gaze direction of a person from eye or facial appearance.

> **Status:** task contract introduced on `feat/gaze-estimation`; model-specific preprocessing/postprocessing and `TaskFactory` registration are not implemented yet.

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

The initial reference implementation is planned around an **ETH-XGaze-style** appearance-based model contract, followed by **GazeTR** or another architecture once the common preprocessing/postprocessing boundary is stable.

Planned implementation steps:

1. gaze-specific face/eye preprocessing and normalization;
2. model output decoding;
3. pitch/yaw to normalized 3D direction conversion where required;
4. task implementation returning `GazeEstimation`;
5. tests for angle/vector conventions and tensor shapes;
6. `TaskFactory` aliases only after the corresponding model implementation is present and tested.

## TaskFactory status

There is currently **no gaze-estimation model alias registered in `TaskFactory`**. The task type and result contract are intentionally introduced first so that model wrappers share one stable, framework-agnostic output representation.
