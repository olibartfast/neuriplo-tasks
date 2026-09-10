# Gaze Estimation

Gaze estimation predicts the visual gaze direction of a person from eye or facial appearance.

This task is **distinct from 2D human pose estimation**. In `neuriplo-tasks`, pose estimation refers to human skeletal keypoints such as the COCO body keypoints; gaze estimation instead describes where the eyes are directed.

## Canonical result

`GazeEstimation` exposes:

- `pitch`: vertical gaze angle in radians, when provided or derivable by the model.
- `yaw`: horizontal gaze angle in radians, when provided or derivable by the model.
- `direction`: normalized 3D gaze direction `[x, y, z]`.
- `confidence`: optional model confidence in `[0, 1]`.

The normalized 3D direction is the task-level representation intended to make models with different native outputs interoperable. Models that directly regress pitch/yaw can convert those angles during postprocessing; models that directly regress a gaze vector can populate `direction` natively.

## Input scope

The first implementation targets appearance-based gaze estimation from a normalized face or eye crop. Face detection, facial landmarks, head-pose estimation, temporal filtering, fixation classification, blink detection, and calibration are related but separate concerns and should not be silently folded into this task contract.

A higher-level application may compose these stages, for example:

```text
face detection -> face normalization -> gaze estimation -> temporal filtering
```

## Initial model plan

The initial reference implementation is planned around an ETH-XGaze-style model contract, followed by GazeTR or another architecture once the common preprocessing/postprocessing boundary is stable.

TaskFactory aliases should only be added when the corresponding model implementation is present and tested.
