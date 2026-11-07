# Pose Estimation Task Scope

The `PoseEstimation` task in `neuriplo-tasks` is specifically for **2D human pose estimation**.

It represents human anatomical keypoints projected in image coordinates, for example COCO-style joints such as shoulders, elbows, wrists, hips, knees, and ankles. Detection-based pose models may additionally provide a 2D bounding box around the person.

Supported model families in this category include YOLO pose, ViTPose, EdgeCrafter pose, and RF-DETR keypoint pose.

## What this task does not represent

`PoseEstimation` must not be used as a generic pose abstraction for other geometric tasks. In particular, it does **not** represent:

- 3D human pose estimation, where anatomical joints are recovered in 3D coordinates;
- 6-DoF object pose estimation, where a rigid object's translation and rotation relative to a camera or world frame are estimated;
- monocular 3D object detection, where an object's 3D extent, cuboid, dimensions, location, orientation, or projected 3D box geometry are estimated.

These tasks have different semantics and should use dedicated task/result types even when a model internally encodes its output using keypoints.

## Why the distinction matters

The presence of `keypoints` in a model output does not by itself make that model a human pose-estimation model. Keypoints can also be used as an intermediate representation for rigid-object geometry.

For example, UrbanOmniDetect predicts eight projected corners of a 3D cuboid for road users. Those points describe object-box geometry rather than anatomical joints, so UrbanOmniDetect belongs under the monocular 3D object-detection task rather than `PoseEstimation`.

This distinction should be preserved in `TaskType`, `Result`, `TaskFactory`, documentation, and future model integrations.