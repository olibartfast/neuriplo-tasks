# RF-DETR Keypoints Pose Estimation — Implementation Plan

Reference: https://github.com/olibartfast/rf-detr-cpp-inference/tree/master/src
Export: https://github.com/olibartfast/rf-detr-cpp-inference/blob/master/deploy/export_keypoint.py

## ONNX Model Output Contract

| Tensor | Shape | Description |
|--------|-------|-------------|
| `dets` | `[1, N, 4]` | bbox centers (cx, cy, w, h) normalized to input size |
| `labels` | `[1, N, C]` | class logits (offset +1; sigmoid'd, best class selected) |
| `keypoints` | `[1, N, C*K_max, 8]` | per-keypoint 8-channel tensor |

8 channels per keypoint: x, y, findability_logit, visibility_logit, log_l11, l21, log_l22, class_boost

Per-keypoint decoding:
- x, y → scale by orig_w, orig_h
- findability = sigmoid(logit) → maps to Keypoint::confidence
- visibility = sigmoid(logit) → new field
- Cholesky factor L → precision P = L·L^T → invert → pixel covariance Σ
- Uncertainty-weighted score fusion: final_score *= exp(-α·log(avg_trace))

## Steps

### 1. Extend `Keypoint` struct
File: `include/neuriplo/tasks/core/result_types.hpp`
Add `visibility` (float) and `covariance[4]` (float) fields, default-zero for backward compat.

### 2. Create `RfDetrPosePostprocessor` header
New file: `include/neuriplo/tasks/pose_estimation/rfdetr_pose_postprocessor.hpp`
Class inherits `PosePostprocessor`, holds `confidence_threshold_`, `keypoint_uncertainty_alpha_`, `keypoint_counts_`.

### 3. Create `RfDetrPosePostprocessor` implementation
New file: `src/pose_estimation/rfdetr_pose_postprocessor.cpp`
Core logic ported from rf-detr-cpp-inference: validate 3 tensors, decode bbox, decode keypoints (Cholesky→precision→covariance), uncertainty-weighted score fusion.

### 4. Add `RFDETRPOSE` to `PoseEstimationTask` ModelType
File: `src/pose_estimation/pose_estimation_task.cpp`
Internal enum + detectModelType() for rfdetr+pose strings.

### 5. Wire preprocessor for RF-DETR pose
File: `src/pose_estimation/pose_estimation_task.cpp`
Reuse existing `RfDetrPreprocessor` (same ImageNet norm, square resize, BGR→RGB).

### 6. Wire postprocessor creation
File: `src/pose_estimation/pose_estimation_task.cpp`
Create `RfDetrPosePostprocessor` in factory method.

### 7. Update TaskFactory routing
File: `src/core/task_factory.cpp`
Insert rfdetr+pose rule BEFORE generic rfdetr detection rule. Match strings: `rfdetrpose`, `rfdetrkeypoint`, or any `rfdetr`+`pose` combo.

### 8. Write tests
File: `tests/test_pose_estimation.cpp`
- DetectsSinglePerson test
- CovarianceRoundTrips test
- FiltersBelowThreshold test
- FactoryRoutesRfDetrPose test

### 9. Add export script
New files: `export/pose_estimation/rfdetr/export_keypoint.py`, `export/pose_estimation/rfdetr/README.md`

### 10. Update `export/README.md`
Add rfdetr pose directory to tree.

### 11. Update `README.md`
Features bullet + TASKFACTORY_MODEL_LIST entries.

### 12. Build & verify
cmake configure, build, ctest, clang-format check.

## Design decisions
- Extend `Keypoint` (not new type): backward-compatible, zero-filled new fields
- Reuse `RfDetrPreprocessor`: same preprocessing as RF-DETR detection
- Route `rfdetr+pose` before generic `rfdetr`: follows YOLO-pose pattern
- Covariance always computed: consumers ignore if unneeded
- `keypoint_counts_` hardcoded to COCO person: sufficient for v1

## Parallelization plan (3 waves)

### Wave 1 (no interdependencies — parallel)
| Subagent | Steps | Files |
|----------|-------|-------|
| A | 3. RfDetrPosePostprocessor impl | `src/pose_estimation/rfdetr_pose_postprocessor.cpp` (new) |
| B | 4–7. ModelType + wiring (preprocessor, postprocessor, factory) | `pose_estimation_task.cpp`, `task_factory.cpp` |
| C | 9–10. Export script + export README | `export/pose_estimation/rfdetr/` (new) |
| D | 11. README.md update | `README.md` |

### Wave 2 (depends on A+B)
| | Steps | Files |
|--|-------|-------|
| | 8. Tests | `tests/test_pose_estimation.cpp` |

### Wave 3 (depends on everything)
| | Steps | Files |
|--|-------|-------|
| | 12. Build & verify | cmake + ctest + clang-format |
