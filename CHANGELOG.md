# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.6.1] - 2026-07-16

### Fixed
- EdgeCrafter detection, segmentation, and pose export documentation links now
  appear inside their corresponding supported-model subsections.

## [0.6.0] - 2026-07-10

### Added
- Backend-neutral `core/vision` image, geometry, and operation APIs.
- Optional `vision-stb` image I/O and `vision-opencv` interoperability targets.

### Changed
- Core task APIs now use native `vision::Image`, `vision::Size`, and `vision::PixelType` contracts.
- OpenCV is no longer a required or transitive dependency of `neuriplo-tasks`.

### Removed
- Completed implementation roadmaps and retired planning documents; batch documentation now stands on its own as current consumer guidance.


## [0.5.0] - 2026-06-24

### Added
- RF-DETR keypoint pose estimation (`RfDetrPose` task family). TaskFactory now
  routes `rfdetrpose`, `rfdetr-pose`, `rfdetrkeypoint`, `rfdetr-keypoint`,
  `rfdetrkpt`, and `rfdetr-kpt` to single-stage pose estimation returning the
  bounding box plus 17 COCO keypoints with per-keypoint visibility and a 2x2
  pixel covariance (decoded from the Cholesky factor via the `log_l11`, `l21`,
  and `log_l22` ONNX channels).
- `Keypoint` extended with `visibility` and `covariance` fields, backed by the
  new `RfDetrPosePostprocessor`.
- Batch (shape[0]) support across 10 postprocessors and multi-image stacking in
  4 preprocess strategies; 16 model families are now batch-ready.
- Windows (MSVC + vcpkg) build and test support, including
  `target_link_whole_archive` wiring for post-merge tests.
- `export/pose_estimation/rfdetr/export_keypoint.py` keypoint exporter.

### Fixed
- Pose uncertainty reduction is clamped so keypoint scores can no longer exceed 1.0.
- RF-DETR keypoint export made compatible with the rfdetr export API.
- Windows CI: OpenCV DLL directory added to PATH; the windows-2022 runner is
  pinned and uses pre-built OpenCV binaries.

### Changed
- CI gained a PR branch-policy workflow and link linting.

## [0.4.1] - 2026-06-14

### Fixed
- YOLOv10/YOLO26 NMS-free detection now applies class-aware NMS so duplicate TFLite end-to-end detections collapse without suppressing overlapping detections from different classes.
- Object detection tasks now handle YOLO models with a generic `size` input (no explicit anchor count) without crashing.

## [0.4.0] - 2026-06-07

### Changed
- **Repository and API rename:** `vision-core` → `neuriplo-tasks`.
  C++ namespace `vision_core` → `neuriplo_tasks`; public include root
  `vision-core/...` → `neuriplo/tasks/...`; CMake project/target/package
  `vision-core` → `neuriplo-tasks`; version env `VISION_CORE_VERSION` →
  `NEURIPLO_TASKS_VERSION`. Task contracts, result schema, tensor shapes, and
  model type strings are unchanged.
- Documentation updated to reference renamed siblings (`neuriplo-infer`,
  `neuriplo-track`).

### Added
- Batch processing utilities: `batch_types.hpp`, `batchPreprocess`, `batchPostprocess`
- Domain batch adoption for classification (`[N,C]` logits), YOLO standard detection, depth, and ViTPose
- `tests/test_batch_integration.cpp` — end-to-end `batchPreprocess` + `batchPostprocess` for classification and YOLO detection
- `docs/batch_processing.md` — consumer migration guide (engine vs library, N=2 worked examples)
- `task_pipeline.hpp` — composable `Result` pipeline stages for detection → pose / segmentation workflows
- CI Valgrind job for Debug test binaries with strict definite/indirect leak and memory-error checks

### Changed
- TaskFactory registry now uses named `TaskDescriptor` entries grouped by task family, with additional boundary tests for routing precedence.

## [0.3.2] - 2026-05-28

### Fixed
- YOLOv10/YOLO26 NMS-free postprocessor now scales normalized model output
  coordinates to pixel space before the inverse letterbox transformation.
  Without this, detections land at negative coordinates and render off-screen.

## [0.3.1] - 2026-05-21

### Added
- Reverse README ↔ TaskFactory contract test: every model-key string literal in `task_factory.cpp` must be documented in the README model-type block (the existing test only checked the forward direction). Plus explicit factory routing tests for hyphenated, long-form, and image-understanding aliases

### Changed
- TaskFactory: removed an unreachable `normalized == "lgm-mini"` comparison in the Gaussian Splatting branch — `normalizeModelType` strips hyphens before the check, so the existing `lgmmini` branch already covers it

### Fixed
- README Features list had drifted from the registered TaskFactory model types: restored RF-DETR and YOLOv4 under Object Detection, and YOLOv10-seg / YOLO26-seg under Instance Segmentation
- README "Supported Model Types" table now documents previously undocumented routable aliases (`rtdetrultralytics`, `gemma`/`llama`/`llamacpp`) and the `resnet*` / `*tensorflow*` classification matching rules
- `test_readme_model_types` resolves the README and `task_factory.cpp` paths relative to the `tests/` directory (`CMAKE_CURRENT_SOURCE_DIR`) instead of `CMAKE_SOURCE_DIR`, so the contract test works when neuriplo-tasks is built as a FetchContent sub-project of another repo

## [0.3.0] - 2026-05-21

### Added
- Image Understanding task with VLM multimodal preprocessing (`ImageUnderstandingTask`, `gemma4` / `imageunderstanding` aliases)
- Gaussian Splatting task: LGM, LGM-mini, and GRM support
- Grounding DINO open-vocabulary detection
- Agent tooling kit: Claude Code hooks for formatting, CI gating (local), and CI guardian skill
- LGM Gaussian Splatting ONNX export script and model validation test
- Image Understanding export guide and updated export README
- GTest contract locking README ↔ TaskFactory alias consistency

### Changed
- CI lint workflow skips on docs-only pushes (`**.md`, `export/**`, `docs/**`)

### Fixed
- `InputDimensionError` restored for invalid spatial shapes; `memcpy` `-DWERROR` fix
- Gaussian Splatting task: parse NHWC input shapes before choosing resize dimensions
- LGM postprocessor: preserve all batches in [N,G,14] Gaussian outputs

## [0.2.0] - 2026-03-31

### Added
- OWLv2 open-vocabulary detection scaffold and CLIP tokenizer
- OWLv2 ONNX export script

### Fixed
- Use int64 for tokenizer byte buffer to match ONNX model expectations

## [0.1.0] - 2026-03-02

### Added
- Unified task design with `TaskInterface` and `TaskFactory`
- `TaskConfig` struct for passing configuration parameters to tasks
- Object detection: YOLO (v4/v5/v7/v8/v10/v11), RT-DETR, RF-DETR
- Image classification: TorchVision, TensorFlow, ViT
- Video classification: VideoMAE, ViViT, TimeSformer
- Instance segmentation: YOLOSeg, YOLOv10Seg, RF-DETR-Seg
- Pose estimation: ViTPose, YOLO Pose
- Optical flow: RAFT
- Depth estimation: Depth Anything V2
- YOLO pose export script and documentation
- GTest-based unit test suite
- clang-format and clang-tidy configuration
- CI workflow with lint, static analysis, build, and test jobs

[Unreleased]: https://github.com/olibartfast/neuriplo-tasks/compare/v0.6.1...HEAD
[0.6.1]: https://github.com/olibartfast/neuriplo-tasks/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/olibartfast/neuriplo-tasks/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/olibartfast/neuriplo-tasks/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/olibartfast/neuriplo-tasks/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/olibartfast/neuriplo-tasks/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/olibartfast/neuriplo-tasks/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/olibartfast/neuriplo-tasks/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/olibartfast/neuriplo-tasks/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/olibartfast/neuriplo-tasks/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/olibartfast/neuriplo-tasks/releases/tag/v0.1.0
