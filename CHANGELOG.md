# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Gaussian Splatting task: LGM, LGM-mini, and GRM support
- Grounding DINO open-vocabulary detection
- Agent tooling kit: Claude Code hooks for formatting, CI gating (local), and CI guardian skill
- LGM Gaussian Splatting ONNX export script and model validation test

### Fixed
- Gaussian Splatting task: Parse NHWC input shapes before choosing resize dimensions (P1)
- LGM postprocessor: Preserve all batches in [N,G,14] Gaussian outputs (P1)
- AGENTS.md: Documentation for Open-Vocabulary Detection and Gaussian Splatting tasks

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

[Unreleased]: https://github.com/olibartfast/vision-core/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/olibartfast/vision-core/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/olibartfast/vision-core/releases/tag/v0.1.0
