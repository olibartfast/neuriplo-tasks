# Gaze + Monocular 3D — Validation

| ID | Check | How | Status |
|----|-------|-----|--------|
| V-1 | Gaze `[B,2]` decodes pitch/yaw, direction matches ptgaze convention | `test_gaze_estimation` angle + convention tests, round-trip | pass |
| V-2 | Gaze `[B,3]` normalizes, derives pitch/yaw | vector test | pass |
| V-3 | Gaze batch + optional confidence tensor | batch/confidence tests | pass |
| V-4 | Gaze preprocess is 224 ImageNet NCHW float, matches cv2 | preprocess test + showcase parity check | pass |
| V-5 | Mono3D YOLO-pose tensor decodes boxes/corners with letterbox inverse, ground-first order | cuboid tests (square + padded frames, order) | pass |
| V-6 | Mono3D supports channels-first, rows+obj, end-to-end rows, kd=2/3; multi-class argmax; NMS dedup | layout tests incl. real `[B,23,N]` shape | pass |
| V-7 | Thresholds drop low-score rows; malformed returns empty | threshold/empty tests | pass |
| V-8 | Factory routes `ethxgaze`/`gazetr`/`gaze` and `urbanomni`/`monocular3d`/`mono3d`; `Result` index 12 preserves 0–11 | factory + result-type tests, `test_readme_model_types` | pass |
| V-9 | Full gate green | format + cppcheck + `cmake -DBUILD_TESTS=ON -DWERROR=ON` + `ctest` | pass |
| V-10 | Real-model showcase: ORT inference through both tasks | export scripts + C++ harness on real checkpoints | pass (see below) |

## Results

- 2026-11-10 (showcase): real-model end-to-end runs, all through `neuriplo-tasks` C++ code —
  - Mono3D: `urbanomnidetect_yolo11n-p2_640.pt` → `yolo export` → ONNX `output0 [1,23,34100]` (kd=2, nc=3, ground-first — matches decoder). Bangkok street photo (2400x1657): 6 car detections with cuboids via `Monocular3DDetectionTask`; top box `[269,750,226,212]` conf 0.777 vs Ultralytics predict `[268.5,753.0,499.6,969.7]` conf 0.765 (within ~5px; count differs 6 vs 5 from NMS IoU 0.45 vs 0.7). Preprocess parity vs cv2 INTER_LINEAR: max diff 0.0039 (1 gray level).
  - Gaze: `hysts/ptgaze-eth-xgaze-resnet18` safetensors → timm resnet18 → ONNX `gaze [B,2]` (strict weight load, no mismatch). Passport portrait: `[pitch, yaw] = [-0.608, -0.118]` rad → normalized direction via `GazeEstimationTask`. Values are pipeline-correct but only meaningful with ETH-XGaze-normalized face crops (naive resize used here — documented caveat). `[pitch, yaw]` order and ptgaze angle convention verified against `pytorch_mpiigaze_demo/common/gaze.py`.
  - Pre-existing gap found (out of scope, not changed): `YoloPreprocessor` letterboxes onto zeros while Ultralytics/`YoloDepthPreprocessor` use gray-114 — worth a detector-convention packet like the RT-DETR one.
- 2026-11-10: `cmake -DBUILD_TESTS=ON -DWERROR=ON` build clean; `ctest` 33/33 pass; `cppcheck` clean; `clang-format` clean (v21 proxy, v18 unavailable locally). Valgrind unavailable locally; new code is RAII-only — CI valgrind is the gate.
