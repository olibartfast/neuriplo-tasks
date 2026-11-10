# Gaze + Monocular 3D — Plan

1. Spec packet (`requirements.md`, `plan.md`, `validation.md` skeleton) + roadmap `In flight` entry.
2. Gaze domain:
   - `include/neuriplo/tasks/gaze_estimation/{gaze_estimation_preprocessor,gaze_estimation_postprocessor,gaze_estimation_task}.hpp`
   - `src/gaze_estimation/{gaze_estimation_preprocessor,gaze_estimation_postprocessor,gaze_estimation_task}.cpp`
   - Angle/vector conversion helpers with round-trip invariant.
3. Mono3D domain (YOLO-pose-style, revised to match real UrbanOmniDetect):
   - `include/neuriplo/tasks/monocular_3d_detection/{monocular_3d_detection_task,monocular_3d_detection_postprocessor}.hpp`
   - `src/monocular_3d_detection/{monocular_3d_detection_task,monocular_3d_detection_postprocessor}.cpp` reusing `YoloPreprocessor` letterbox + inverse; v1/v2 corner-order handling; NMS.
4. Core wiring:
   - `result_types.hpp`: include cuboid types, append `Monocular3DDetection` to `Result` + `TaskType` (index 12).
   - `task_factory.cpp`: `TaskFamily::{GazeEstimation,Monocular3DDetection}`, matchers (`gaze`/`ethxgaze`/`gazetr`; `urbanomni`/`monocular3d`/`mono3d`), creators.
   - `CMakeLists.txt`: sources + headers.
5. Tests:
   - `tests/test_gaze_estimation.cpp` (angles, vector, batch, confidence, task, factory).
   - `tests/test_monocular_3d_detection.cpp` (corners layouts, scores layouts, letterbox scaling, thresholds, task, factory).
   - Extend `test_result_types.cpp` (index 12, name map).
   - `tests/CMakeLists.txt` wiring.
6. Docs:
   - `README.md` Features bullets + `TASKFACTORY_MODEL_LIST` sections.
   - `export/gaze_estimation/README.md`, `export/monocular_3d_detection/README.md` + `export/README.md` tree/links.
   - `docs/gaze_estimation.md` status flip; new `docs/monocular_3d_detection.md`.
7. Gate: clang-format, cppcheck, `cmake -DBUILD_TESTS=ON -DWERROR=ON`, `ctest`, clang-tidy, valgrind spot-check.
