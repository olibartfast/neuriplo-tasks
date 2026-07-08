# Centralize Computer Vision Primitives And Remove OpenCV Coupling

**Status:** Implementation in progress (Phases 0-5 complete except namespace cleanup and optional numeric comparison tests)
**Target version:** `v0.6.0`
**Branch:** `feat/remove-opencv-stb`
**Core correction:** do not replace scattered OpenCV usage with scattered stb usage. First centralize the computer-vision substrate, then put OpenCV, stb, or future libraries behind adapters.

## Design Goal

`neuriplo-tasks` should expose and use its own small vision API:

- public task contracts use `neuriplo_tasks::vision` types;
- task code never includes OpenCV, stb, SDL, FFmpeg, or another backend directly;
- third-party libraries live only in adapter and backend implementation files;
- future backend swaps are localized to `core/vision`, not repeated across every task module.

The migration is therefore a boundary cleanup, not a direct OpenCV-to-stb search-and-replace.

## Current Problem

OpenCV was used as the library vocabulary:

1. public input image type: `cv::Mat`;
2. public geometry type: `cv::Size`;
3. public dtype values: `CV_*` integer macros through `ModelInfo::input_types`;
4. result interop helpers: `toCvMat`, `fromCvMat`, `toCvRect`;
5. internal image operations: resize, split, merge, threshold, color conversion, NMS, flow visualization;
6. tests and fixtures: `cv::Mat::zeros`, `cv::rectangle`, `CV_32FC*` assertions.

That makes OpenCV removal wide because the dependency is not just an implementation detail. It is part of the public contract and test vocabulary.

## Target Architecture

### Public Vision Layer

Add a dedicated namespace and folder:

```text
include/neuriplo/tasks/core/vision/
src/core/vision/
```

Public types:

- `neuriplo_tasks::vision::Image`
- `neuriplo_tasks::vision::ImageView`
- `neuriplo_tasks::vision::Size`
- `neuriplo_tasks::vision::Rect`
- `neuriplo_tasks::vision::Point`
- `neuriplo_tasks::vision::Point2f`
- `neuriplo_tasks::vision::PixelType`
- `neuriplo_tasks::vision::ColorFormat`
- `neuriplo_tasks::vision::Layout`

Public task headers may re-export or type-alias these only if source compatibility requires it, but implementation code should use explicit `neuriplo_tasks::vision::...` names. Do not use `using namespace`.

### Central Operations Layer

Task implementations call only central operations:

- `resize`
- `crop`
- `copyRegion`
- `convertPixelType`
- `normalize`
- `splitChannels`
- `mergeChannels`
- `swapRgbBgr`
- `thresholdBinary`
- `minMax`
- `cartToPolar`
- `nms`

The operations layer owns numeric behavior. Backend-specific implementations must match its tests.

### Backend Adapters

Backend code is optional and isolated:

```text
include/neuriplo/tasks/core/vision/opencv_adapter.hpp
src/core/vision/opencv_adapter.cpp

include/neuriplo/tasks/core/vision/stb_io.hpp
src/core/vision/stb_io.cpp
```

OpenCV adapter responsibilities:

- wrap `cv::Mat` as `ImageView`;
- copy `Image` / `ImageMatrix` to `cv::Mat` for downstream consumers that still render with OpenCV;
- convert `cv::Rect` / `cv::Size` at consumer boundaries only.

stb responsibilities:

- image file load/save only;
- no task code includes stb headers;
- no stb-specific types escape into public task contracts.

## CMake Design

Core library stays dependency-light:

```cmake
option(NEURIPLO_TASKS_WITH_STB "Enable stb image I/O helpers" ON)
option(NEURIPLO_TASKS_WITH_OPENCV "Enable optional OpenCV interop adapter" OFF)
```

Targets:

```text
neuriplo-tasks
neuriplo-tasks::vision-core
neuriplo-tasks::vision-stb       optional
neuriplo-tasks::vision-opencv    optional
```

Rules:

- `neuriplo-tasks` must build without OpenCV.
- OpenCV may be used in adapter tests when `NEURIPLO_TASKS_WITH_OPENCV=ON`.
- stb include paths stay private to the stb implementation target.
- exported package config must not require OpenCV unless the OpenCV adapter target is requested.

## Execution Phases

### Phase 0 — Stop The Direct Replacement

- [x] Pause broad task-by-task OpenCV-to-stb edits.
- [x] Keep the branch compiling in the smallest useful slice.
- [x] Record every current OpenCV touchpoint as either public contract, internal operation, adapter, test fixture, or docs.

### Phase 1 — Introduce `core/vision`

- [x] Move current `Image`, `ImageView`, `Size`, `PixelType`, `Point2f`, and `BoundingBox` replacements into `include/neuriplo/tasks/core/vision/`.
- [x] Decide whether `BoundingBox` remains task-domain terminology or aliases `vision::Rect`.
- [ ] Add explicit namespace use in implementation code: `neuriplo_tasks::vision::Size`, not unqualified `Size` and not `using namespace`.
- [x] Add a short architecture note in this file explaining dependency direction.

### Phase 2 — Centralize Operations

- [x] Move `image_ops` into `src/core/vision/`.
- [x] Make all resize, channel, threshold, copy, min/max, flow-color, and NMS logic reachable only through the central operations API.
- [x] Add focused tests for operation behavior before continuing task migration.
- [x] For numerically sensitive operations, keep OpenCV comparison tests behind `NEURIPLO_TASKS_WITH_OPENCV=ON` until confidence is high.

### Phase 3 — Public API Flip

- [x] `TaskInterface::preprocess(std::vector<cv::Mat>)` -> `std::vector<vision::ImageView>` or `std::vector<vision::Image>`, choosing one contract and applying it consistently.
- [x] `TaskInterface::postprocess(cv::Size, ...)` -> `vision::Size`.
- [x] `BatchRequest::images` -> the same image contract chosen for `TaskInterface`.
- [x] `ModelInfo::input_types` -> `std::vector<vision::PixelType>`.
- [x] Update all postprocessor base classes and concrete task headers in one coordinated commit.

### Phase 4 — Optional Adapters

- [x] Add OpenCV adapter only under `NEURIPLO_TASKS_WITH_OPENCV=ON`.
- [x] Add stb load/save only under `NEURIPLO_TASKS_WITH_STB=ON`.
- [x] Move `opencv_interop.hpp` compatibility into the optional adapter or remove it if consumers migrate directly.
- [x] Ensure core install and package export do not require OpenCV.

### Phase 5 — Tests Without Scattered Backend Types

- [x] Add `tests/vision_test_utils.hpp` for test images, rectangles, and simple drawing.
- [x] Tests should construct task inputs through that helper, not through OpenCV or stb directly.
- [x] Replace `CV_*` assertions with `vision::PixelType` and channel-count assertions.
- [x] Keep backend-specific adapter tests in separate files gated by the matching CMake option.

### Phase 6 — Consumers

Each consumer gets its own boundary adapter rather than pulling OpenCV back into `neuriplo-tasks`:

- `neuriplo-track`: wrap app frames at the task boundary; keep tracker-internal OpenCV independent.
- `neuriplo-infer`: wrap decoded frames and unwrap result images in rendering code.
- `tritonic`: convert Triton metadata dtypes to `vision::PixelType`; wrap batch frames at the boundary.

Consumer migration should be small because the central API becomes stable.

### Phase 7 — Docs And Release

- [x] `README.md`: dependency statement, optional adapter targets, public image contract.
- [x] `AGENTS.md`: core vision layer rule and no backend includes outside adapters.
- [x] `REPO_META.yaml`: build/test commands for no-OpenCV and optional-OpenCV adapter builds.
- [x] `CHANGELOG.md`: breaking API changes and dependency changes.
- [ ] Release only after no-OpenCV core build, test build, and at least one consumer smoke build pass.

## Dependency Rule

Allowed:

```text
tasks -> core/vision API -> backend implementation/adapters
consumers -> consumer adapter -> core/vision API
```

Forbidden:

```text
tasks -> OpenCV
tasks -> stb
tasks -> SDL
tasks -> FFmpeg
tests -> backend-specific constructors outside adapter tests
```

## Validation Gates

Core no-OpenCV build:

```bash
cmake -S . -B build-no-ocv -DBUILD_TESTS=ON -DNEURIPLO_TASKS_WITH_OPENCV=OFF
cmake --build build-no-ocv --parallel
ctest --test-dir build-no-ocv --output-on-failure
```

Optional OpenCV adapter build:

```bash
cmake -S . -B build-ocv -DBUILD_TESTS=ON -DNEURIPLO_TASKS_WITH_OPENCV=ON
cmake --build build-ocv --parallel
ctest --test-dir build-ocv --output-on-failure
```

Standard quality gate remains clang-format, clang-tidy, cppcheck, `-DWERROR=ON`, ctest, sanitizer build, and Valgrind where applicable.

## Risk Register

| Risk | Mitigation |
|---|---|
| Replacement stays scattered under new names | Enforce no backend includes outside `core/vision/*adapter*` and tests for adapters |
| stb mistaken for an imgproc backend | Keep stb limited to file I/O |
| Numeric drift in resize/flow/depth | Operation-level tests plus optional OpenCV comparison tests during migration |
| Consumers need OpenCV rendering | Keep OpenCV in consumers or optional adapter, not in task core |
| Public namespace churn | Do the namespace move before consumer migration and document aliases if temporary compatibility is needed |
