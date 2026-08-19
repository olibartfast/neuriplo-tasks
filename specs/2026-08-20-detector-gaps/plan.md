# Detector gaps — plan

Task groups in dependency order. Each ends in something observable.

## 1. Reproduce all three failures against real models

Before any edit, confirm each symptom from the actual ONNX file rather than
from reading the code, and capture the evidence that identifies the cause:

- run `yolov10n` and print the raw output tensor's coordinate range;
- run `rtdetrv4` and compare box placement against the same image;
- construct `deimv2` through `TaskFactory` and capture the exception text.

**Observable:** three reproductions, and for the first a concrete number
(maximum coordinate) that settles which convention the file uses.

## 2. NMS-free coordinate convention

- `src/object_detection/yolo_postprocessor.cpp`: scan all four coordinates of
  all detections for the maximum; compare against
  `kNormalizedCoordinateLimit`; scale by input size only when normalized.
- Tests: `YoloNmsFreeAcceptsPixelCoordinates` (pixel-space input decodes
  unscaled) and `YoloNmsFreeKeepsConventionsSeparate` (the same box expressed
  both ways decodes to the same result).

**Observable:** both new tests fail with the fix reverted and pass with it;
the existing `YoloNmsFreeFormat` and `YoloNmsFreeAppliesNmsPerClass` still pass.

## 3. RT-DETR family preprocessing

- Add `EdgeCrafterPreprocessor` retaining ImageNet normalization.
- Remove ImageNet normalization from `RtDetrPreprocessor` and
  `DFinePreprocessor`.
- Point `ObjectDetectionTask`'s `EDGECRAFTER` branch at the new class so its
  behaviour does not move.

**Observable:** `rtdetrv4` returns a full, correctly placed set on the same
image that previously gave five misplaced boxes.

## 4. Factory routing

- `src/core/task_factory.cpp`: extend the DetrDetection matcher with
  `rtdetrv2`, `dfine`, and a `deim` prefix match.
- `object_detection_task.cpp`: match `dfine` and `deim` by prefix in
  `detectModelType`.
- `README.md`: update the `TASKFACTORY_MODEL_LIST` block — a routing change
  that skips it fails `test_readme_model_types`.
- `tests/test_task_factory.cpp`: add the four aliases and their hyphenated,
  spaced, and mixed-case spellings.

**Observable:** `deimv2` constructs and runs; the factory test covers each new
alias.

## 5. Gate and integrate

- Run the full local gate from [AGENTS.md](../../AGENTS.md): clang-format,
  cppcheck, `-DWERROR=ON` build, ctest, Valgrind.
- Add the `CHANGELOG.md` entry under `[Unreleased]`.
- Open the pull request against `develop` — `master` accepts only `develop`,
  `release/*`, and `hotfix/*`.

**Observable:** every CI job green on the pull request.
