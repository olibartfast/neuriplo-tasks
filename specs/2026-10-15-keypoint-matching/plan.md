# Keypoint matching — plan

Task groups in dependency order. Each ends in something observable. Group 1 was
completed before any code, as the workflow requires.

## 1. Spec packet

Write `requirements.md`, this plan, and `validation.md` — including the checks
that define done — before touching `include/` or `src/`.

**Observable:** three documents exist and agree with the constitution's
constraint check.

## 2. Result schema

- `include/neuriplo/tasks/core/result_types.hpp`: add `KeypointMatch` and
  `KeypointMatching`; append to the `Result` variant and `TaskType`.
- `tests/test_result_types.cpp`: cover default construction, variant storage,
  and the every-alternative visitor; assert the new enum value.

**Observable:** the suite compiles — the visitor's `if constexpr` chain forces
the new alternative to be handled — and passes.

## 3. Task module skeleton

- `include/neuriplo/tasks/keypoint_matching/`: `keypoint_matching_task.hpp`,
  `keypoint_matching_preprocessor.hpp`, `keypoint_matching_postprocessor.hpp`,
  `superglue_postprocessor.hpp`, `lightglue_postprocessor.hpp`,
  `matchanything_postprocessor.hpp`.
- `src/keypoint_matching/`: `keypoint_matching_task.cpp`,
  `keypoint_matching_preprocessor.cpp`, one `.cpp` per family postprocessor.
- `CMakeLists.txt`: sources and headers.
- Task behavior: exactly 2 images, `getRequiredFrames() == 2`, records pair
  sizes for postprocess, one `KeypointMatching` result per pair.
- Family postprocessors throw `std::logic_error` with the reason and the layout
  they expect.

**Observable:** the library builds with `-DWERROR=ON`; the factory constructs a
task for every routed alias.

## 4. Factory routing and the README contract

- `src/core/task_factory.cpp`: `KeypointMatching` descriptor —
  `superglue*`, `lightglue*`, `matchanything*`, `keypointmatching`.
- `README.md`: `## Features` bullet and the `TASKFACTORY_MODEL_LIST` block
  (a routing change that skips it fails `test_readme_model_types`).

**Observable:** `test_readme_model_types` passes — both directions — with the
new aliases.

## 5. Tests and consumer documentation

- `tests/test_keypoint_matching.cpp` + `tests/CMakeLists.txt` registration:
  routing (case/hyphen/spaced forms, prefix revisions), unknown family
  rejected, pair validation, preprocess buffer count, postprocess seams,
  empty-tensor behavior.
- `export/keypoint_matching/README.md`: where to get SuperGlue / LightGlue /
  MatchAnything ONNX exports, the I/O contracts, and how to probe a file
  before implementing its decoder. Update `export/README.md` tree and links.
- `docs/batch_support_matrix.md`: new family row and the non-batch note.
- `CHANGELOG.md` under `[Unreleased]`; `specs/roadmap.md` in flight.

**Observable:** `ctest` green including the new suite.

## 6. Gate

Run the local gate from [AGENTS.md](../../AGENTS.md): clang-format, cppcheck,
`-DWERROR=ON` build, ctest, Valgrind. Record honest results in
`validation.md`, including anything that could not run.

**Observable:** every runnable check green before the branch is offered for
review.

## 7. SuperGlue decoder (delivered 2026-11-02)

Added after the scaffold merged to the branch, once an export could be probed.

- `include/.../superglue_preprocessor.hpp` + `src/.../superglue_preprocessor.cpp`:
  grayscale, `[0,1]`, pair packed into one `[2, 3, H, W]` buffer.
  `KeypointMatchingPreprocessor::preprocess_pair` becomes virtual so the family
  can override it.
- `SuperGluePostprocessor`: shape-driven output classification, mask-aware
  keypoint collection, match index remapping, score threshold.
- `KeypointMatchingTask`: rank-5 input-size extraction (`[1, 2, C, H, W]`),
  SuperGlue preprocessor/postprocessor selection, 640x480 default, threshold
  from `TaskConfig::confidence_threshold` via `TaskFactory`.
- `tests/test_keypoint_matching.cpp`: packing, grayscale values, verified-export
  decode, threshold, missing mask, malformed tensors, task-level pair sizes.
- Docs: README contract, export guide with the verified table and
  `probe_superglue.py`, batch matrix, changelog, roadmap, this packet.

**Observable:** `test_keypoint_matching` green (17 tests), and the decoder
reproduces the Python reference on a real ONNX export (validation V-12/V-13).
