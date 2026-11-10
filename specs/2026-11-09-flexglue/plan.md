# Plan — FlexGlue keypoint matching

Task groups in dependency order. Each ends in something observable.
Group 1 (this packet) was completed before any code, as the workflow requires.

## 1. Spec packet

Write `requirements.md`, this plan, and `validation.md` — including the checks
that define done — before touching `include/` or `src/`.

**Observable:** three documents exist and agree with the constitution's
constraint check.

## 2. Probe the real exports (before any decoder)

- Download `flexglue_descriptor.onnx` + `flexglue_matcher.onnx` (v0.1 release,
  SHA-pinned) to a scratch dir (never committed).
- Add `export/keypoint_matching/probe_flexglue.py`: print graph I/O, run the
  descriptor on two test images (squashed bilinear, RGB `[0,1]`), detect SIFT
  keypoints (+0.5 corner offset), run the sparse matcher, dump shapes/dtypes
  plus the values the decoder depends on (keypoint coordinate space, match
  index space, covisibility gating, `max_distance_px=0` vs `inf` behavior).
- Record the verified contract table in the export guide.

**Observable:** probe output + contract table; any deviation from the assumed
table in requirements updates requirements/validation in the same branch.

## 3. Preprocessor + task skeleton

- `include/neuriplo/tasks/keypoint_matching/flexglue_preprocessor.hpp` +
  `src/keypoint_matching/flexglue_preprocessor.cpp`: subclass
  `KeypointMatchingPreprocessor`, override `preprocess_pair()` to emit **two**
  RGB `[0,1]` NCHW buffers squashed to the descriptor size (BGR→RGB swap,
  bilinear, no grayscale, no mean/std). Channel count from model input.
- `KeypointMatchingTask`: `ModelType::FLEXGLUE`, `detectModelType`
  (`flexglue*` prefix), preprocessor/postprocessor selection, 4-D input-size
  path for the descriptor shape, default 512×512 fallback.
- `src/core/task_factory.cpp`: `flexglue*` descriptor entry (prefix matcher,
  same normalization as siblings).
- Until the decoder lands, `FlexGluePostprocessor::postprocess` throws
  `std::logic_error` with the expected layout (per D-5).

**Observable:** library builds with `-DWERROR=ON`; factory constructs the
task for `flexglue` aliases; postprocess throws instead of guessing.

## 4. Postprocessor (after probe)

- `include/.../flexglue_postprocessor.hpp` +
  `src/.../flexglue_postprocessor.cpp`: shape-driven classification —
  `kpts [1, N, 2]` float ×2, `matches [1, N0]` int, `scores [1, N0]` float,
  optional `covisibility [1]` float. Validate element counts, index bounds,
  finiteness; keep scores strictly above `score_threshold_`; pass keypoints
  through unscaled; return one `KeypointMatching`.
- Malformed inputs (missing kpts/matches/scores, count mismatches,
  out-of-range indices, non-finite coords) throw `std::invalid_argument`;
  empty-tensor input returns empty (consistent with siblings).

**Observable:** unit decode of probe tensors equals the Python reference
(max delta ~1e-6 float32 rounding).

## 5. Tests and consumer documentation

- `tests/test_keypoint_matching.cpp`: routing (case/hyphen/space/prefix
  forms, unknown family still rejected), pair validation, two-buffer RGB
  values (red→0.2989-adjacent? No — FlexGlue is **not** grayscale: assert
  per-channel RGB planes), verified-export decode, threshold filtering,
  forwarded-keypoint errors, malformed tensors, task-level pair sizes.
- `export/keypoint_matching/README.md`: FlexGlue section (two-model flow,
  SIFT note, corner convention, 3-step consumer snippet, probe command).
  `export/README.md` tree/links, `docs/batch_support_matrix.md` row (one
  pair per call), `CHANGELOG.md` `[Unreleased]`, `specs/roadmap.md` in-flight
  entry.
- `README.md`: `## Features` bullet + `TASKFACTORY_MODEL_LIST` block
  (asserted by `test_readme_model_types` in both directions).

**Observable:** `ctest` green including the new suite.

## 6. Gate

Run the local gate from [AGENTS.md](../../AGENTS.md): clang-format-18,
cppcheck, clang-tidy-18, `-DWERROR=ON` build, `ctest`, Valgrind. Record
honest results in `validation.md`, including anything that could not run.

**Observable:** every runnable check green before review.

## Notes

- Conventions reused: shape-driven output classification, strictly-greater
  threshold, `startsWith` prefix routing, per-family preprocessor split.
- FlexGlue breaks the "one buffer per image / packed pair" dichotomy: its
  preprocess emits one buffer **per descriptor run** (two total) while its
  postprocess consumes **matcher** tensors — the two graphs meet in the
  consumer's session, not in the task.
