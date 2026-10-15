# Keypoint matching — validation

Written before implementation. Each requirement maps to an observable check;
the Results section records what was actually run, and a check that could not
be run is **not executed**, not a pass.

## Checks

| # | Requirement | Check | Result |
|---|-------------|-------|--------|
| V-1 | R-1 result schema | `tests/test_result_types.cpp` covers `KeypointMatch`/`KeypointMatching` default construction, variant storage, the every-alternative visitor, and the `TaskType` value; passes | **pass** |
| V-2 | R-2 pair contract | `tests/test_keypoint_matching.cpp` asserts `getRequiredFrames() == 2`; `preprocess` throws `std::invalid_argument` for 1 and 3 images; 2 images return 2 buffers | **pass** |
| V-3 | R-3 routing | Factory tests: `superglue`, `superglue-2020`, `lightglue`, `LightGlue`, `lightglue_v2`, `matchanything`, `MatchAnything-EloFTR` all construct `KeypointMatchingTask`; `sift` and the family-less `keypointmatching` still throw | **pass** |
| V-4 | R-3 README contract | `test_readme_model_types` passes in both directions (documented aliases route; every factory literal is documented) | **pass** |
| V-5 | R-4 preprocessing | Pair preprocessor yields one non-empty buffer per image of equal expected length; empty pair rejected | **pass** |
| V-6 | R-5 postprocess seams | Family postprocessors throw `std::logic_error` while undecoded; empty tensor list returns no results | **pass** |
| V-7 | Constitution | No new runtime dependencies; no new OpenCV use; existing suites unchanged and green | **pass** |
| V-8 | Gate | clang-format-18 `--dry-run --Werror`, cppcheck, `-DWERROR=ON` build, `ctest` | **pass** |
| V-9 | Gate | Valgrind on the Debug test build (new suite included) | **not executed** |
| V-10 | Consumer docs | `export/README.md` links resolve; `export/keypoint_matching/README.md` present; batch matrix has the new family row | **pass** |
| V-11 | Deferred work is honest | README and changelog say decoding is not implemented; no decoder claims behavior that was never run against an ONNX file | **pass** |

**Explicitly not executed in this packet:**

- End-to-end keypoint matching against a real SuperGlue / LightGlue /
  MatchAnything ONNX export — no verified file exists in this repo yet; that is
  the next packet (requirements open questions 1–3).
- Matching accuracy against labelled pairs — needs a dataset and an evaluation
  harness this repository does not have.
- Multi-pair / batched matching — out of scope by decision 2.

## Results

**Date:** 2026-10-16 · **Branch:** `feature/keypoint-matching` at first full
local run. Build: Debug, `-DWERROR=ON`, GCC, in a scratch build directory.

| Check | Evidence |
|-------|----------|
| V-1 | `test_result_types` passes (11 alternatives visited, enum value 10 asserted). |
| V-2 | `test_keypoint_matching` — 9 tests, all pass: exactly-two-images, empty-image, 2-buffer preprocessing, direct pair preprocessor. |
| V-3 | Same suite: 7 prefixed aliases route (`getRequiredFrames() == 2` each); `sift` and `keypointmatching` throw `std::invalid_argument`. |
| V-4 | `test_readme_model_types`: 3/3 pass, including reverse direction (no factory literal missing from the README block). |
| V-5 | `PreprocessReturnsOneBufferPerImage` asserts `2 × 3 × 512 × 512 × 4` bytes; `PreprocessorRejectsEmptyImage` passes. |
| V-6 | `DecodingIsNotImplementedYet` (all three families throw `std::logic_error`) and `PostprocessWithEmptyTensorsReturnsEmpty` pass. |
| V-7 | `ctest`: 30/30 suites pass; no new dependency or OpenCV include in the diff. |
| V-8 | `clang-format 18.1.8 --dry-run --Werror`: clean (installed to match CI's pin); `cppcheck` exit 0 (58/58 files); library build with `-DWERROR=ON` exit 0. `clang-tidy` run on the new sources: no warnings (a newer local clang-tidy 21 reported only pre-existing `image.hpp` warnings under a check added after 18). |
| V-9 | **Not executed.** Valgrind is not installed on the authoring machine. The CI `valgrind` job runs the full suite and is the authoritative check; the new suite has no test-only heap state beyond standard GTest fixtures. |
| V-10 | `export/keypoint_matching/README.md` added; `export/README.md` tree, quick-start pointer, and reference link updated; `docs/batch_support_matrix.md` gained the keypoint-matching row and the pair-semantics note. Relative references resolve; the absolute GitHub link follows the existing pattern and resolves once merged to `master`. |
| V-11 | README Features bullet, `TASKFACTORY_MODEL_LIST` entry, changelog entry, and both postprocessor headers all state that decoding is not implemented pending a verified export. |

## Deviations recorded during the work

- **The generic `keypointmatching` alias was dropped mid-implementation.** The
  factory routed it, but `KeypointMatchingTask::detectModelType` had no family
  to select a decoder for, so the task threw one layer later. Requirements
  decision 3 and open question 5 were updated in the same branch to record the
  reason and the condition under which the alias becomes meaningful. The test
  now asserts the name is rejected.
