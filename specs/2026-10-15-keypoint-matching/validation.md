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
| V-9 | Gate | Valgrind on the Debug test build (new suite included) | **pass (CI)** — local Valgrind unavailable; the CI `valgrind` job ran the full suite green |
| V-10 | Consumer docs | `export/README.md` links resolve; `export/keypoint_matching/README.md` present; batch matrix has the new family row | **pass** |
| V-11 | Deferred work is honest | README and changelog say decoding is not implemented; no decoder claims behavior that was never run against an ONNX file | **pass** (scaffold), superseded for SuperGlue by V-13 |
| V-12 | In scope — SuperGlue preprocessing | `SuperGluePreprocessor` emits one buffer of `2 × 3 × H × W × 4` bytes; grayscale values match torchvision weights on a palette image; the two channels planes are identical | **pass** |
| V-13 | In scope — SuperGlue decoding | On a real ONNX export, C++ decode equals the Python `post_process_keypoint_matching` reference (match count and per-match score/coordinates) | **pass** |
| V-14 | Decision 7 — score threshold | Matches at or below `TaskConfig::confidence_threshold` are dropped; a threshold above every score yields no matches | **pass** |
| V-15 | Gate | clang-format-18 `--dry-run --Werror`, cppcheck, clang-tidy-18, `-DWERROR=ON` build, `ctest`, Valgrind | **pass (local)** |

**Explicitly not executed in this packet:**

- End-to-end keypoint matching against a real LightGlue / MatchAnything ONNX
  export — no verified file exists for those families yet; that is the next
  slice (requirements open questions 1–4). **SuperGlue was executed — see the
  2026-11-02 results below.**
- Matching accuracy against labelled pairs — needs a dataset and an evaluation
  harness this repository does not have.
- Multi-pair / batched matching — out of scope by decision 2.

## Results — SuperGlue decoder (2026-11-02)

**Model:** `KnightsAnalytics/superglue_outdoor` (Optimum export of
`magic-leap-community/superglue_outdoor`, opset 17, input
`pixel_values [1, 2, 3, 480, 640]`).
**Images:** magicleap phototourism Capitol pair, `1016x762` (source) and
`1048x626` (target).
**Merged:** PR [#21](https://github.com/olibartfast/neuriplo-tasks/pull/21) into
`develop` (`6b78f7f`); post-merge develop CI run
`41440696505` green on all seven jobs.

| Check | Evidence |
|-------|----------|
| V-12 | `SuperGluePreprocessPacksOnePairBuffer`: one buffer of `2 × 3 × 480 × 640 × 4` bytes. `SuperGluePreprocessAppliesGrayscaleAndUnitScaling`: red→`0.2989`, blue→`0.1140`, green→`0.5870`, white→`0.9999`, all three channel planes identical, second slot carries the target image. |
| V-13 | Real export: `N=903`, 795 source / 903 target valid keypoints, 233 matches above score 0. C++ decode reproduced the Python reference on all 233 matches — max score/coordinate delta `1.17e-6` (float32 rounding). Preprocessed pixels versus the transformers pipeline (PIL resize + torchvision grayscale): `mean |diff| = 0.004`, `max 0.21` on the `[0,1]` scale; the residual is PIL's antialiased bilinear downscale versus this library's plain bilinear, not a convention error. A `Linear`-kernel buffer is the closest of the three available kernels (mean 0.004 vs 0.012 `Area`, 0.007 `Cubic`). |
| V-14 | `SuperGlueDropsMatchesBelowThreshold`: 1 match at threshold 0, 0 matches at 0.995. `SuperGlueTaskScalesToRecordedPairSizes`: factory default (`confidence_threshold=0.25`) keeps the 0.99 match and rescales coordinates by the recorded pair sizes. |
| V-15 | `clang-format-18 --dry-run --Werror` clean; `cppcheck` exit 0 (59 files); `clang-tidy-18` clean on the three touched sources (an earlier `readability-function-cognitive-complexity` warning on `extractInputSize` was removed by the rank switch, not suppressed); `-DWERROR=ON` build exit 0; `ctest` 30/30; Valgrind on the Debug `test_keypoint_matching` (17 tests) `0 errors, 0 leaks`. |

**Deviations recorded during the SuperGlue work:**

- **`preprocess()` returns one buffer for SuperGlue, not two.** The export has a
  single rank-5 input holding the pair, and every buffer maps 1:1 onto a model
  input, so `SuperGluePreprocessor` packs both images into one buffer. The
  scaffold's "one buffer per image" rule holds only for two-input families;
  requirements decision 5 was rewritten to record this.
- **The scaffold's RGB preprocessor was wrong for this family.** Verified
  grayscale with torchvision weights; the split the scaffold deferred is now
  implemented as `SuperGluePreprocessor`.
- **`matches` indexes padded keypoints, not masked ones.** Verified by checking
  that every referenced index falls inside the mask and that the mask is
  prefix-packed. The decoder still guards `mask[1][k] > 0`, so it stays correct
  if a future export changes the packing.
- **No detector score in the export.** `Keypoint::confidence` stays `0` for
  SuperGlue keypoints; this is documented on the postprocessor, in the README
  contract, and in the export guide rather than faked with a placeholder value.
- **Raising `TaskConfig::confidence_threshold` changes match counts.** On the export's shipped
  reference pair (`reference_0.npz`), 184 of 184 matches survive 0, 174 survive the factory default
  0.25, and 156 survive 0.9. Consumers relying on "every mutual match" should
  set the threshold to 0.

## Results — scaffold (2026-10-16)

**Branch:** `feature/keypoint-matching` at first full
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
| V-9 | **Passed in CI, not locally.** Valgrind is not installed on the authoring machine, so the gate was left to the CI `valgrind` job, which runs every test binary under `--leak-check=full`; it completed green on run `38814696657`. |
| V-10 | `export/keypoint_matching/README.md` added; `export/README.md` tree, quick-start pointer, and reference link updated; `docs/batch_support_matrix.md` gained the keypoint-matching row and the pair-semantics note. Relative references resolve; the absolute GitHub link follows the existing pattern and resolves once merged to `master`. |
| V-11 | README Features bullet, `TASKFACTORY_MODEL_LIST` entry, changelog entry, and both postprocessor headers all state that decoding is not implemented pending a verified export. |

**CI:** all seven jobs green on run
[`38814696657`](https://github.com/olibartfast/neuriplo-tasks/actions/runs/38814696657) —
Format Check, Clang-Tidy, Cppcheck, Build & Test, Build with Strict Warnings,
Valgrind, Windows MSVC.

## Deviations recorded during the work

- **The generic `keypointmatching` alias was dropped mid-implementation.** The
  factory routed it, but `KeypointMatchingTask::detectModelType` had no family
  to select a decoder for, so the task threw one layer later. Requirements
  decision 3 and open question 5 were updated in the same branch to record the
  reason and the condition under which the alias becomes meaningful. The test
  now asserts the name is rejected.
