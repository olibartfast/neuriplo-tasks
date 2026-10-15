# Keypoint matching — validation

Written before implementation. Each requirement maps to an observable check;
the Results section is filled in after the checks are actually run, and a check
that could not be run is recorded as **not executed**, not as a pass.

## Checks

| # | Requirement | Check | Result |
|---|-------------|-------|--------|
| V-1 | R-1 result schema | `tests/test_result_types.cpp` covers `KeypointMatch`/`KeypointMatching` default construction, variant storage, the every-alternative visitor, and the `TaskType` value; passes | pending |
| V-2 | R-2 pair contract | `tests/test_keypoint_matching.cpp` asserts `getRequiredFrames() == 2`; `preprocess` throws `std::invalid_argument` for 1 and 3 images; 2 images return 2 buffers | pending |
| V-3 | R-3 routing | Factory tests: `superglue`, `superglue-2020`, `lightglue`, `LightGlue`, `lightglue_v2`, `matchanything`, `MatchAnything-EloFTR`, `keypoint-matching` all construct `KeypointMatchingTask`; `sift` still throws | pending |
| V-4 | R-3 README contract | `test_readme_model_types` passes in both directions (documented aliases route; every factory literal is documented) | pending |
| V-5 | R-4 preprocessing | Pair preprocessor yields one non-empty buffer per image of equal expected length; empty pair rejected | pending |
| V-6 | R-5 postprocess seams | Family postprocessors throw `std::logic_error` while undecoded; empty tensor list returns no results | pending |
| V-7 | Constitution | No new runtime dependencies; no new OpenCV use; existing suites unchanged and green | pending |
| V-8 | Gate | clang-format-18 `--dry-run --Werror`, cppcheck, `-DWERROR=ON` build, `ctest` | pending |
| V-9 | Gate | Valgrind on the Debug test build (new suite included) | pending |
| V-10 | Consumer docs | `export/README.md` links resolve; `export/keypoint_matching/README.md` present; batch matrix has the new family row | pending |
| V-11 | Deferred work is honest | README and changelog say decoding is not implemented; no decoder claims behavior that was never run against an ONNX file | pending |

**Explicitly not executed in this packet:**

- End-to-end keypoint matching against a real SuperGlue / LightGlue /
  MatchAnything ONNX export — no verified file exists in this repo yet; that is
  the next packet (requirements open questions 1–3).
- Matching accuracy against labelled pairs — needs a dataset and an evaluation
  harness this repository does not have.
- Multi-pair / batched matching — out of scope by decision 2.

## Results

Filled in after implementation — see branch commits for the exact runs.
