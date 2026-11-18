# Depth Anything 3 — Validation

Written before implementation; Results filled in after.

## Checks

- V-1 — `DepthEstimationTask` constructs for `da3-small`, `da3mono-large`,
  `depth-anything-3` (504 fallback geometry on empty shapes; explicit shapes
  honored) and throws `std::invalid_argument` for `da3metric-large`,
  `da3-nested-giant-large`, and unknown depth names.
- V-2 — Synthetic `[1,H,W]` tensor through `postprocess` returns one
  `DepthEstimation` per batch row with correct min/max and frame-size resize.
- V-3 — Factory routes `da3-small`, `depth-anything-3`, `DA3_Mono_Large`,
  ` da3 base ` to `TaskType::DepthEstimation`; README block aliases resolve
  (`test_readme_model_types` green).
- V-4 — Full gate green: clang-format `--dry-run --Werror`, cppcheck,
  `cmake -DBUILD_TESTS=ON -DWERROR=ON` build, `ctest --output-on-failure`,
  valgrind on `test_depth_estimation`, hyperlink check on touched docs.
- V-5 — Real-ONNX inference against DA3 weights: NOT EXECUTED here (no
  onnxruntime, no weights in this environment). Contract cross-checked
  against the DA3 `export_onnx.py` output names/shapes (`depth`/`conf`,
  `[1,H,W]`); a weights-holding environment should run the export guide
  end to end as follow-up.

## Results

- V-1 — pass (`DepthAnything3*` fallback/reject tests green).
- V-2 — pass (synthetic `[1,4,5]` exact min/max at frame size; resize path
  geometry-only — cubic ringing makes edge-exact values meaningless).
- V-3 — pass (`test_readme_model_types` 3/3, incl. the reverse
  factory-key check that caught the missing `"da3"` alias).
- V-4 — pass locally except valgrind (binary not installed in this
  environment; covered by CI): clang-format `--dry-run --Werror` clean,
  cppcheck clean, `cmake -DBUILD_TESTS=ON -DWERROR=ON` build clean,
  `ctest` 33/33 green, `git diff --check` clean. New absolute
  `blob/master` links follow existing entries and resolve at merge.
- V-5 — Not executed (see above).
