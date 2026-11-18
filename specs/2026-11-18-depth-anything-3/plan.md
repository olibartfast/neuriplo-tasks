# Depth Anything 3 — Plan

1. **Task core** (`include` + `src/depth_estimation/`): add
   `ModelType::DEPTH_ANYTHING_3` to `DepthEstimationTask`; add
   `DepthAnything3Preprocessor` (504 default); extend `detectModelType`
   (metric/nested reject with reason), `createPreprocessor`,
   `createPostprocessor` (reuse `DepthAnythingV2Postprocessor`), and the
   `extractInputSize` empty-shape fallback (504 for DA3). Observable: task
   constructs for `da3-small`, throws for `da3metric-large`.
2. **Factory** (`src/core/task_factory.cpp`): add the `DepthAnything3`
   descriptor after YOLO-depth. Observable: `createTaskInstance("da3-small")`
   and `("depth-anything-3")` return `TaskType::DepthEstimation`.
3. **Unit tests**: extend `tests/test_depth_estimation.cpp` (DA3 preprocess
   explicit shapes, 504 fallback on empty shapes, synthetic `[1,H,W]`
   postprocess, metric rejection) and `tests/test_task_factory.cpp`
   (creation, domain list, stable-alias list, normalization entries).
   Observable: `ctest -R "depth|task_factory|readme"` green.
4. **Docs**: README `TASKFACTORY_MODEL_LIST` Depth block (auto-checked),
   Features depth bullet, `export/depth_estimation/depth_anything_3/README.md`
   + `export/README.md` tree entry, CHANGELOG `[Unreleased]`, roadmap link.
   Observable: `test_readme_model_types` green, hyperlink check clean.
5. **Gate**: clang-format, cppcheck, full `cmake -DBUILD_TESTS=ON
   -DWERROR=ON` build, `ctest`, valgrind on the depth binaries.
