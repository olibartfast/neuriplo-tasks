# neuriplo-tasks — Agent & Contributor Instructions

This file is the **single source of truth** for coding conventions, tooling, and
rules for working in `neuriplo-tasks`. Any other agent-specific instruction file
must defer to this one.

## Repo metadata

- Machine-readable entrypoints and owned paths live in
  [`REPO_META.yaml`](./REPO_META.yaml). Prefer it as the source for build/test
  commands and allowed change classes when automating.
- **Branches**: `develop` is the integration branch for normal work; `master`
  is release-only.
- **Releases must align with tags**: Every Git tag (e.g., `v0.4.1`) must have
  a corresponding GitHub Release. Never push a tag without also creating the
  release via `gh release create`. Release notes must come from `CHANGELOG.md`
  — never use `--generate-notes`. Pass `--latest` on the newest
  release and `--latest=false` when backfilling older tags: GitHub assigns the
  "Latest" badge by publish time, not semver, so backfilling newest-first puts
  the badge on the oldest version. If a tag exists without a release, create
  the release immediately. See [`docs/Versioning.md`](./docs/Versioning.md)
  for the full release workflow.
- **Release/hotfix branch cleanup**: once a `release/*` or `hotfix/*` branch is
  merged to `master` (tagged `vX.Y.Z`) and back-merged to `develop`, delete it
  locally and on `origin`. The tag on `master` is the immutable release record;
  never leave a finished branch on the remote as version history. Do not push a
  release branch solely to watch CI when `develop` covers the same commits —
  wait on the `develop`/`master` workflow unless a PR is requested. Checklist:
  merge to `master` → tag and push → merge back to `develop` → push →
  `git branch -d <branch>` → `git push origin --delete <branch>` → confirm
  `git branch -a | grep release` shows no finished branch.
- **Priorities when reviewing a change**: correctness → backward compatibility
  → task-contract stability → shape/dtype assumptions → performance regressions.
- **Specifications**: [`specs/`](./specs/) holds the project constitution
  (`mission.md`, `tech-stack.md`, `roadmap.md`) and one dated packet per
  delivered slice. Read the constitution before planning a feature; it covers
  *what the library is and must stay*, while this file covers *how to work on
  it*. See [`specs/README.md`](./specs/README.md) for the workflow.

---

## Project overview

- **Artifact**: `libneuriplo-tasks.a` — C++17 static library
- **Source roots**: `src/`, `include/neuriplo/tasks/`
- **Tests**: `tests/` (GoogleTest, fetched via CMake `FetchContent`)
- **Core runtime dependencies**: none; optional adapters provide stb image I/O and OpenCV interop
- **Consumers**: [tritonic](https://github.com/olibartfast/tritonic),
  [neuriplo-infer](https://github.com/olibartfast/neuriplo-infer)
- **GitHub repo**: `https://github.com/olibartfast/neuriplo-tasks`

`neuriplo-tasks` provides framework-agnostic computer-vision pre/postprocessing
for inference engines.

---

## Development setup

```bash
# Standard build (no tests)
cmake -S . -B build
cmake --build build --parallel

# Build with tests
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --parallel

# Run all tests
ctest --test-dir build --output-on-failure

# Run a single test binary directly
./build/tests/test_task_factory

# Build with sanitizers (separate directory)
cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON -DSANITIZERS=ON
cmake --build build-san --parallel
ctest --test-dir build-san --output-on-failure

# Treat warnings as errors (matches the build-warnings CI job)
cmake -S . -B build -DWERROR=ON
cmake --build build --parallel

# Run tests under Valgrind (matches the valgrind CI job)
cmake -S . -B build-valgrind -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
cmake --build build-valgrind --parallel
for test_bin in build-valgrind/tests/test_*; do
  [ -x "$test_bin" ] || continue
  valgrind \
    --error-exitcode=1 \
    --leak-check=full \
    --show-leak-kinds=definite,indirect \
    --errors-for-leak-kinds=definite,indirect \
    --track-origins=yes \
    --num-callers=25 \
    "$test_bin"
done
```

Pre-commit hooks run clang-format and cppcheck automatically:
`pip install pre-commit && pre-commit install`.

---

## Code quality rules

### Tooling

| Tool           | Purpose                                  | Config              |
|----------------|------------------------------------------|---------------------|
| `clang-format-21` | Formatting (LLVM base, 4-space, 120 col) | `.clang-format`     |
| `clang-tidy-18`   | Static analysis on `src/` + `include/`   | `.clang-tidy`       |
| `cppcheck`        | Additional static analysis               | `.pre-commit-config.yaml` |
| `ctest` / GoogleTest | Unit tests                            | `tests/CMakeLists.txt` |
| `valgrind`        | Runtime memory/leak checks on test binaries | `.github/workflows/lint.yml` |
| `-DWERROR=ON`     | Treat compiler warnings as errors        | `CMakeLists.txt`    |

### Formatting

```bash
# Check (dry-run, CI mode)
find src include tests -name '*.cpp' -o -name '*.hpp' | \
  xargs clang-format-21 --dry-run --Werror

# Apply
find src include tests -name '*.cpp' -o -name '*.hpp' | \
  xargs clang-format-21 -i
```

### Static analysis

```bash
# clang-tidy (needs compile_commands.json)
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
find src -name '*.cpp' | xargs clang-tidy-18 -p build

# cppcheck
cppcheck --enable=warning --std=c++17 \
  --suppress=missingIncludeSystem \
  --suppress=unmatchedSuppression \
  --suppress=*:3rdparty/stb/* \
  --error-exitcode=1 \
  -I include src/
```

### Conventions

- **C++ standard**: C++17
- **Column limit**: 120
- **Indent**: 4 spaces, no tabs
- **Pointer alignment**: left (`T* p`, not `T *p`)
- **Includes**: sorted (case-sensitive), regrouped by `.clang-format`
- **Warnings**: `-DWERROR=ON` must stay green — no new warnings
- Never force-push shared branches (`develop`, `master`, `release/*`), and
  never delete `.github/` or `3rdparty/`.
- Do **not** add docstrings, comments, or type hints to code you did not change.
- Do **not** add error handling for scenarios that cannot happen.
- Do **not** introduce helper abstractions for one-off operations.
- Do **not** commit build artefacts, model weights, ONNX graphs, tokenizers, or
  sample media. Blocked extensions: `*.weights`, `*.onnx`, `*.pt`, `*.pth`,
  `*.bin`, `*.safetensors`, `*.engine`, `*.trt`, `*.mp4`, `*.avi`, `*.mov`,
  `*.mkv`, `*.webm`, `*.jpg`, `*.jpeg`, `*.png`, `*.tiff`; blocked names:
  `vocab.json`, `merges.txt`, `labels.txt`, `tokenizer.json`, `model.weights`,
  `model.onnx`. Small graphics under `docs/` and `README*` are allowed.

---

## Architecture

### Two usage patterns

1. **Direct preprocessor/postprocessor** — maximum flexibility, use individual
   classes directly.
2. **`TaskInterface` / `TaskFactory`** — unified interface;
   `TaskFactory::createTaskInstance(model_type_string, model_info)` returns a
   `std::unique_ptr<TaskInterface>` that handles both pre- and postprocessing.
   Task creation is a **built-in, compile-time registry** in `task_factory.cpp`.
   Third-party or runtime task plugins are **out of scope** unless explicitly
   added as a product requirement (use a separate extension registry, not an
   ever-growing internal table).

### Core abstractions (`include/neuriplo/tasks/core/`)

| File                  | Purpose |
|-----------------------|---------|
| `task_interface.hpp`  | Abstract base: `preprocess(vector<Image>) -> vector<vector<uint8_t>>`, `postprocess(vision::Size, vector<Tensor>) -> vector<Result>` |
| `task_factory.hpp`    | `TaskFactory::createTaskInstance(string, ModelInfo)` — normalises the model-type string (strip `-`, `_`, whitespace; lowercase) and dispatches |
| `result_types.hpp`    | `Result` variant plus optional `visitResult()` helper (forwards to `std::visit`); OpenCV-free (`BoundingBox`, `ImageMatrix`) |
| `bounding_box.hpp`    | Pixel-space `BoundingBox` replacing `cv::Rect` in public result types |
| `image_matrix.hpp`    | Opaque `ImageMatrix` replacing `cv::Mat` in public result types |
| `vision/opencv_adapter.hpp` | Optional OpenCV conversions; available only through `vision-opencv` |
| `vision/stb_io.hpp` | Optional file load/save helpers; available only through `vision-stb` |
| `model_info.hpp`      | `ModelInfo`: `input_shapes`, `input_formats` (`FORMAT_NCHW` / `FORMAT_NHWC`), `input_names`, `output_names` |
| `batch_types.hpp`     | `BatchRequest`, `BatchPreprocessOutput`, `BatchPostprocessOutput`, invariant helpers |
| `batch_preprocess.hpp`| `batchPreprocess(task, BatchRequest)` — per-image preprocess + `batch_size` metadata |
| `batch_postprocess.hpp`| `batchPostprocess(task, frame_size, tensors, batch_size)` — postprocess + batch alignment |
| `task_pipeline.hpp` | `TaskPipeline` / `SequentialTaskPipeline` — explicit `Result` stage composition for multi-task flows |
| `preprocessor.hpp`    | Base preprocessor utilities |
| `bbox_processor.hpp`  | Bounding-box coordinate transformations |

### Task modules

Each domain has a `*_task.cpp` (implements `TaskInterface`) plus one or more
`*_postprocessor.cpp` files. Postprocessors are internal to the task — only
task headers are needed by consumers.

| Domain                | Task class                   | Postprocessors                                                        |
|-----------------------|------------------------------|-----------------------------------------------------------------------|
| Object Detection      | `ObjectDetectionTask`        | `YoloPostprocessor`, `RtDetrPostprocessor`, `RfDetrPostprocessor`     |
| Instance Segmentation | `InstanceSegmentationTask`   | `YoloSegmentationPostprocessor`, `RfDetrSegmentationPostprocessor`    |
| Classification        | `ClassificationTask`         | `TorchvisionPostprocessor`, `VitPostprocessor`, `TensorflowPostprocessor` |
| Video Classification  | `VideoClassificationTask`    | `VideoClassificationPostprocessor`                                    |
| Optical Flow          | `OpticalFlowTask`            | `RaftPostprocessor`                                                   |
| Pose Estimation       | `PoseEstimationTask`         | `VitPosePostprocessor`, `YoloPosePostprocessor`                       |
| Depth Estimation      | `DepthEstimationTask`        | `DepthAnythingV2Postprocessor`                                        |
| Open-Vocabulary Detection | `OpenVocabDetectionTask` | `OwlV2Postprocessor`, `GroundingDinoPostprocessor`                    |
| Gaussian Splatting    | `GaussianSplattingTask`      | `LgmPostprocessor`                                                    |
| Image Understanding   | `ImageUnderstandingTask`     | _(none — response decoded directly from float-encoded bytes)_         |

### `Tensor` type

```cpp
struct Tensor {
    std::vector<TensorElement> data;  // variant<float, int32_t, int64_t, uint8_t>
    std::vector<int64_t> shape;
};
```

### `TaskFactory` routing

`normalizeModelType()` strips `-`, `_`, whitespace and lowercases. Rules are
checked in order:

1. Segmentation before generic YOLO: `"yoloseg"`, `"rfdetrseg"`, `yolo*seg*`
1b. EdgeCrafter segmentation: `ecseg*`, `edgecrafter*seg*` → `InstanceSegmentationTask` (with `EdgeCrafterSegmentationPostprocessor`)
2. YOLO pose: `yolo*pose*` → `PoseEstimationTask` (with `YoloPosePostprocessor`)
2b. EdgeCrafter pose: `ecpose*`, `edgecrafter*pose*` → `PoseEstimationTask` (with `EdgeCrafterPosePostprocessor`)
3. YOLO prefix: any `yolo*` → `ObjectDetectionTask`
4. `"rtdetr"`, `"rfdetr"` → `ObjectDetectionTask`
4b. EdgeCrafter detection: `ecdet*`, `edgecrafter*` → `ObjectDetectionTask` (with `EdgeCrafterPostprocessor`)
5. `"owlv2"`, `"owlvit"`, `"groundingdino"` → `OpenVocabDetectionTask`
6. `"lgm"`, `"grm"`, `*splat*` → `GaussianSplattingTask`
7. `"torchvisionclassifier"`, `"vitclassifier"`, `"tensorflowclassifier"`, `resnet*`, `*tensorflow*` → `ClassificationTask`
8. `*depthanythingv2*` → `DepthEstimationTask`
9. `"videomae"`, `"vivit"`, `"timesformer"` → `VideoClassificationTask`
10. `"raft"` → `OpticalFlowTask`
11. `"vitpose"` → `PoseEstimationTask`
12. `"gemma4"`, `"imageunderstanding"` → `ImageUnderstandingTask`

### `export/` directory

Python scripts for exporting models to ONNX. Each subdirectory corresponds to
a task domain and includes its own README. Not part of the C++ library.

### `3rdparty/` directory

Vendored third-party code (RT-DETRv4 PyTorch implementation). Not compiled
into the library. Do not edit files under `3rdparty/` to fix lint — suppress
them in the tool config instead.

---

## CI

`.github/workflows/lint.yml` runs six jobs in parallel on every push/PR:

1. **format-check** — `clang-format-21 --dry-run --Werror` on `src include tests`
2. **clang-tidy** — `clang-tidy-18 -p build` on every `src/**/*.cpp`
3. **cppcheck** — `cppcheck --enable=warning --std=c++17 --error-exitcode=1 -I include src/`
4. **build-and-test** — `cmake -DBUILD_TESTS=ON`, `cmake --build`, `ctest --output-on-failure`
5. **valgrind** — Debug test build, every `build-valgrind/tests/test_*` binary under Valgrind
6. **build-warnings** — `cmake -DWERROR=ON`, `cmake --build` (no tests)

If any job is red, the push is broken.

---

## CI failure playbook

Operational catalogue for every failure class this repo's CI has actually hit.
It is tool-agnostic: read the matching section, apply the catalogued fix, and
verify with the listed local command.

### Triage workflow

1. **Fetch the failing log:**
   ```bash
   gh run list --branch "$(git branch --show-current)" --limit 5
   gh run view <run-id> --log-failed
   ```
   If a log is pasted, use it directly. If `gh` is unauthenticated, stop and ask
   for `gh auth login` instead of guessing.
2. **Classify the earliest non-cascade error** against the bucket table below.
   Later failures are usually consequences of the first.
3. **Read the matching section before editing.** Use the catalogued fix when one
   exists; propose a novel fix only for a genuinely new failure class, and flag
   that this playbook should be updated.
4. **Apply the minimum change.** Never broaden suppressions, disable a check, or
   turn off `-DWERROR=ON` to get green.
5. **Verify locally** with the section command, then the one local gate command
   at the end of this file.

| Signal in log | Bucket |
|---|---|
| `clang-format-21 --dry-run --Werror` reports a diff | Format |
| `clang-tidy-18` diagnostic on `src/**/*.cpp` | clang-tidy |
| `cppcheck` error/warning with `--error-exitcode=1` | cppcheck |
| `CMake Error`, missing package, FetchContent failure | CMake configure |
| compiler `error:` during `cmake --build` | Build |
| compiler `warning:` promoted by `-DWERROR=ON` | Werror |
| `ctest` failure / GoogleTest assertion | ctest |
| `undefined reference to cv::...` | OpenCV |
| GoogleTest `FetchContent` failure | FetchContent |

### Format

**Reproduce / verify:**
```bash
find src include tests -name '*.cpp' -o -name '*.hpp' | \
  xargs clang-format-21 --dry-run --Werror
```

**Fix:**
```bash
find src include tests -name '*.cpp' -o -name '*.hpp' | \
  xargs clang-format-21 -i
```

There is no judgment call. Never add `// clang-format off` around long lines —
break at a comma or paren. Never edit `.clang-format` to make one file pass.

### clang-tidy

**Reproduce / verify:**
```bash
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
find src -name '*.cpp' | xargs clang-tidy-18 -p build
```

Checks are configured in `.clang-tidy` (`bugprone-*`, `cppcoreguidelines-*`,
`modernize-*`, `performance-*`, `readability-*`, `clang-analyzer-*`).
`HeaderFilterRegex` covers only `src/` and `include/`.

**Auto-fix first, then hand-fix the residue:**
```bash
find src -name '*.cpp' | xargs clang-tidy-18 -p build --fix --fix-errors
find src -name '*.cpp' | xargs clang-tidy-18 -p build
```

| Check | Fix |
|---|---|
| `modernize-use-nullptr` | `NULL` / `0` → `nullptr` (auto-fixable). |
| `modernize-use-override` | Add `override` to virtual overrides (auto-fixable). |
| `modernize-use-using` | `typedef X Y;` → `using Y = X;` (auto-fixable). |
| `modernize-loop-convert` | Rewrite the index loop as a range-`for` (usually auto-fixable). |
| `modernize-use-equals-default` | `Foo() {}` → `Foo() = default;` (auto-fixable). |
| `readability-implicit-bool-conversion` | `if (ptr)` → `if (ptr != nullptr)`; `if (n)` → `if (n != 0)`. |
| `readability-else-after-return` | Drop the `else`; the `if` body returns. |
| `readability-braces-around-statements` | Add `{ ... }` to one-line bodies. |
| `readability-container-size-empty` | `v.size() == 0` → `v.empty()`. |
| `bugprone-narrowing-conversions` | Add an explicit `static_cast<T>(...)`; never silence. |
| `bugprone-unused-return-value` | Use the value or cast to `(void)`. |
| `cppcoreguidelines-init-variables` | `int x;` → `int x = 0;` (same for `bool`, `float`, pointers). |
| `cppcoreguidelines-pro-type-member-init` | Initialise in the ctor init list or at declaration. |
| `cppcoreguidelines-explicit-virtual-functions` | Add `override` (pairs with `modernize-use-override`). |
| `performance-unnecessary-copy-initialization` | `auto x = container[i];` → `const auto& x = container[i];`. |
| `performance-for-range-copy` | `for (auto x : v)` → `for (const auto& x : v)`. |
| `clang-analyzer-core.NullDereference` | Real bug — add the null check; never suppress. |

Use `// NOLINTNEXTLINE(check-name)` with a one-line justification, never bare
`// NOLINT` or multi-function `NOLINTBEGIN`/`NOLINTEND`. Never disable a whole
check in `.clang-tidy` to get green; surface a genuinely wrong check to the user
first. If clang-tidy reports a missing adapter header, configure the tidy build
with that adapter option enabled; keep a separate core-only build job for
dependency-free coverage. Keep a lint fix lint-only — do not add files, modules,
or tests in the same change; hand build-system failures to a separate CMake fix.

### cppcheck

**Reproduce / verify:**
```bash
cppcheck --enable=warning --std=c++17 \
  --suppress=missingIncludeSystem \
  --suppress=unmatchedSuppression \
  --error-exitcode=1 \
  -I include src/
```

| Message id | Fix |
|---|---|
| `uninitMemberVar` | Initialise in the ctor init list or at declaration (default member init is preferred). |
| `noExplicitConstructor` | Add `explicit` to single-argument constructors. |
| `passedByValue` | Change the parameter to `const T&`. |
| `constVariable` / `constParameter` | Add `const`. |
| `unusedFunction` | Delete it, or if public API, suppress with `// cppcheck-suppress unusedFunction` in the header. |
| `memleak` / `resourceLeak` | Real bug — fix ownership; prefer RAII over manual `delete`. |
| `danglingLifetime` | Real bug — fix the lifetime; never suppress. |

Do not broaden suppressions in `.pre-commit-config.yaml` or `lint.yml`. A
targeted inline suppression with a reason is acceptable for a false positive.

### Werror (`-DWERROR=ON`)

**Reproduce / verify:**
```bash
rm -rf build-werror
cmake -S . -B build-werror -DWERROR=ON
cmake --build build-werror --parallel
```

| Warning | Fix |
|---|---|
| `-Wunused-variable` / `-Wunused-parameter` | Remove it, or prefix an interface-mandated parameter with `_`. |
| `-Wunused-but-set-variable` | The assignment is dead — delete it. |
| `-Wsign-compare` | `static_cast<size_t>(...)` on the signed side, or match the loop-index type. |
| `-Wreturn-type` | Fix the control flow — real bug. |
| `-Wshadow` | Rename the inner variable; do not disable `-Wshadow` globally. |
| `-Wdeprecated-declarations` on OpenCV | Use the new `cv::` API; targeted `#pragma` plus TODO only if genuinely transitional. |
| `-Wconversion` / `-Wnarrowing` | Add an explicit `static_cast`. |
| `-Wreorder` | Reorder the ctor init list to match member declaration order. |
| `-Wunused-result` on `fread`/`fwrite` | Capture and check, or cast to `(void)` with a reason. |

Never disable `-Werror`. Never sprinkle `#pragma GCC diagnostic ignored`; scope a
targeted `target_compile_options(tgt PRIVATE -Wno-<name>)` to one target and
justify it in the diff.

MSVC `/WX` note: `std::transform(..., ::tolower)` fails C4244 because the
callable returns `int`. Use an unsigned-char lambda:
```cpp
[](unsigned char character) { return static_cast<char>(std::tolower(character)); }
```

### CMake configure

**Reproduce / verify:**
```bash
rm -rf build
cmake -S . -B build -DBUILD_TESTS=ON -DWERROR=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

| Symptom | Cause / fix |
|---|---|
| `Could NOT find OpenCV` | Runner missing `libopencv-dev`; add it to the apt install in `lint.yml`. Locally: `sudo apt-get install libopencv-dev`. |
| FetchContent hang / GoogleTest network error | Stale `build/_deps/`; delete `build/` and reconfigure. If recurring, pin `GIT_TAG` to a release tag. |
| `add_subdirectory given source ... not an existing directory` | Typo after a module rename. |
| `target_link_libraries called with non-compatible type: UTILITY` | Linking against a non-library target. |
| Source directory does not contain `CMakeLists.txt` | Wrong working directory or a FetchContent clone failed. |
| `fatal error: neuriplo/tasks/core/xxx.hpp: No such file` | New header missing from `target_include_directories` or not installed. |
| `add_test` runs 0 tests | New test file missing `add_executable` + `gtest_discover_tests` in `tests/CMakeLists.txt`. |
| Consumer cannot find the `neuriplo-tasks::` target | Missing `export()`/alias target in the root `CMakeLists.txt`. |
| `-Werror=...` during `cmake --build` | A source warning slipped in — fix the source; touch CMake only if the flag itself is new. |

If the symptom is not in this table, stop and ask before editing CMake. Edit
`CMakeLists.txt` and `cmake/` only; do not patch `src/` to work around a
build-system error, and do not change CMake to hide a source warning. Prefer
`target_*` commands with correct `PRIVATE`/`PUBLIC`/`INTERFACE` scope; never
`link_directories()`; never `file(GLOB)` source lists; never silently bump a
FetchContent `GIT_TAG`. Never commit `build/`, `_deps/`, or
`compile_commands.json`.

### Build

Read the compiler error before rearranging includes; C++ errors usually say
exactly what is wrong.

| Symptom | Fix |
|---|---|
| `fatal error: opencv2/...: No such file or directory` | Add `${OpenCV_INCLUDE_DIRS}` / `${OpenCV_LIBS}` to the target. |
| `use of undeclared identifier 'cv::...'` | Add the missing `<opencv2/...>` include. |
| `undefined reference to cv::...` at link | Add `${OpenCV_LIBS}` to `target_link_libraries`. |
| `static assertion failed: ... std::variant` | `Result` variant mismatch — a postprocessor returns a type missing from `result_types.hpp`. |
| `no matching function for call to 'Tensor'` | `Tensor` constructor signature changed; update the call site. |

### ctest

Run the single failing test with verbose output **before** changing code:
```bash
ctest --test-dir build -R "^test_yolo_postprocessor$" --output-on-failure -V
./build/tests/test_yolo_postprocessor --gtest_filter=<Suite.Test>
```

Common causes: `Result` variant reordered (use `std::get<Type>`, not
`std::get<N>`), postprocessor output shape changed, `EXPECT_EQ` used on floats
(use `EXPECT_NEAR` / `EXPECT_FLOAT_EQ`), or missing test fixtures. Never "fix" a
test by deleting the assertion.

### Valgrind

**Reproduce / verify:**
```bash
cmake -S . -B build-valgrind -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
cmake --build build-valgrind --parallel
for test_bin in build-valgrind/tests/test_*; do
  [ -x "$test_bin" ] || continue
  valgrind --error-exitcode=1 --leak-check=full \
    --show-leak-kinds=definite,indirect \
    --errors-for-leak-kinds=definite,indirect \
    --track-origins=yes --num-callers=25 "$test_bin"
done
```

Invalid read/write or use-after-free: fix ownership/lifetime and add or tighten
the regression test. `definitely lost` / `indirectly lost`: free the owning
allocation or replace manual ownership with RAII. Use a narrow suppression only
after proving the report is outside `neuriplo-tasks` code; never hide real leaks.

### OpenCV

```bash
dpkg -l | grep libopencv-dev
pkg-config --modversion opencv4
```

| Symptom | Fix |
|---|---|
| `undefined reference to cv::dnn::...` | Link `${OpenCV_LIBS}` including `opencv_dnn`; check the `find_package` components. |
| `cv::Exception: OpenCV(4.x) error: (-215)` | Preconditions violated (wrong type, empty matrix, shape mismatch) — fix the caller. |
| Runner/local version mismatch | Compare `pkg-config --modversion opencv4`; pin or upgrade the apt source. |

Do not wrap OpenCV calls in `try/catch(cv::Exception)` to silence test failures.

### FetchContent / GoogleTest

1. Delete `build/_deps/` and retry.
2. If the remote is flaky, cache more aggressively in `lint.yml`, keyed on the
   CMake `GIT_TAG`.
3. Never pin `GIT_TAG` to `main`; always a release tag.

### When to update this playbook

If you fix a CI failure whose cause is not in any section above, add a section
at the point of fix with: the exact error line (so it can be grepped), the root
cause in 1–3 sentences, the fix as a command or code pattern, and the local
reproduce command. A playbook that is not maintained is worse than none.

---

## Rules enforced by CI — do not skip locally

### act CI emulation — which jobs run where

The pre-push hook does **not** run the full `lint.yml` via `act` because two
job categories consistently fail in local Docker:

| Job | Why it fails locally | How it's covered |
|-----|----------------------|-----------------|
| `format-check` | `clang-format-21` not installed in the act container image | Run locally by the pre-push hook before calling act |
| `clang-tidy`, `build-and-test`, `build-warnings` | `ccache-action` requires GitHub auth (token) inside Docker | Validated by real GitHub CI after push |
| `cppcheck` | Works fine in act | Run via act in both pre-commit and pre-push hooks |
| `valgrind` | Runtime job is intentionally slow for local hooks | Run locally on demand or validated by real GitHub CI after push |

**Rule**: never expand the pre-push hook to `act push -W .github/workflows/lint.yml`
(no `-j` filter) — that runs all jobs and will always fail locally on the
ccache-auth, clang-format-21, and runtime-job issues above. Only add a job to
the hook when you have verified it works inside the act container without
network access.

---

### clang-format before every commit

Run `clang-format-21 -i` on every `.cpp`/`.hpp` you touch before staging.
CI runs `clang-format-21 --dry-run --Werror` on all files under `src/`,
`include/`, and `tests/`. A single misformatted line fails the push.

```bash
# Format all changed files
git diff --name-only HEAD | grep -E '\.(cpp|hpp|h)$' | xargs clang-format-21 -i

# Or format all source files at once
find src include tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-21 -i

# Verify (same command CI runs)
find src include tests -name '*.cpp' -o -name '*.hpp' | \
  xargs clang-format-21 --dry-run --Werror
```

### Hyperlink verification

When editing `README.md` or any documentation with hyperlinks:
- Verify all relative links resolve to existing files in the repo (`ls <path>`).
- Verify absolute GitHub URLs are reachable (use `curl -sI <url>` or a quick fetch).
- Prefer absolute GitHub blob/tree URLs over fragile cross-repo relative paths (e.g. `../../../neuriplo/docs/foo.md`).

### Update README.md when adding a new task type

When adding a new task type to `TaskFactory`, **always update ALL of the following in `README.md`**:

1. **`## Features` bullet list** — add a line for the new task (same section, lines ~10–22).
2. **`<!-- TASKFACTORY_MODEL_LIST:START/END -->` block** — add recognized type string(s), input/output contract, and backend/dependency requirements.
3. **`export/` directory** — add a matching `export/<task_domain>/` subdirectory with a setup/download guide; update `export/README.md` directory tree and reference links (use absolute GitHub URLs so they survive being synced into other repos).

Failing to update any of these makes the task invisible to consumers and agents inspecting the library without reading source code.

---

## Skipping CI for docs-only commits

CI workflows have `paths-ignore` for `**.md`, `export/**`, and `docs/**` — pure documentation pushes skip CI automatically.

For mixed commits (docs + code) where CI is still unnecessary, add `[skip ci]` to the commit message subject line.

## The one local gate command

Run this manually before pushing anything non-trivial — CI will fail on the
same checks:

```bash
find src include tests -name '*.cpp' -o -name '*.hpp' | \
  xargs clang-format-21 --dry-run --Werror && \
cppcheck --enable=warning --std=c++17 \
  --suppress=missingIncludeSystem \
  --suppress=unmatchedSuppression \
  --suppress=*:3rdparty/stb/* \
  --error-exitcode=1 -I include src/ && \
cmake -S . -B build -DBUILD_TESTS=ON -DWERROR=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
cmake --build build --parallel && \
ctest --test-dir build --output-on-failure && \
cmake -S . -B build-valgrind -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON && \
cmake --build build-valgrind --parallel && \
for test_bin in build-valgrind/tests/test_*; do \
  [ -x "$test_bin" ] || continue; \
  valgrind --error-exitcode=1 --leak-check=full \
    --show-leak-kinds=definite,indirect \
    --errors-for-leak-kinds=definite,indirect \
    --track-origins=yes --num-callers=25 "$test_bin"; \
done
```

If this is green locally, CI will be green.
