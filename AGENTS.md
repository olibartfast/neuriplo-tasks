# vision-core — Agent & Contributor Instructions

This file is the **single source of truth** for coding conventions, tooling, and
rules for working in `vision-core`. All other AI agent config files
(`CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md`) defer to this
file.

## Repo metadata

- Machine-readable entrypoints and owned paths live in
  [`REPO_META.yaml`](./REPO_META.yaml). Prefer it as the source for build/test
  commands and allowed change classes when automating.
- **Branches**: `develop` is the integration branch for normal work; `master`
  is release-only.
- **Priorities when reviewing a change**: correctness → backward compatibility
  → task-contract stability → shape/dtype assumptions → performance regressions.

---

## Project overview

- **Artifact**: `libvision-core.a` — C++17 static library
- **Source roots**: `src/`, `include/vision-core/`
- **Tests**: `tests/` (GoogleTest, fetched via CMake `FetchContent`)
- **Only runtime dependency**: OpenCV
- **Consumers**: [tritonic](https://github.com/olibartfast/tritonic),
  [vision-inference](https://github.com/olibartfast/vision-inference)
- **GitHub repo**: `https://github.com/olibartfast/vision-core`

`vision-core` provides framework-agnostic computer-vision pre/postprocessing
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
```

Pre-commit hooks run clang-format and cppcheck automatically:
`pip install pre-commit && pre-commit install`.

---

## Code quality rules

### Tooling

| Tool           | Purpose                                  | Config              |
|----------------|------------------------------------------|---------------------|
| `clang-format-18` | Formatting (LLVM base, 4-space, 120 col) | `.clang-format`     |
| `clang-tidy-18`   | Static analysis on `src/` + `include/`   | `.clang-tidy`       |
| `cppcheck`        | Additional static analysis               | `.pre-commit-config.yaml` |
| `ctest` / GoogleTest | Unit tests                            | `tests/CMakeLists.txt` |
| `-DWERROR=ON`     | Treat compiler warnings as errors        | `CMakeLists.txt`    |

### Formatting

```bash
# Check (dry-run, CI mode)
find src include tests -name '*.cpp' -o -name '*.hpp' | \
  xargs clang-format-18 --dry-run --Werror

# Apply
find src include tests -name '*.cpp' -o -name '*.hpp' | \
  xargs clang-format-18 -i
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
- Do **not** add docstrings, comments, or type hints to code you did not change.
- Do **not** add error handling for scenarios that cannot happen.
- Do **not** introduce helper abstractions for one-off operations.
- Do **not** commit build artefacts, model weights, or sample media
  (`*.weights`, `*.onnx`, `*.mp4`, `*.jpg`, `vocab.json`, `merges.txt`, etc.).
  The `block_large_binaries.sh` hook enforces this.

---

## Architecture

### Two usage patterns

1. **Direct preprocessor/postprocessor** — maximum flexibility, use individual
   classes directly.
2. **`TaskInterface` / `TaskFactory`** — unified interface;
   `TaskFactory::createTaskInstance(model_type_string, model_info)` returns a
   `std::unique_ptr<TaskInterface>` that handles both pre- and postprocessing.

### Core abstractions (`include/vision-core/core/`)

| File                  | Purpose |
|-----------------------|---------|
| `task_interface.hpp`  | Abstract base: `preprocess(vector<cv::Mat>) → vector<vector<uint8_t>>`, `postprocess(cv::Size, vector<Tensor>) → vector<Result>` |
| `task_factory.hpp`    | `TaskFactory::createTaskInstance(string, ModelInfo)` — normalises the model-type string (strip `-`, `_`, whitespace; lowercase) and dispatches |
| `result_types.hpp`    | `Result = std::variant<Classification, Detection, InstanceSegmentation, OpticalFlow, VideoClassification, PoseEstimation, DepthEstimation>` |
| `model_info.hpp`      | `ModelInfo`: `input_shapes`, `input_formats` (`FORMAT_NCHW` / `FORMAT_NHWC`), `input_names`, `output_names` |
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
2. YOLO pose: `yolo*pose*` → `PoseEstimationTask` (with `YoloPosePostprocessor`)
3. YOLO prefix: any `yolo*` → `ObjectDetectionTask`
4. `"rtdetr"`, `"rfdetr"` → `ObjectDetectionTask`
5. `"torchvisionclassifier"`, `"vitclassifier"`, `"tensorflowclassifier"`, `resnet*`, `*tensorflow*` → `ClassificationTask`
6. `*depthanythingv2*` → `DepthEstimationTask`
7. `"videomae"`, `"vivit"`, `"timesformer"` → `VideoClassificationTask`
8. `"raft"` → `OpticalFlowTask`
9. `"vitpose"` → `PoseEstimationTask`

### `export/` directory

Python scripts for exporting models to ONNX. Each subdirectory corresponds to
a task domain and includes its own README. Not part of the C++ library.

### `3rdparty/` directory

Vendored third-party code (RT-DETRv4 PyTorch implementation). Not compiled
into the library. Do not edit files under `3rdparty/` to fix lint — suppress
them in the tool config instead.

---

## CI

`.github/workflows/lint.yml` runs five jobs in parallel on every push/PR:

1. **format-check** — `clang-format-18 --dry-run --Werror` on `src include tests`
2. **clang-tidy** — `clang-tidy-18 -p build` on every `src/**/*.cpp`
3. **cppcheck** — `cppcheck --enable=warning --std=c++17 --error-exitcode=1 -I include src/`
4. **build-and-test** — `cmake -DBUILD_TESTS=ON`, `cmake --build`, `ctest --output-on-failure`
5. **build-warnings** — `cmake -DWERROR=ON`, `cmake --build` (no tests)

If any job is red, the push is broken.

---

## Agent tooling (Claude Code)

This repo ships a Claude Code kit under `.claude/` that nudges the rules
above during edits. It does **not** gate `git push` — CI is the gate. If
you use another agent (Copilot, Codex, Gemini), the catalogue in
`.claude/skills/ci-guardian/SKILL.md` is still the right reference — point
your tool at it.

### Hooks

| Hook                         | Event                              | What it does |
|------------------------------|------------------------------------|--------------|
| `format_on_edit.sh`          | `PostToolUse(Edit\|Write\|MultiEdit)` | Runs `clang-format-18 -i` on every edited `.cpp`/`.hpp`/`.h` under `src/`, `include/`, `tests/`, then re-checks with `--dry-run --Werror`. Blocks the turn if formatting issues remain. |
| `block_large_binaries.sh`    | `PreToolUse(Bash)`                 | Denies `git add` / `git commit` of model artefacts and sample media: `*.weights`, `*.onnx`, `*.mp4`, `*.jpg`, `*.png` (outside `docs/`), `vocab.json`, `merges.txt`, `labels.txt`. |
| `inject_ci_context.sh`       | `UserPromptSubmit`                 | Prepends a short reminder of the core CI rules (clang-format, clang-tidy, cppcheck, `-DWERROR=ON`, no large binaries) to every prompt. |

### Subagents

| Subagent             | When to invoke |
|----------------------|----------------|
| `@ci-triage`         | A CI run is red. Reads failing logs via `gh`, classifies against the catalogue, produces a minimal fix plan. **Does not edit.** |
| `@clang-tidy-fixer`  | clang-format / clang-tidy / cppcheck errors need mechanical cleanup. Runs `--fix` loops and hand-fixes the residue using the catalogue. |
| `@cmake-triage`      | CMake configure/build errors, FetchContent failures, linker errors, or unexpected `-DWERROR` breakage. Diagnoses and applies the minimum CMakeLists change. |

### Skill

`.claude/skills/ci-guardian/SKILL.md` — the playbook. One section per error
class actually seen in this repo's CI history (clang-format, clang-tidy
checks, cppcheck, `-DWERROR` warnings, CMake configure, ctest, OpenCV
linkage). Every section includes the exact local-reproduce command.

If you fix a CI failure whose cause isn't yet catalogued, **add a section to
the skill** in the same PR. The skill compounds in value — don't let it decay.

---

## The one local gate command

Run this manually before pushing anything non-trivial — CI will fail on the
same checks:

```bash
find src include tests -name '*.cpp' -o -name '*.hpp' | \
  xargs clang-format-18 --dry-run --Werror && \
cppcheck --enable=warning --std=c++17 \
  --suppress=missingIncludeSystem \
  --suppress=unmatchedSuppression \
  --error-exitcode=1 -I include src/ && \
cmake -S . -B build -DBUILD_TESTS=ON -DWERROR=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
cmake --build build --parallel && \
ctest --test-dir build --output-on-failure
```

If this is green locally, CI will be green.
