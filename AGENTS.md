# neuriplo-tasks — Agent & Contributor Instructions

This file is the **single source of truth** for coding conventions, tooling, and
rules for working in `neuriplo-tasks`. All other AI agent config files
(`CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md`) defer to this
file.

## Repo metadata

- Machine-readable entrypoints and owned paths live in
  [`REPO_META.yaml`](./REPO_META.yaml). Prefer it as the source for build/test
  commands and allowed change classes when automating.
- **Branches**: `develop` is the integration branch for normal work; `master`
  is release-only.
- **Releases must align with tags**: Every Git tag (e.g., `v0.4.1`) must have
  a corresponding GitHub Release. Never push a tag without also creating the
  release via `gh release create`. Release notes must come from `CHANGELOG.md`
  — never use `--generate-notes`. If a tag exists without a release, create
  the release immediately. See [`docs/Versioning.md`](./docs/Versioning.md)
  for the full release workflow.
- **Priorities when reviewing a change**: correctness → backward compatibility
  → task-contract stability → shape/dtype assumptions → performance regressions.

---

## Project overview

- **Artifact**: `libneuriplo-tasks.a` — C++17 static library
- **Source roots**: `src/`, `include/neuriplo/tasks/`
- **Tests**: `tests/` (GoogleTest, fetched via CMake `FetchContent`)
- **Only runtime dependency**: OpenCV
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
| `clang-format-18` | Formatting (LLVM base, 4-space, 120 col) | `.clang-format`     |
| `clang-tidy-18`   | Static analysis on `src/` + `include/`   | `.clang-tidy`       |
| `cppcheck`        | Additional static analysis               | `.pre-commit-config.yaml` |
| `ctest` / GoogleTest | Unit tests                            | `tests/CMakeLists.txt` |
| `valgrind`        | Runtime memory/leak checks on test binaries | `.github/workflows/lint.yml` |
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
   Task creation is a **built-in, compile-time registry** in `task_factory.cpp`.
   Third-party or runtime task plugins are **out of scope** unless explicitly
   added as a product requirement (use a separate extension registry, not an
   ever-growing internal table).

### Planned work

Atomic roadmap (batch utilities, refactor phases, composite pipelines):
[`docs/ROADMAP.md`](./docs/ROADMAP.md). Batch readiness audit (B0):
[`docs/batch_support_matrix.md`](./docs/batch_support_matrix.md). Factory/strategy
refactor detail: [`docs/plans/task_refactor_atomic_plan.md`](./docs/plans/task_refactor_atomic_plan.md).

### Core abstractions (`include/neuriplo/tasks/core/`)

| File                  | Purpose |
|-----------------------|---------|
| `task_interface.hpp`  | Abstract base: `preprocess(vector<cv::Mat>) → vector<vector<uint8_t>>`, `postprocess(cv::Size, vector<Tensor>) → vector<Result>` |
| `task_factory.hpp`    | `TaskFactory::createTaskInstance(string, ModelInfo)` — normalises the model-type string (strip `-`, `_`, whitespace; lowercase) and dispatches |
| `result_types.hpp`    | `Result` variant plus optional `visitResult()` helper (forwards to `std::visit`); OpenCV-free (`BoundingBox`, `ImageMatrix`) |
| `bounding_box.hpp`    | Pixel-space `BoundingBox` replacing `cv::Rect` in public result types |
| `image_matrix.hpp`    | Opaque `ImageMatrix` replacing `cv::Mat` in public result types |
| `opencv_interop.hpp`  | `toCvRect` / `fromCvRect` / `toCvMat` / `fromCvMat` conversion at OpenCV boundaries |
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

1. **format-check** — `clang-format-18 --dry-run --Werror` on `src include tests`
2. **clang-tidy** — `clang-tidy-18 -p build` on every `src/**/*.cpp`
3. **cppcheck** — `cppcheck --enable=warning --std=c++17 --error-exitcode=1 -I include src/`
4. **build-and-test** — `cmake -DBUILD_TESTS=ON`, `cmake --build`, `ctest --output-on-failure`
5. **valgrind** — Debug test build, every `build-valgrind/tests/test_*` binary under Valgrind
6. **build-warnings** — `cmake -DWERROR=ON`, `cmake --build` (no tests)

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

## Rules enforced by CI — do not skip locally

### act CI emulation — which jobs run where

The pre-push hook does **not** run the full `lint.yml` via `act` because two
job categories consistently fail in local Docker:

| Job | Why it fails locally | How it's covered |
|-----|----------------------|-----------------|
| `format-check` | `clang-format-18` not installed in the act container image | Run locally by the pre-push hook before calling act |
| `clang-tidy`, `build-and-test`, `build-warnings` | `ccache-action` requires GitHub auth (token) inside Docker | Validated by real GitHub CI after push |
| `cppcheck` | Works fine in act | Run via act in both pre-commit and pre-push hooks |
| `valgrind` | Runtime job is intentionally slow for local hooks | Run locally on demand or validated by real GitHub CI after push |

**Rule**: never expand the pre-push hook to `act push -W .github/workflows/lint.yml`
(no `-j` filter) — that runs all jobs and will always fail locally on the
ccache-auth, clang-format-18, and runtime-job issues above. Only add a job to
the hook when you have verified it works inside the act container without
network access.

---

### clang-format before every commit

**Run `clang-format-18 -i` on every `.cpp`/`.hpp` you touch before staging.**
CI runs `clang-format-18 --dry-run --Werror` on all files under `src/`,
`include/`, and `tests/`. A single misformatted line fails the push.

```bash
# Format all changed files
git diff --name-only HEAD | grep -E '\.(cpp|hpp|h)$' | xargs clang-format-18 -i

# Or format all source files at once
find src include tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 -i

# Verify (same command CI runs)
find src include tests -name '*.cpp' -o -name '*.hpp' | \
  xargs clang-format-18 --dry-run --Werror
```

The `format_on_edit.sh` hook does this automatically for Claude Code edits,
but running it manually is the safest gate before `git commit`.

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
  xargs clang-format-18 --dry-run --Werror && \
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
