# Task Refactor Atomic Plan

Goal: improve control over task creation, preprocessing, postprocessing, result handling, and multi-task composition without changing public task contracts in one large step.

Non-goals:

- Do not change `TaskInterface` behavior.
- Do not change `Result` variant schema.
- Do not change tensor shape, dtype, or model I/O contracts.
- Do not rewrite every task in one pass.
- Do not add runtime dependencies beyond OpenCV.

Guardrails:

- Keep `TaskFactory::createTaskInstance(model_type, model_info)` source-compatible.
- Preserve existing model-type normalization rules.
- Preserve README model list behavior tested by `tests/test_readme_model_types.cpp`.
- Add focused tests before or with each behavior move.
- Keep each phase mergeable on its own.

## Phase 0: Baseline And Safety Net

Purpose: lock current behavior before refactor.

Steps:

1. List every supported model type from `README.md` and `tests/test_task_factory.cpp`.
2. Add missing factory coverage for aliases that exist in README but lack tests.
3. Add regression tests for normalized names with hyphen, underscore, whitespace, and case differences.
4. Add one test per task domain proving factory returns expected concrete task family.
5. Run format, build, and tests.

Files likely touched:

- `tests/test_task_factory.cpp`
- `tests/test_readme_model_types.cpp`

Verification:

```bash
cmake -S . -B build -DBUILD_TESTS=ON -DWERROR=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Stop criteria:

- Tests describe current behavior.
- No production code changed except needed test-only access helpers.

## Phase 1: Extract Factory Registry Without Behavior Change

Purpose: replace hard-coded creation blocks with data-driven creators while preserving rule order.

Steps:

1. Add internal creator type in `src/core/task_factory.cpp`.
2. Move each current routing rule into ordered registry entries.
3. Keep `normalizeModelType()` behavior unchanged.
4. Keep first-match semantics unchanged.
5. Keep all constructors and task headers unchanged.
6. Add tests proving segmentation and pose aliases still match before generic YOLO detection.

Suggested internal shape:

```cpp
using TaskCreator = std::function<std::unique_ptr<TaskInterface>(const std::string&, const ModelInfo&)>;

struct TaskRegistration {
    std::string name;
    std::function<bool(const std::string&)> matches;
    TaskCreator create;
};
```

Files likely touched:

- `src/core/task_factory.cpp`
- `tests/test_task_factory.cpp`

Verification:

```bash
./build/tests/test_task_factory
ctest --test-dir build --output-on-failure
```

Stop criteria:

- Public factory API unchanged.
- Registry order visible in one local function.
- All old model strings still pass.

## Phase 2: Add Optional Factory Extension Point

Purpose: let future tasks register cleanly without editing one long dispatch function.

Steps:

1. Add private/internal registry accessor.
2. Add `registerTaskType()` only if needed by tests or external consumers.
3. Prefer compile-time static registry if no runtime extension requirement exists.
4. Add duplicate-name and null-creator guard only for states that can occur.
5. Document registration order contract in code only if public extension is added.

Files likely touched:

- `include/neuriplo/tasks/core/task_factory.hpp`
- `src/core/task_factory.cpp`
- `tests/test_task_factory.cpp`

Verification:

```bash
./build/tests/test_task_factory
```

Stop criteria:

- New task addition needs one local registry entry.
- Existing `createTaskInstance()` callers unchanged.

## Phase 3: Introduce Strategy Interfaces In One Domain

Purpose: prove strategy split on lowest-risk repeated behavior before broad migration.

Recommended first domain: object detection, because it already has YOLO, RT-DETR, RF-DETR, and EdgeCrafter variation.

Steps:

1. Identify duplicated preprocessing and postprocessing knobs in object detection only.
2. Introduce internal strategy interface only if it removes real duplication.
3. Move one variation behind strategy, keeping public task constructors unchanged.
4. Add tests comparing old and new outputs on existing synthetic tensors.
5. Avoid changing tensor interpretation or bbox math.

Candidate interfaces:

```cpp
class DetectionPreprocessStrategy {
public:
    virtual ~DetectionPreprocessStrategy() = default;
    virtual std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& images) = 0;
};

class DetectionPostprocessStrategy {
public:
    virtual ~DetectionPostprocessStrategy() = default;
    virtual std::vector<Result> postprocess(const cv::Size& original_size, const std::vector<Tensor>& outputs) = 0;
};
```

Files likely touched:

- `src/object_detection/object_detection_task.cpp`
- `include/neuriplo/tasks/object_detection/object_detection_task.hpp`
- Existing object detection postprocessors only when duplication is proven.
- `tests/test_yolo_postprocessor.cpp`
- `tests/test_rtdetr_postprocessor.cpp`
- `tests/test_edgecrafter.cpp`

Verification:

```bash
./build/tests/test_task_factory
./build/tests/test_yolo_postprocessor
./build/tests/test_rtdetr_postprocessor
./build/tests/test_edgecrafter
```

Stop criteria:

- One domain uses strategies internally.
- Existing direct postprocessor APIs still work.
- No other domain touched.

## Phase 4: Promote Shared Task Lifecycle Only After Duplication Is Measured

Purpose: add Template Method only if repeated lifecycle code exists across tasks.

Steps:

1. Compare task implementations for repeated `preprocess`, output validation, and `postprocess` flow.
2. Add `BaseTask` only if at least two domains can use it without semantic changes.
3. Keep `TaskInterface` unchanged.
4. Make shared lifecycle final only where override freedom is not needed.
5. Migrate one task first.
6. Add tests for empty image input, invalid output count, and expected result conversion in migrated task.

Candidate shape:

```cpp
class BaseTask : public TaskInterface {
public:
    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& images) override;
    std::vector<Result> postprocess(const cv::Size& original_size, const std::vector<Tensor>& outputs) override;

protected:
    virtual void validateOutputs(const std::vector<Tensor>& outputs) const = 0;
    virtual std::vector<Result> decode(const cv::Size& original_size, const std::vector<Tensor>& outputs) = 0;
};
```

Files likely touched:

- `include/neuriplo/tasks/core/task_interface.hpp` only if necessary.
- New `include/neuriplo/tasks/core/base_task.hpp`
- New `src/core/base_task.cpp`
- One task domain implementation.
- Matching task tests.

Verification:

```bash
./build/tests/test_task_interface
ctest --test-dir build --output-on-failure
```

Stop criteria:

- At least one migrated task has simpler code.
- No public lifecycle behavior changed.
- No abstract base exists without immediate users.

## Phase 5: Repeat Strategy Migration Domain By Domain

Status: implemented for current high-value scope (detection, instance segmentation, pose); remaining domains stay optional unless duplication appears.

Purpose: make preprocessing and postprocessing control explicit without broad churn.

Order:

1. Object detection.
2. Instance segmentation.
3. Pose estimation.
4. Classification.
5. Depth estimation.
6. Open-vocabulary detection.
7. Image understanding.
8. Gaussian splatting.
9. Video classification and optical flow only if duplication exists.

Per-domain checklist:

1. Add or reuse strategy only when it removes duplication or isolates model-specific behavior.
2. Keep direct preprocessor and postprocessor headers source-compatible.
3. Add tests using current synthetic inputs.
4. Run domain test binary plus full `ctest`.
5. Update README only if recognized model strings or contracts change.

Stop criteria:

- Each domain lands as separate commit or PR.
- No cross-domain edits unless shared interface already exists.

## Phase 6: Add Result Visitor Helpers

Purpose: reduce repeated consumer-side `std::visit` boilerplate without changing `Result`.

Steps:

1. Search for repeated result handling in tests and examples.
2. Add visitor helpers only if repetition is real.
3. Keep `Result = std::variant<...>` unchanged.
4. Provide helpers as optional utilities, not required dispatch path.
5. Add tests proving every result alternative is accepted.

Candidate shape:

```cpp
template <typename Visitor>
decltype(auto) visitResult(Result& result, Visitor&& visitor);

template <typename Visitor>
decltype(auto) visitResult(const Result& result, Visitor&& visitor);
```

Files likely touched:

- `include/neuriplo/tasks/core/result_types.hpp`
- `tests/test_result_types.cpp`

Verification:

```bash
./build/tests/test_result_types
```

Stop criteria:

- Result schema unchanged.
- Helper removes caller boilerplate.
- No task implementation depends on visitor unless useful.

## Phase 7: Add Composite Pipelines As Separate API

Status: implemented via `task_pipeline.hpp`, `task_pipeline.cpp`, and `test_task_pipeline.cpp`.

Purpose: support multi-task workflows without overloading single-model task contracts.

Initial target:

- Detection + pose.
- Detection + segmentation.

Steps:

1. Define pipeline API separately from `TaskInterface`.
2. Keep each task independently runnable.
3. Pass intermediate results explicitly.
4. Avoid hiding inference engine boundary inside `neuriplo-tasks`.
5. Add tests with mocked task outputs.

Candidate shape:

```cpp
class TaskPipeline {
public:
    virtual ~TaskPipeline() = default;
    virtual std::vector<Result> run(const std::vector<Result>& inputs) = 0;
};
```

Files likely touched:

- New `include/neuriplo/tasks/core/task_pipeline.hpp`
- New `src/core/task_pipeline.cpp`
- New pipeline tests.

Verification:

```bash
ctest --test-dir build --output-on-failure
```

Stop criteria:

- Composite API does not change `TaskInterface`.
- Pipeline stages remain inspectable and testable.

## Phase 8: Cleanup And Documentation Sync

Purpose: make refactor visible and keep consumers aligned.

Steps:

1. Remove obsolete helper functions only after all users migrate.
2. Keep compatibility wrappers for one release if public API changed.
3. Update README examples only after final API shape is stable.
4. Update `REPO_META.yaml` only if entrypoints or owned paths change.
5. Add migration notes for downstream consumers if public headers changed.

Files likely touched:

- `README.md`
- `docs/Versioning.md`
- `REPO_META.yaml` only if needed.

Verification:

```bash
find src include tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror
cppcheck --enable=warning --std=c++17 --suppress=missingIncludeSystem --suppress=unmatchedSuppression --error-exitcode=1 -I include src/
cmake -S . -B build -DBUILD_TESTS=ON -DWERROR=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Stop criteria:

- Docs match public API.
- Local gate passes.
- Downstream contract changes are explicit.

## Atomic Commit Sequence

Recommended commit boundaries:

1. `test: lock task factory routing behavior`
2. `refactor: move task factory routing to registry`
3. `test: cover factory registry ordering`
4. `refactor: isolate detection postprocess strategy`
5. `refactor: isolate detection preprocess strategy`
6. `refactor: add base task lifecycle for one domain`
7. `refactor: migrate segmentation strategies`
8. `refactor: migrate pose strategies`
9. `feat: add optional result visitor helpers`
10. `feat: add task pipeline API`
11. `docs: sync task refactor notes`

Each commit must:

- Build alone.
- Pass relevant test binary.
- Avoid unrelated formatting churn.
- Preserve existing public behavior unless commit message says otherwise.

## Decision Checklist Before Each Phase

Ask:

1. Does this phase reduce real duplication or isolate real variation?
2. Can this phase be tested with current synthetic tensors?
3. Can this phase land without changing model I/O contracts?
4. Can this phase be reverted independently?
5. Does README need update because supported type strings or contracts changed?

If answer to 2, 3, or 4 is no, split phase smaller before editing code.
