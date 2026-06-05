# vision-core Roadmap

Atomic execution plan for planned library work. Each step is sized for a
single PR or commit series: build alone, tests green, no unrelated churn.

**Related docs**

| Document | Scope |
|----------|--------|
| [batch_support_matrix.md](./batch_support_matrix.md) | Per-family batch readiness (B0 audit) |
| [batch_processing.md](./batch_processing.md) | Consumer migration guide (B6) |
| [task_refactor_atomic_plan.md](./task_refactor_atomic_plan.md) | Factory registry, strategies, `visitResult`, composite pipelines |
| [Versioning.md](./Versioning.md) | Release and changelog workflow |
| [AGENTS.md](../AGENTS.md) | CI gate, contracts, coding rules |

**Prod today:** built-in tasks at `batch_size = 1`, stable `TaskInterface` /
`Result` schema. Roadmap items below are **enhancements**, not blockers for
shipping inference on documented model types.

---

## Status overview

| Track | Item | Status |
|-------|------|--------|
| Core | `visitResult()` helpers | **Done** (`0.3.x`) |
| Core | TaskFactory compile-time registry | **Done** (Phase 1 refactor) |
| Core | Plugin scope documented | **Done** (README, AGENTS, header) |
| Refactor | Phase 0 — factory/README test baseline | **Done** (contract tests exist) |
| Refactor | Phase 1 — registry extraction | **Done** |
| Refactor | Phase 2 — optional runtime extension | **Deferred** (out of scope unless product requires plugins) |
| Refactor | Phases 3–5 — per-domain strategies / `BaseTask` | **Done** (detection, pose, segmentation strategies; depth + classification `BaseTask` pilots) |
| Refactor | Phase 6 — result visitor helpers | **Done** |
| Refactor | Phase 7 — composite `TaskPipeline` API | **Done** |
| Refactor | Phase 8 — docs/sync cleanup | **Planned** (ongoing per release) |
| Factory | Track D — descriptor registry auditability | **Done** |
| **Batch** | B0 — capability audit (`batch_support_matrix.md`) | **Done** |
| **Batch** | B1 — batch contract types (`batch_types.hpp`) | **Done** |
| **Batch** | B2 — preprocess batch packer (`batch_preprocess`) | **Done** |
| **Batch** | B3 — postprocess batch splitter (`batch_postprocess`) | **Done** |
| **Batch** | B4 — domain adoption (classification, det, depth, pose) | **Done** |
| **Batch** | B5 — integration test & README contract | **Done** |
| **Batch** | B6 — consumer migration guide | **Done** |

Update the **Status** column when a track lands; check README roadmap bullets
against this file on each merge.

---

## Guardrails (all tracks)

- Preserve `TaskInterface` method signatures unless a phase explicitly
  documents a breaking change and version bump.
- Preserve `Result` variant member layout (add alternatives only with semver
  minor/major policy in `docs/Versioning.md`).
- Do not change tensor dtype, channel order, or bbox math without tests and
  consumer migration notes.
- No runtime dependencies beyond OpenCV.
- Third-party / runtime task plugins remain **out of scope** unless product
  requests Phase 2 in [task_refactor_atomic_plan.md](./task_refactor_atomic_plan.md).

**Local gate** (run before every PR):

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

---

## Track A — Task architecture refactor

Full step-by-step phases 0–8 live in
[task_refactor_atomic_plan.md](./task_refactor_atomic_plan.md).

**Next recommended phases** (only when duplication or audit pain is real):

1. Maintain **Phase 4/5** only where future domains show real duplicate lifecycle or model-specific branching.
2. Use **Phase 7** `TaskPipeline` for detection+pose / detection+seg workflows; pipelines compose *task results*, batch utilities compose *images* within one task.
3. Keep **Phase 8** docs/sync cleanup current when public headers or task contracts change.

Do **not** start Phase 2 (runtime plugin registry) without an explicit product
requirement.

---

## Track B — Batch processing utilities

**Goal:** first-class library helpers so consumers can run `N > 1` images through
the same task contract without reimplementing pack/split logic per domain.

**Non-goals**

- Do not embed inference scheduling, GPU queueing, or Triton client code.
- Do not require batching for tasks that are inherently single-frame unless
  `ModelInfo.max_batch_size_ > 1` is set by the consumer.
- Do not change ONNX export scripts in the same PR as C++ utilities unless
  export axis docs must align.

**Current baseline**

- `ModelInfo` already exposes `batch_size_`, `max_batch_size_`, and per-I/O
  `input_batch_sizes` / `output_batch_sizes`.
- `preprocess(vector<cv::Mat>)` returns one buffer per `Mat` (multi-input /
  multi-view, not a unified batched tensor).
- Some postprocessors already iterate the leading batch dimension when tensors
  are batched (e.g. depth, ViT pose, LGM `[N,G,14]`).
- Tests and README contract tests default to `batch_size_ = 1`.

---

### B0 — Batch capability audit ✅

**Purpose:** know which domains are batch-ready before writing shared utilities.

**Deliverable:** [`batch_support_matrix.md`](./batch_support_matrix.md) — summary table,
preprocess/postprocess patterns, `N=1` domains, test gaps, B5 adoption order.

**Stop criteria:** met — every `TaskFactory` family has Ready / Partial / N/A.

**Next:** B1 — `batch_types.hpp` contract header (no task changes).

---

### B1 — Batch contract header ✅

**Deliverable:** `include/vision-core/core/batch_types.hpp` — `BatchRequest`,
`BatchPreprocessOutput`, `BatchPostprocessOutput`, invariant helpers;
`tests/test_batch_types.cpp`.

**Stop criteria:** met — header in `CORE_HEADERS`; no task includes it yet.

**Next:** B3 — `batch_postprocess` helper.

---

### B2 — Preprocess batch packer ✅

**Deliverable:** `batch_preprocess.hpp` / `batch_preprocess.cpp` —
`batchPreprocess()` wraps `TaskInterface::preprocess`, sets `batch_size`, validates
empty batch and `max_batch_size_`; `TaskInterface::getModelInfo()` for limits;
`tests/test_batch_preprocess.cpp` (resnet + yolov8, N=1/N=2).

**Stop criteria:** met — bit-identical to direct preprocess at N=1; N=2 yields two
buffers when `max_batch_size_ >= 2`.

**Next:** B3 — `batch_postprocess` helper.

---

### B3 — Postprocess batch splitter ✅

**Purpose:** map batched inference tensors to `vector<Result>` with stable
per-index ordering.

**Steps**

1. Add `include/vision-core/core/batch_postprocess.hpp`:
   - `BatchPostprocessOutput batchPostprocess(TaskInterface& task, const cv::Size& frame_size, const std::vector<Tensor>& tensors, int batch_size);`
2. Default implementation: call `task.postprocess` once; if result count equals
   `batch_size`, return as-is; if result count is 1 and `batch_size > 1`,
   document duplication policy or delegate to domain override hook.
3. Add **domain overrides** only where existing postprocessors already emit
   multiple results per batch (depth, gaussian splatting): thin wrappers, no
   math changes in first PR.
4. Tests: synthetic tensors with leading dim `N=2` for one Ready domain from
   B0 matrix.

**Files**

- `include/vision-core/core/batch_postprocess.hpp`
- `src/core/batch_postprocess.cpp`
- `tests/test_batch_postprocess.cpp`
- Domain files only when override is required

**Verification**

```bash
./build/tests/test_batch_postprocess
```

**Stop criteria:** met — depth `N = 2` round-trip; `N = 1` matches direct
`postprocess`; Gaussian splatting keeps single aggregate result.

**Next:** B4 — domain adoption (one PR per family).

---

### B4 — Domain adoption ✅

**Purpose:** move from generic helpers to tested batch contracts per task.

**Order** (lowest risk first):

1. Classification
2. Object detection (YOLO family)
3. Depth estimation
4. Pose estimation
5. Remaining families per `batch_support_matrix.md`

**Per-domain steps** (repeat)

1. Confirm preprocess output can be stacked or fed as separate inputs per engine
   contract; adjust only if required.
2. Ensure postprocess returns `batch_size` results or add splitter override.
3. Add `tests/test_<domain>_batch.cpp` with `ModelInfo.batch_size_ = 2`.
4. Update one `export/<domain>/` doc paragraph on batch axis if export supports
   dynamic batch.

**Stop criteria:** met for classification, YOLO detection, depth, and ViTPose —
`test_*_batch.cpp` suites added; `batch_size = 1` regressions unchanged.

**Next:** B5 — integration test and README contract.

---

### B5 — Integration test and README contract ✅

**Purpose:** lock public promise for batch utilities.

**Steps**

1. Add `tests/test_batch_integration.cpp`: factory-create task, `batchPreprocess`
   + `batchPostprocess` for two domains (classification + one detection).
2. Document batch utilities in `README.md` API section (not only roadmap bullet).
3. Check off README roadmap item; add `[Unreleased]` changelog entry under **Added**.
4. If new headers are public entrypoints, note them in `AGENTS.md` core abstractions
   table.

**Files**

- `tests/test_batch_integration.cpp`
- `README.md`
- `CHANGELOG.md`
- `AGENTS.md` (optional one-line table row)

**Verification**

```bash
ctest --test-dir build --output-on-failure
```

**Stop criteria:** met — `test_batch_integration`, README batch API section, CHANGELOG,
AGENTS table rows, roadmap checkbox checked.

**Next:** Track B complete — maintain [batch_support_matrix.md](./batch_support_matrix.md) as domains evolve.

---

### B6 — Consumer migration note ✅

**Purpose:** downstream repos (tritonic, vision-inference) know how to adopt.

**Steps**

1. Add `docs/batch_processing.md`: worked example (N=2 images, ModelInfo fields,
   call sequence, engine responsibility vs library responsibility).
2. Link from README Roadmap section and `export/README.md` if batch axis is
   export-relevant.

**Stop criteria:** met — `docs/batch_processing.md`, README + export/README links,
no model weights in examples.

---

### Track B — Atomic commit sequence

| # | Commit subject (suggested) |
|---|----------------------------|
| 1 | `docs: add batch support matrix` |
| 2 | `feat(core): add batch contract types` |
| 3 | `feat(core): add batch preprocess helper` |
| 4 | `feat(core): add batch postprocess helper` |
| 5 | `test(classification): batch size 2 coverage` |
| 6 | `test(object_detection): batch size 2 coverage` |
| 7 | `test: batch preprocess/postprocess integration` |
| 8 | `docs: batch processing guide and README` |

Each commit must pass the local gate for touched targets.

---

## Track C — Composite multi-task pipelines

**Goal:** detection + pose, detection + segmentation, without overloading
`TaskInterface`.

Follow **Phase 7** in [task_refactor_atomic_plan.md](./task_refactor_atomic_plan.md).

**Atomic summary**

1. Add `task_pipeline.hpp` / `task_pipeline.cpp` (new API, no `TaskInterface` change). ✅
2. Implement one pipeline (e.g. detection → pose) with explicit intermediate
   `vector<Result>` passing. ✅
3. Mock-task unit tests; no inference engine inside vision-core. ✅
4. README + CHANGELOG under **Added**. ✅

**Independent of Track B:** pipelines chain tasks; batch utilities scale images
within one task.

---

## Track D — Factory maintainability (optional)

**Status:** Done — `TaskDescriptor` metadata and grouped registry comments are
in `task_factory.cpp`; factory boundary tests cover routing precedence.

**Goal:** easier audit as model families grow (pattern review “Low” finding).

**Steps**

1. Introduce internal `TaskDescriptor` `{ name, matcher, family, creator }` in
   `task_factory.cpp` only.
2. Group registry table by task family in source with comments; no behavior change.
3. Extend `test_task_factory.cpp` with one test per family group boundary
   (seg before yolo, pose before yolo, etc.).

**Stop criteria**

- Met — zero routing behavior change; `test_readme_model_types` and factory tests green.

---

## Decision checklist (before starting any step)

1. Does this step require a **minor** or **major** semver bump?
2. Can it be tested with existing synthetic tensors / mats (`N = 2` minimum for batch)?
3. Can it revert independently?
4. Does README model list or `TASKFACTORY_MODEL_LIST` need updating?
5. Are we accidentally conflating **multi-input** (views/frames) with **batch N**?

If (2) or (3) is no, split the step smaller.

---

## Explicitly out of scope

- Third-party runtime task plugins (unless product reopens Phase 2 refactor).
- Inference server batch scheduling (Triton dynamic batcher, TensorRT profiles).
- Model weight or sample media in the repo.

---

## Maintaining this file

When completing a track step:

1. Mark the step done in **Status overview** or the track stop criteria.
2. Update `README.md` roadmap checkboxes.
3. Add a line under `[Unreleased]` in `CHANGELOG.md`.
4. If a new error class appears in CI, extend
   [.claude/skills/ci-guardian/SKILL.md](../.claude/skills/ci-guardian/SKILL.md).
