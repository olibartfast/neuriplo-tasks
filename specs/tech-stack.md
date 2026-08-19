# Tech stack

*Living document — describes the project as it is now. If it disagrees with the
code, this file is wrong and should be fixed.*

**Boundary:** [AGENTS.md](../AGENTS.md) is the single source of truth for coding
conventions, tooling commands, and CI rules, and [REPO_META.yaml](../REPO_META.yaml)
for machine-readable entrypoints. This file does not restate them — it covers
the shape of the library, the contracts that must not break, and the durable
findings promoted out of feature packets.

## Language and artifact

| | |
|---|---|
| Standard | C++17 |
| Artifact | `libneuriplo-tasks.a` — static library |
| Required runtime dependencies | **none** |
| Public headers | `include/neuriplo/tasks/` |

## Targets

| Target | Option | Default | Contains |
|--------|--------|---------|----------|
| `neuriplo-tasks::neuriplo-tasks` (alias `::vision-core`) | — | always | Tasks, pre/postprocessors, `vision::Image` and ops |
| `neuriplo-tasks::vision-stb` | `NEURIPLO_TASKS_WITH_STB` | `ON` | `loadImage` / `decodeImage` / `saveImage`, stb vendored in `3rdparty/` |
| `neuriplo-tasks::vision-opencv` | `NEURIPLO_TASKS_WITH_OPENCV` | `OFF` | `cv::Mat` ↔ `vision::Image` conversions |

Image I/O and OpenCV interop are **separate targets on purpose**: linking the
core must not drag in either. A consumer that has its own decoder links only
`vision-core`.

Other options: `BUILD_TESTS` (`OFF`, fetches GoogleTest), `WERROR` (`OFF`,
CI runs `ON`), `SANITIZERS` (`OFF`, ASan + UBSan).

## Layout

| Path | Holds |
|------|-------|
| `include/neuriplo/tasks/core/` | `TaskInterface`, `TaskFactory`, `Result`, `Tensor`, `ModelInfo`, batch and pipeline helpers, `vision/` |
| `include/neuriplo/tasks/<domain>/` | Per-domain task headers — the only headers a consumer needs |
| `src/<domain>/` | `*_task.cpp` plus its postprocessors, which stay internal to the domain |
| `tests/` | GoogleTest suites, one per unit, registered with ctest |
| `export/` | Python ONNX-export scripts, one directory per domain. Not compiled |
| `3rdparty/` | Vendored code (stb, RT-DETRv4 reference). Never edited to satisfy lint — suppress instead |
| `docs/` | Consumer-facing guides and [Versioning.md](../docs/Versioning.md) |
| `specs/` | This constitution and the dated feature packets |

Domains: `object_detection`, `instance_segmentation`, `classification`,
`video_classification`, `pose_estimation`, `depth_estimation`, `optical_flow`,
`open_vocab_detection`, `gaussian_splatting`, `image_understanding`, `core`.

## The two contracts that must not break

1. **Task contract** — `preprocess(vector<Image>) -> vector<vector<uint8_t>>`,
   `postprocess(Size, vector<Tensor>) -> vector<Result>`.
2. **Result schema** — the `Result` variant and its members.

Consumers pin a tag and upgrade on their own schedule; changing either silently
breaks them at a distance. Shape and dtype changes are review-mandatory
(`review_shape_dtype_changes` in REPO_META.yaml).

## Routing is two matchers, not one

Adding a model alias means touching **both**, or the alias half-works:

| Matcher | Lives in | Decides |
|---------|----------|---------|
| `TaskFactory::createTaskInstance` → `normalizeModelType` | `src/core/task_factory.cpp` | *which task* — an unmatched name is rejected outright with `Unrecognized model type` |
| `detectModelType` | e.g. `src/object_detection/object_detection_task.cpp` | *which postprocessor* inside the task |

A name known to the second but not the first never reaches it. Both matchers
normalize by stripping `-`, `_`, and whitespace, then lowercasing, so aliases
should be matched by **prefix** (`rfind(x, 0) == 0`) where a family has
numbered revisions — `deim`, `deimv2`, and whatever comes next then resolve
without another patch.

`README.md`'s `<!-- TASKFACTORY_MODEL_LIST -->` block is not documentation
courtesy: `tests/test_readme_model_types.cpp` asserts against it, so a routing
change that skips the README fails the build.

## Detector conventions worth writing down

Discovered by probing real ONNX files; see
[2026-08-20-detector-gaps](2026-08-20-detector-gaps/requirements.md).

- **NMS-free YOLO** (`v10`, `26`) emits `[1, 300, 6]` as `x1 y1 x2 y2 score class`
  in **input-space pixels**, not normalized units — decoders must not assume
  `[0,1]`. The two conventions exist in the wild for the same tensor shape, so
  the convention is detected per tensor from the coordinate magnitude rather
  than assumed from the model type.
- **RT-DETR, RT-DETRv2, D-FINE, DEIM** take `images` scaled to `[0,1]` with
  **no ImageNet mean/std**, plus `orig_target_sizes`; they return
  `labels`/`boxes`/`scores`. **RF-DETR and EdgeCrafter do apply ImageNet
  statistics.** Sharing a preprocessor across these families is how the
  distinction gets lost.
- A detector that returns nothing, or boxes clustered off-frame, is nearly
  always a normalization or coordinate-space mismatch — not a broken model.

## Build and CI

Entrypoints live in [REPO_META.yaml](../REPO_META.yaml); the full local gate
command and the six CI jobs are in [AGENTS.md](../AGENTS.md). Two notes that
belong to the stack rather than the tooling:

- CI pins **clang-format-18**. A newer local clang-format formats some
  constructs differently; after reformatting, re-run the exact CI command
  before pushing.
- `paths-ignore` covers `**.md`, `export/**`, `docs/**` — documentation-only
  pushes skip CI by design, so a green tick on such a push means "not run".

## Branch and release model

`develop` integrates, `master` is release-only, enforced by
`.github/workflows/branch-policy.yml`: `master` accepts pull requests only from
`develop`, `release/*`, or `hotfix/*`, and `feature/*` may only target
`develop`. `VERSION` is the single source of version truth (CMake reads it),
`CHANGELOG.md` is the single source of release notes, and every tag must have a
matching GitHub Release created from that changelog — never `--generate-notes`.
Full procedure: [docs/Versioning.md](../docs/Versioning.md).
