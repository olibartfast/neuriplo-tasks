# Platform alignment plan (neuriplo-platform orchestrator → neuriplo-tasks)

Date: 2026-06-20

Status: Planned

Source: [neuriplo-platform production roadmap](https://github.com/olibartfast/neuriplo-platform/blob/main/docs/architecture/production-roadmap.md)
(items 9, 11), [ADR 0006](https://github.com/olibartfast/neuriplo-platform/blob/main/docs/adr/0006-generative-serving-over-openai-protocol.md),
[ADR 0009](https://github.com/olibartfast/neuriplo-platform/blob/main/docs/adr/0009-evolve-to-ai-infrastructure-gpu-serving-platform.md),
[task contract](https://github.com/olibartfast/neuriplo-platform/blob/main/contracts/task-contract.md),
[result contract](https://github.com/olibartfast/neuriplo-platform/blob/main/contracts/result-contract.md).

Goal: align `neuriplo-tasks` with the neuriplo-platform production roadmap
(ADRs 0006, 0009; production-roadmap items 9, 11; task/result contract
graduation) in atomic reviewable steps.

Non-goals:

- Do not change `TaskInterface`, `Result` variant schema, tensor shapes, or
  model I/O contracts.
- Do not add runtime dependencies beyond OpenCV.
- Do not reimplement serving or inference-side serialization — this repo owns
  the typed result structures and their semantics only.
- Do not rename repos, namespaces, or CMake targets (ADR 0004 already settled).

Guardrails:

- Preserve `TaskFactory::createTaskInstance` source compatibility.
- Preserve README model list behavior tested by `test_readme_model_types.cpp`.
- Every phase must pass the local gate: clang-format, cppcheck, `-DWERROR=ON`
  build, ctest green.

---

## Phase 1 — Positioning refresh (ADR 0009 alignment)

Purpose: reflect the platform's evolution from "computer-vision
toolkit" to "AI inference task contracts — CV as first domain." README,
AGENTS.md, and header-landing-page comments updated in one PR. No code
changes.

Steps:

1. **README.md**:
   - Line 3: drop the "Under Development" banner (library is at v0.4.1, 40+
     model types, stable TaskInterface).
   - Line 5–8: replace "framework-agnostic computer vision algorithms including
     common pre-processing and post-processing steps" with "AI inference task
     contracts providing preprocessing, postprocessing, and typed result
     structures. Computer vision is the first supported task domain; NLP
     embeddings, audio transcription, and tabular inference are planned
     extensions."
   - After Features bullet list, add one line: "Computer vision is the first
     task domain. The `TaskInterface` contract is domain-agnostic — additional
     domains are extensions, not exceptions."

2. **AGENTS.md**:
   - "Project overview" paragraph: replace "framework-agnostic computer-vision
     pre/postprocessing" with "AI inference task contracts (CV as first
     domain)."
   - Consumers line: keep tritonic and neuriplo-infer; add a note that
     neuriplo-kserve-runtime consumes via the task contract at serving scale.

3. **CHANGELOG.md**: add `[Unreleased]` entry under **Changed**:
   "README and AGENTS repositioned to reflect AI-infrastructure platform
   framing (CV as first task domain)."

4. **Header check**: grep `include/` for "vision" framing in doc comments.
   Fix any that say "vision task" where "task" alone is accurate.

Files:

- `README.md`
- `AGENTS.md`
- `CHANGELOG.md`
- Possibly 1–2 headers with stale framing

Verification:

```bash
find src include tests -name '*.cpp' -o -name '*.hpp' | \
  xargs clang-format-18 --dry-run --Werror
cmake -S . -B build -DBUILD_TESTS=ON -DWERROR=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Stop criteria: README no longer says "Under Development" or "computer-vision
algorithms" in the lede; AGENTS matches; all existing tests green.

---

## Phase 2 — Contract audit and boundary tests

Purpose: the platform `contracts/task-contract.md` and
`contracts/result-contract.md` are Draft. Audit `neuriplo-tasks` code
against their stated compatibility rules and add boundary tests that
lock the rules in this repo so future changes can't drift.

Steps:

1. Audit check (read-only, one document update):
   - Walk each compatibility rule in task-contract.md and result-contract.md.
   - Tick off which rules are already verified by existing tests and which are
     implicit.
   - Add a `docs/plans/contract-audit.md` (short, 1-page) with findings.
     Example: "Result variant layout stability — no explicit test but covered
     implicitly by factory + postprocessor tests that extract specific
     alternatives."

2. Add `tests/test_contract_semantics.cpp`:
   - `Result` variant layout: assert `std::holds_alternative<Detection>(Result{})`
     for each concrete type; assert variant index ordering matches `TaskType`
     enum ordering.
   - `BoundingBox` coordinate contract: assert `x`, `y`, `width`, `height` are
     pixel-space, top-left origin, non-negative (test construction + access).
   - `Classification` confidence range: assert `class_confidence` in [0.0, 1.0]
     for well-known explicit-constructor values.
   - `OpenVocabDetection` label semantics: assert `label` is a non-empty string
     when constructed via the 4-arg ctor.
   - `TaskType` ↔ `Result` mapping: for each `TaskType` value, assert the
     corresponding `Result` alternative index is the same ordinal.
   - `Detection` IS-A `Classification`: assert inheritance (compile-time
     `std::is_base_of_v` check).
   - No test changes legacy behavior — these are pure-contract assertions.

3. Run the local gate.

Files:

- `docs/plans/contract-audit.md`
- `tests/test_contract_semantics.cpp`
- `tests/CMakeLists.txt` (add test executable)

Verification:

```bash
./build/tests/test_contract_semantics
ctest --test-dir build --output-on-failure
```

Stop criteria: audit doc lists every contract compatibility rule with status
(covered / implicit / gap); `test_contract_semantics` passes; no existing
test regresses.

---

## Phase 3 — Architecture fitness test (production-roadmap Item 9)

Purpose: the platform requires "neuriplo-tasks must not depend on
neuriplo-infer." Add a compile-time guard that breaks the build if anyone
accidentally adds such a dependency.

Steps:

1. Add `tests/test_architecture_fitness.cpp`:
   - Compile-time check: `#ifdef` test that `NEURIPLO_INFER_VERSION` or
     equivalent neuriplo-infer header is NOT defined. If someone adds a
     `#include <neuriplo/infer/...>` to a public header, the test compile
     fails.
   - More robust: use CMake `check_include_file_cxx` in the test's
     `CMakeLists.txt` to assert `neuriplo/infer/...` is NOT findable from the
     neuriplo-tasks include path. If found, test fails at configure time.
   - Also check: `include/neuriplo/tasks/` tree contains no `#include` of
     anything from `neuriplo/`, `neuriplo-infer/`, `neuriplo-kserve-*/`.

2. Run the local gate.

Files:

- `tests/test_architecture_fitness.cpp`
- `tests/CMakeLists.txt`

Verification:

```bash
cmake -S . -B build -DBUILD_TESTS=ON
# configure must succeed
./build/tests/test_architecture_fitness
```

Stop criteria: test passes (no neuriplo-infer dependency); any attempt to
add such a dependency fails the test.

---

## Phase 4 — Result serialization helpers (opt-in, low-risk)

Purpose: the platform `result-contract.md` has a draft JSON schema for
detection results, but "No producer ships this yet." Provide zero-dependency
serialization helpers in neuriplo-tasks so `neuriplo-infer` and
`neuriplo-kserve-runtime` can emit machine-readable results without
reimplementing the schema.

Non-goals: do not add a JSON library dependency (nlohmann, RapidJSON, etc.).
The helper emits structured data (key-value pairs, nested maps) as a simple
`std::string` builder or `std::map<std::string, std::variant<...>>`. The
consumer chooses the wire format.

Steps:

1. Add `include/neuriplo/tasks/core/result_serialization.hpp`:
   - `std::string serializeDetection(const Detection& d, const std::string& label, int image_width, int image_height)`
     → produces the JSON-object shape matching `result-contract.md` schema.
   - `std::string serializeOpenVocabDetection(const OpenVocabDetection& d, int image_width, int image_height)`
     → same.
   - Both are < 50 lines each. No third-party dependency. Uses `std::ostringstream`
     or `fmt`-style manual construction.
   - Declared in `CORE_HEADERS` section of `CMakeLists.txt`.

2. Add `tests/test_result_serialization.cpp`:
   - Round-trip assertions: construct a `Detection`, serialize, parse
     key fields from output string, assert values match.
   - Test edge cases: zero-confidence, zero-size bbox, empty label.
   - Test both `serializeDetection` and `serializeOpenVocabDetection`.

3. Update `AGENTS.md` core abstractions table with the new header.

Files:

- `include/neuriplo/tasks/core/result_serialization.hpp`
- `src/core/result_serialization.cpp`
- `tests/test_result_serialization.cpp`
- `tests/CMakeLists.txt`
- `CMakeLists.txt` (add to `CORE_HEADERS`)
- `AGENTS.md` (one row in core abstractions table)

Verification:

```bash
./build/tests/test_result_serialization
ctest --test-dir build --output-on-failure
```

Stop criteria: `test_result_serialization` passes; serialized strings match
the field semantics in `result-contract.md`; no new dependencies.

---

## Phase 5 — Sync docs and CHANGELOG

Purpose: close the loop. After Phases 1–4 land, update the internal roadmap
and CHANGELOG so the next agent or contributor sees the current state.

Steps:

1. Update `docs/ROADMAP.md` Status overview table:
   - Add rows: "Platform alignment — Phase 1 positioning | Done", "Phase 2
     contract tests | Done", "Phase 3 fitness test | Done", "Phase 4
     serialization | Done".
2. Update `CHANGELOG.md` `[Unreleased]`:
   - **Changed**: positioning refresh (Phase 1).
   - **Added**: contract boundary tests, architecture fitness test, result
     serialization helpers (Phases 2–4).
3. If `docs/plans/contract-audit.md` has gaps, note them as follow-up issues.
4. Run `scripts/check_platform.py` in `neuriplo-platform` to confirm no
   platform-level validation breaks against the new state.

Files:

- `docs/ROADMAP.md`
- `CHANGELOG.md`
- Possibly `docs/plans/contract-audit.md`

Stop criteria: roadmap and changelog reflect completed phases; platform
validator green (or explainable skip).

---

## Decision checklist (per phase)

1. Does this phase require a semver bump? Phase 4 adds a public header →
   minor bump (`v0.5.0`). Phases 1–3 are docs+test → patch bump.
2. Can it revert independently? Yes — each phase touches disjoint files.
3. Does README model list or `TASKFACTORY_MODEL_LIST` need updating? Only
   Phase 1 framing; no model-type changes.
4. Are we conflating multi-input with batch N? No — these are contract and
   docs changes.

---

## Commit sequence (suggested)

| # | Commit subject | Phase |
|---|----------------|-------|
| 1 | `docs: reposition README and AGENTS for AI-infrastructure framing (ADR 0009)` | 1 |
| 2 | `test: add contract boundary tests for Result, BoundingBox, TaskType semantics` | 2 |
| 3 | `test: add architecture fitness test (no neuriplo-infer dependency)` | 3 |
| 4 | `feat(core): add result serialization helpers for Detection and OpenVocabDetection` | 4 |
| 5 | `docs: update ROADMAP and CHANGELOG for platform alignment phases 1-4` | 5 |

Each commit must pass the local gate.
