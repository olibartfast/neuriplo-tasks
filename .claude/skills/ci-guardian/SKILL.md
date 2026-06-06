---
name: ci-guardian
description: Use whenever a vision-core CI failure needs fixing, whenever you see clang-format / clang-tidy / cppcheck / cmake / ctest output, whenever -DWERROR=ON build warnings need resolving, or whenever you're about to push and want to pre-empt the common failures. Contains the catalogued fix for every error class that has appeared in vision-core's CI history, plus the local-verify command for each.
---

# ci-guardian — vision-core CI failure playbook

This is the single operational reference for every CI failure class
`vision-core` has actually hit. Each section is a **bucket** the `ci-triage`
subagent maps onto, and each section ends with the **exact local command**
that reproduces and verifies the fix.

If a new failure mode appears that isn't catalogued, fix it, then **add a
section here**. The skill compounds in value; do not let it decay.

The repo's six CI jobs (`.github/workflows/lint.yml`) are:

1. `format-check` — clang-format-18 dry-run
2. `clang-tidy` — clang-tidy-18 over `src/**/*.cpp`
3. `cppcheck` — cppcheck with `--error-exitcode=1`
4. `build-and-test` — cmake + build + ctest
5. `valgrind` — Debug test build under Valgrind
6. `build-warnings` — cmake with `-DWERROR=ON` + build

---

## §format — clang-format check failures

**Symptom.** CI `format-check` job fails with:
```
error: code should be clang-formatted [-Wclang-format-violations]
```
and a unified diff.

**Local reproduce.**
```bash
find src include tests -name '*.cpp' -o -name '*.hpp' | \
  xargs clang-format-18 --dry-run --Werror
```

**Fix.**
```bash
find src include tests -name '*.cpp' -o -name '*.hpp' | \
  xargs clang-format-18 -i
```

There is no judgment call here. If the `format_on_edit.sh` hook is installed
and executable, this bucket should never fire.

**Active config** (`.clang-format`): LLVM base, 4-space, 120 col, left
pointer alignment, regrouped case-sensitive includes, C++17.

### Non-fixes

- Do **not** add `// clang-format off` around long lines. Break the line
  at a comma or paren. The column limit is a hard rule.
- Do **not** modify `.clang-format` to make a file pass. Either the
  file is wrong, or the whole project's style changes — never just one
  file.

---

## §clang-tidy — static analysis failures

**Symptom.** CI `clang-tidy` job fails on a `src/**/*.cpp` file with
warnings or errors matching a check in `.clang-tidy`.

**Local reproduce.**
```bash
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
find src -name '*.cpp' | xargs clang-tidy-18 -p build
```

**Active check set** (`.clang-tidy`):
- Enabled: `bugprone-*`, `cppcoreguidelines-*`, `modernize-*`,
  `performance-*`, `readability-*`, `clang-analyzer-*`
- Disabled:
  `modernize-use-trailing-return-type`,
  `readability-magic-numbers`,
  `cppcoreguidelines-avoid-magic-numbers`,
  `cppcoreguidelines-pro-bounds-pointer-arithmetic`,
  `cppcoreguidelines-pro-type-reinterpret-cast`,
  `cppcoreguidelines-pro-type-const-cast`,
  `cppcoreguidelines-avoid-non-const-global-variables`,
  `cppcoreguidelines-pro-bounds-constant-array-index`,
  `cppcoreguidelines-special-member-functions`,
  `bugprone-easily-swappable-parameters`,
  `readability-uppercase-literal-suffix`,
  `readability-identifier-length`,
  `readability-convert-member-functions-to-static`,
  `modernize-return-braced-init-list`,
  `modernize-use-auto`,
  `modernize-pass-by-value`,
  `cppcoreguidelines-non-private-member-variables-in-classes`
- `HeaderFilterRegex: (src|include)/.*` — only the library's own headers,
  never system or 3rdparty.

### Common checks seen in vision-core

| Check | Fix |
|---|---|
| `modernize-use-nullptr` | `NULL` / `0` → `nullptr`. Auto-fixable with `--fix`. |
| `modernize-use-override` | Add `override` to virtual overrides. Auto-fixable. |
| `modernize-use-using` | `typedef X Y;` → `using Y = X;`. Auto-fixable. |
| `modernize-loop-convert` | Rewrite index loop as range-`for`. Usually auto-fixable. |
| `modernize-use-equals-default` | `Foo() {}` → `Foo() = default;`. Auto-fixable. |
| `readability-implicit-bool-conversion` | `if (ptr)` → `if (ptr != nullptr)`; `if (n)` → `if (n != 0)`. |
| `readability-else-after-return` | Drop the `else` branch; the `if` body returns. |
| `readability-braces-around-statements` | Add `{ ... }` to one-line `if`/`for`/`while` bodies. |
| `readability-container-size-empty` | `v.size() == 0` → `v.empty()`. |
| `bugprone-narrowing-conversions` | Add explicit `static_cast<T>(...)`. Never silence. |
| `bugprone-unused-return-value` | Assign to `(void)fn(...)` or actually use the value. |
| `cppcoreguidelines-init-variables` | `int x;` → `int x = 0;`. Same for `bool`, `float`, pointers. |
| `cppcoreguidelines-pro-type-member-init` | Initialise the member in the ctor init list or at declaration. |
| `cppcoreguidelines-explicit-virtual-functions` | Pair with `modernize-use-override` — add `override`. |
| `performance-unnecessary-copy-initialization` | `auto x = container[i];` → `const auto& x = container[i];`. |
| `performance-for-range-copy` | `for (auto x : v)` → `for (const auto& x : v)`. |
| `clang-analyzer-core.NullDereference` | Real bug. Add the null check; never suppress. |

### Workflow

```bash
# 1. Auto-fix
find src -name '*.cpp' | \
  xargs clang-tidy-18 -p build --fix --fix-errors
# 2. Re-run to see what's left
find src -name '*.cpp' | xargs clang-tidy-18 -p build
# 3. Hand-fix per table above
```

### Non-fixes

- **Do not** add `// NOLINT` without a specific check name:
  `// NOLINTNEXTLINE(check-name)` with a one-line justification.
- **Do not** disable a whole check in `.clang-tidy` to get green. If a
  check is genuinely wrong for this codebase, surface to the user first.
- **Do not** edit `3rdparty/`. It's already outside `HeaderFilterRegex`.

---

## §cppcheck — cppcheck failures

**Symptom.** CI `cppcheck` job exits non-zero (`--error-exitcode=1`).

**Local reproduce.**
```bash
cppcheck --enable=warning --std=c++17 \
  --suppress=missingIncludeSystem \
  --suppress=unmatchedSuppression \
  --error-exitcode=1 \
  -I include src/
```

### Common messages

| Message id | Fix |
|---|---|
| `uninitMemberVar` | Initialise the member in the ctor init list or at declaration. Default-member-init is preferred for simple types. |
| `noExplicitConstructor` | Add `explicit` to single-argument constructors. |
| `passedByValue` | Change the parameter to `const T&`. |
| `constVariable` / `constParameter` | Add `const`. |
| `unusedFunction` | Check whether the function is part of the public API; if yes, suppress with `// cppcheck-suppress unusedFunction` in the header. If no, delete it. |
| `memleak` / `resourceLeak` | Real bug — fix it; prefer RAII (`std::unique_ptr`, `cv::Mat` etc.) over manual `delete`. |
| `danglingLifetime` | Real bug — fix the lifetime. Never suppress. |

### Non-fixes

- **Do not** broaden `--suppress=` in `.pre-commit-config.yaml` or
  `.github/workflows/lint.yml` to make CI pass. Targeted inline
  suppressions with a reason are acceptable for false positives; entire
  categories are not.

---

## §werror — -DWERROR=ON build errors

**Symptom.** CI `build-warnings` job fails with a compiler diagnostic
promoted to error, e.g.:
```
error: unused variable 'x' [-Werror,-Wunused-variable]
```

**Local reproduce.**
```bash
rm -rf build-werror
cmake -S . -B build-werror -DWERROR=ON
cmake --build build-werror --parallel
```

### Common warnings

| Warning | Fix |
|---|---|
| `-Wunused-variable` / `-Wunused-parameter` | Remove it; or mark as `[[maybe_unused]]` (not for new code — usually a sign you should delete); or prefix param name with `_` if it's an interface requirement |
| `-Wunused-but-set-variable` | The assignment is dead — delete it. Don't keep it "for debugging". |
| `-Wsign-compare` | Add `static_cast<size_t>(...)` on the signed side, or use the matching loop-index type |
| `-Wreturn-type` | Function can reach the end without returning — fix the control flow. Real bug. |
| `-Wshadow` | Rename the inner variable. Do not disable `-Wshadow` globally. |
| `-Wdeprecated-declarations` on OpenCV | Use the new API in `cv::`; if genuinely transitional, wrap the single line with a targeted `#pragma` plus a TODO |
| `-Wconversion` / `-Wnarrowing` | Add explicit `static_cast`. |
| `-Wreorder` | Reorder the ctor init list to match member declaration order. |
| `-Wunused-result` on `fread`/`fwrite` etc. | Capture and check, or cast to `(void)` with a reason |

### Non-fixes

- **Do not** disable `-Werror` in `CMakeLists.txt` to get green.
- **Do not** sprinkle `#pragma GCC diagnostic ignored` — scope targeted
  suppressions to single lines via `target_compile_options(tgt PRIVATE
  -Wno-<name>)` on a tiny internal target, and justify in the diff.

---

## §cmake — CMake configure failures

**Symptom.** CI fails at the `cmake -S . -B build ...` step, before any
compilation.

**Local reproduce.**
```bash
rm -rf build
cmake -S . -B build -DBUILD_TESTS=ON -DWERROR=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

### Common causes

| Symptom | Fix |
|---|---|
| `Could NOT find OpenCV` | Runner missing `libopencv-dev`. Check `.github/workflows/lint.yml` installs it for this job. Locally: `sudo apt-get install libopencv-dev`. |
| FetchContent hang / network error on GoogleTest | Stale `build/_deps/`. Delete `build/` and re-configure. If recurring, pin `GIT_TAG` in `tests/CMakeLists.txt`. |
| `add_subdirectory given source ... which is not an existing directory` | Typo in CMakeLists after a module rename. |
| `target_link_libraries called with non-compatible type: UTILITY` | Linking against a non-library target. |
| `CMake Error: The source directory ... does not appear to contain CMakeLists.txt` | Running from the wrong dir, or a FetchContent source didn't clone. |

### Non-fixes

- **Do not** `file(GLOB)` source lists — explicit lists only, so CMake
  re-runs when files are added.
- **Do not** commit `build/`, `_deps/`, `compile_commands.json`.

---

## §build — compile errors during cmake --build

**Symptom.** CI `build-and-test` fails at the build step with a compiler
`error:` (not promoted warning — see §werror for those).

**First move.** Read the error. Don't start rearranging includes.
Compile errors in C++ are almost always exactly what they say.

### Common causes

| Symptom | Fix |
|---|---|
| `fatal error: opencv2/...: No such file or directory` | Missing `target_link_libraries(... ${OpenCV_LIBS})` or `target_include_directories(... ${OpenCV_INCLUDE_DIRS})`. Hand to `@cmake-triage`. |
| `error: use of undeclared identifier 'cv::...'` | Missing `#include <opencv2/...>` in the file; or a new OpenCV version moved the symbol. |
| `undefined reference to cv::...` at link | Missing `${OpenCV_LIBS}` in `target_link_libraries`. |
| `static assertion failed: ... std::variant` | Usually a `Result` variant mismatch — new postprocessor returning a type not in `result_types.hpp`. Update `Result` or the return type. |
| `no matching function for call to 'Tensor'` | `Tensor` constructor signature changed; update the call site. |

---

## §ctest — test failures

**Symptom.** CI `build-and-test` fails at the `ctest` step.
```
The following tests FAILED:
  3 - test_yolo_postprocessor (Failed)
```

**First move.** Run the single failing test with verbose output *before*
doing anything else:

```bash
ctest --test-dir build -R "^test_yolo_postprocessor$" --output-on-failure -V
# or run the binary directly for gdb/lldb:
./build/tests/test_yolo_postprocessor --gtest_filter=<FailingSuite.FailingTest>
```

Never mutate code before you've read the actual GoogleTest assertion
output. The most common accident when chasing lint failures is "fixing"
a test by deleting the assertion.

### Common causes

- **`Result` variant reordered** → GoogleTest `std::get<N>(variant)` by
  index breaks. Use `std::get<Type>(variant)` instead.
- **Postprocessor output shape changed** → test expects old tensor
  layout. Fix the test if the postprocessor change is intentional.
- **Floating-point compare** → use `EXPECT_NEAR` / `EXPECT_FLOAT_EQ`,
  not `EXPECT_EQ`, on `float`.
- **Test data missing** → file at `tests/fixtures/...` not present or
  not installed by CMake. Hand to `@cmake-triage`.

---

## §valgrind — runtime memory failures

**Symptom.** CI `valgrind` job fails with Valgrind output containing one of:
```
ERROR SUMMARY: N errors from N contexts
definitely lost:
Invalid read
Invalid write
Use of uninitialised value
```

**Local reproduce.**
```bash
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

**Fix.**
- Invalid read/write or use-after-free: fix ownership/lifetime, then add or
  tighten the regression test that exercises the lifetime.
- `definitely lost` / `indirectly lost`: free the owning allocation or replace
  manual ownership with RAII.
- Third-party or OpenCV noise: prefer a narrow Valgrind suppression file only
  after proving the report is outside vision-core code.

Do not hide real vision-core leaks with suppressions.

---

## §opencv — OpenCV-specific failures

**Symptom.** Any of:
- `undefined reference to cv::dnn::...`
- `no matching function for call to cv::Mat::...`
- `cv::Exception: OpenCV(4.x) ...`

**Local reproduce.** Same as §build, plus checking the installed OpenCV
version:
```bash
dpkg -l | grep libopencv-dev
pkg-config --modversion opencv4
```

### Common causes

| Symptom | Fix |
|---|---|
| `undefined reference to cv::dnn::readNetFromONNX` | Link against `${OpenCV_LIBS}` which must include `opencv_dnn`. Check the `find_package(OpenCV REQUIRED COMPONENTS ...)` line. |
| Runtime `cv::Exception: OpenCV(4.x) error: (-215)` | Preconditions for `cv::Mat` operation violated (wrong type, empty matrix, shape mismatch). Real bug; fix the caller. |
| Version mismatch between runner and local | Use `pkg-config --modversion opencv4` on both; if CI is older, pin or upgrade the apt source. |

### Non-fixes

- **Do not** wrap OpenCV calls in `try/catch(cv::Exception)` to silence
  test failures. The exception is telling you something is genuinely
  wrong upstream.

---

## §fetchcontent — FetchContent / GoogleTest issues

**Symptom.** CI build fails in `_deps/googletest-src` or the configure
step times out waiting on a network fetch.

**Fix procedure.**
1. Delete `build/_deps/` and retry.
2. If the remote is flaky, cache more aggressively in
   `.github/workflows/lint.yml` (use `actions/cache` keyed on the CMake
   `GIT_TAG`).
3. Never pin `GIT_TAG` to `main` — always a release tag.

---

## Local CI gate — the one command to rule them all

Before pushing, run:

```bash
find src include tests -name '*.cpp' -o -name '*.hpp' | \
  xargs clang-format-18 --dry-run --Werror && \
cppcheck --enable=warning --std=c++17 \
  --suppress=missingIncludeSystem \
  --suppress=unmatchedSuppression \
  --error-exitcode=1 -I include src/ && \
cmake -S . -B build -DBUILD_TESTS=ON -DWERROR=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && \
cmake --build build --parallel && \
find src -name '*.cpp' | xargs clang-tidy-18 -p build && \
ctest --test-dir build --output-on-failure && \
cmake -S . -B build-valgrind -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON && \
cmake --build build-valgrind --parallel && \
for test_bin in build-valgrind/tests/test_*; do \
  [ -x "$test_bin" ] || continue; \
  valgrind --error-exitcode=1 --leak-check=full \
    --show-leak-kinds=definite,indirect \
    --errors-for-leak-kinds=definite,indirect \
    --track-origins=yes --num-callers=25 "$test_bin"; \
done && \
echo "OK — safe to push"
```

If this command is green locally, CI will be green.

---

## When to update this file

If you fix a CI failure whose cause doesn't appear in any section above,
**add a new section** at the point of fix. Include:

1. The exact error line (so future `grep` finds it).
2. Root cause in 1–3 sentences.
3. The fix, as a command or a code pattern.
4. The local-reproduce command.

A playbook that isn't maintained is worse than no playbook.
