#!/usr/bin/env bash
# .claude/hooks/pre_push_ci_gate.sh
#
# PreToolUse hook for Bash. If Claude is about to run `git push`, we first run
# the exact same gates that .github/workflows/lint.yml runs, locally. If any
# gate fails, we block the push and hand the output back to Claude.
#
# Turns the slow loop (push → wait 2-5 min → CI red → fix → repeat) into an
# instant one. That's the whole point of this kit.
#
# The commands MUST stay in sync with .github/workflows/lint.yml.

set -euo pipefail

payload="$(cat)"

command="$(printf '%s' "$payload" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
print(data.get("tool_input", {}).get("command", ""))
' 2>/dev/null || true)"

# Only intercept actual `git push` invocations — not strings or comments that
# happen to contain the phrase. Match `git push` as a leading token, or right
# after a shell separator (`;`, `&&`, `||`, `|`, `&`).
if ! printf '%s' "$command" | \
    grep -qE '(^|[;&|]|&&|\|\|)[[:space:]]*git[[:space:]]+push([[:space:]]|$)'; then
  exit 0
fi

# Respect explicit opt-out.
if [[ "$command" == *"--no-verify"* ]] || [[ "${VISION_CORE_SKIP_CI_GATE:-}" == "1" ]]; then
  echo "[pre_push_ci_gate] opt-out detected; skipping local CI gate" >&2
  exit 0
fi

echo "[pre_push_ci_gate] running local CI gate before push..." >&2

fail() {
  {
    echo
    echo "============================================================"
    echo "  LOCAL CI GATE FAILED — push blocked"
    echo "============================================================"
    echo "$1"
    echo
    echo "Fix the above, then retry the push."
    echo "To bypass (not recommended):"
    echo "  VISION_CORE_SKIP_CI_GATE=1 git push ..."
    echo "  or:   git push --no-verify ..."
  } >&2
  exit 2
}

# Use the build directory CI would use. Cache is preserved between runs.
BUILD_DIR="${VISION_CORE_GATE_BUILD_DIR:-build}"

# --- 1. format-check job -------------------------------------------------
CF=""
if command -v clang-format-18 >/dev/null 2>&1; then
  CF=clang-format-18
elif command -v clang-format >/dev/null 2>&1; then
  CF=clang-format
fi

if [[ -n "$CF" ]]; then
  mapfile -t cpp_files < <(find src include tests \
    \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) 2>/dev/null)
  if (( ${#cpp_files[@]} > 0 )); then
    if ! fmt_out="$("$CF" --dry-run --Werror "${cpp_files[@]}" 2>&1)"; then
      fail "clang-format check failed (run: find src include tests -name '*.cpp' -o -name '*.hpp' | xargs $CF -i):
$fmt_out"
    fi
  fi
else
  echo "[pre_push_ci_gate] warning: clang-format-18 not installed, skipping format stage" >&2
fi

# --- 2. cppcheck job -----------------------------------------------------
if command -v cppcheck >/dev/null 2>&1; then
  if ! cc_out="$(cppcheck --enable=warning --std=c++17 \
      --suppress=missingIncludeSystem \
      --suppress=unmatchedSuppression \
      --error-exitcode=1 \
      -I include src/ 2>&1)"; then
    fail "cppcheck failed:
$cc_out

See .claude/skills/ci-guardian/SKILL.md §cppcheck for the catalogue."
  fi
else
  echo "[pre_push_ci_gate] warning: cppcheck not installed, skipping cppcheck stage" >&2
fi

# --- 3. build-and-test + build-warnings jobs -----------------------------
# Combined into a single -DBUILD_TESTS=ON -DWERROR=ON build so the gate is fast.
if command -v cmake >/dev/null 2>&1; then
  if ! cfg_out="$(cmake -S . -B "$BUILD_DIR" \
      -DBUILD_TESTS=ON -DWERROR=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 2>&1)"; then
    fail "cmake configure failed:
$cfg_out

See .claude/skills/ci-guardian/SKILL.md §cmake for the catalogue."
  fi

  if ! build_out="$(cmake --build "$BUILD_DIR" --parallel 2>&1)"; then
    fail "cmake --build failed (with -DWERROR=ON):
$build_out

See .claude/skills/ci-guardian/SKILL.md §werror for the catalogue."
  fi
else
  fail "cmake is required for the local CI gate but was not found"
fi

# --- 4. clang-tidy job ---------------------------------------------------
# Run only if compile_commands.json exists (just produced above).
CT=""
if command -v clang-tidy-18 >/dev/null 2>&1; then
  CT=clang-tidy-18
elif command -v clang-tidy >/dev/null 2>&1; then
  CT=clang-tidy
fi

if [[ -n "$CT" ]] && [[ -f "$BUILD_DIR/compile_commands.json" ]]; then
  mapfile -t src_cpp < <(find src -name '*.cpp' 2>/dev/null)
  if (( ${#src_cpp[@]} > 0 )); then
    if ! ct_out="$("$CT" -p "$BUILD_DIR" "${src_cpp[@]}" 2>&1)"; then
      fail "clang-tidy failed:
$ct_out

See .claude/skills/ci-guardian/SKILL.md §clang-tidy for the catalogue."
    fi
  fi
else
  [[ -z "$CT" ]] && echo "[pre_push_ci_gate] warning: clang-tidy-18 not installed, skipping" >&2
fi

# --- 5. ctest job --------------------------------------------------------
if ! test_out="$(ctest --test-dir "$BUILD_DIR" --output-on-failure 2>&1)"; then
  fail "ctest failed:
$test_out"
fi

echo "[pre_push_ci_gate] OK — local CI gate passed, allowing push" >&2
exit 0
