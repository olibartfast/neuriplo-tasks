#!/usr/bin/env bash
# .claude/hooks/format_on_edit.sh
#
# PostToolUse hook for Edit/Write/MultiEdit.
#
# Reads the tool payload from stdin, pulls the edited file path, and, if it's
# a C/C++ source or header under src/, include/, or tests/, runs:
#   clang-format-18 -i <file>
# then re-checks with:
#   clang-format-18 --dry-run --Werror <file>
#
# The re-check fails if anything can't be auto-formatted cleanly (e.g. a
# `// clang-format off` block that's itself malformed). This is the single
# biggest source of CI breakage — the `format-check` job in lint.yml runs
# exactly this command.
#
# Exits:
#   0 → all good, continue
#   2 → block the tool and surface the clang-format output back to Claude

set -euo pipefail

payload="$(cat)"

file_path="$(printf '%s' "$payload" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
tool = data.get("tool_input", {})
print(tool.get("file_path") or "")
' 2>/dev/null || true)"

[[ -z "$file_path" ]] && exit 0

# Only format C/C++ files.
case "$file_path" in
  *.cpp|*.hpp|*.h|*.cc|*.cxx) ;;
  *) exit 0 ;;
esac

[[ ! -f "$file_path" ]] && exit 0

# Only format files CI would format.
case "$file_path" in
  */src/*|*/include/*|*/tests/*|src/*|include/*|tests/*) ;;
  *) exit 0 ;;
esac

# Never format vendored code.
case "$file_path" in
  */3rdparty/*|3rdparty/*) exit 0 ;;
esac

# Pick clang-format-18 first (the pinned CI version), fall back to clang-format.
CF=""
if command -v clang-format-18 >/dev/null 2>&1; then
  CF=clang-format-18
elif command -v clang-format >/dev/null 2>&1; then
  CF=clang-format
else
  echo "[format_on_edit] clang-format-18 not installed; skipping" >&2
  exit 0
fi

# 1. Apply formatting in place.
"$CF" -i "$file_path" >/dev/null 2>&1 || true

# 2. Strict re-check — anything left is a hard block.
if ! out="$("$CF" --dry-run --Werror "$file_path" 2>&1)"; then
  {
    echo "clang-format found issues in $file_path that auto-format could not fix:"
    echo
    echo "$out"
    echo
    echo "Consult .claude/skills/ci-guardian/SKILL.md §format for the common causes."
  } >&2
  exit 2
fi

exit 0
