#!/usr/bin/env bash
# .claude/hooks/block_large_binaries.sh
#
# PreToolUse hook for Bash.
#
# vision-core is a static C++ library; it has NO reason to ship model weights,
# ONNX graphs, tokenizers, or sample media in the repo. These files commonly
# appear in the working tree during local inference testing and it's very easy
# to stage them by accident (especially with `git add .`).
#
# This hook rejects any `git add` / `git commit` command that would stage any
# of the blocklisted patterns.
#
# Blocks by exiting 2 with an explanatory message.

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

[[ -z "$command" ]] && exit 0

# Patterns we never want in this repo. Keep tight to avoid false positives.
BLOCKED_RE='\.(weights|onnx|pt|pth|bin|safetensors|engine|trt|mp4|avi|mov|mkv|webm|jpg|jpeg|png|tiff)$|(^|/)(vocab\.json|merges\.txt|labels\.txt|tokenizer\.json|model\.weights|model\.onnx)$'

is_blocked() {
  local file="$1"
  # Allow CI badges / small PNGs under data/, docs/, or README assets.
  # The point is to catch model artefacts and sample media, not UI graphics.
  case "$file" in
    */docs/*|docs/*|*/README*|README*) return 1 ;;
  esac
  [[ "$file" =~ $BLOCKED_RE ]]
}

reject() {
  {
    echo "[block_large_binaries] refused: this command would stage a blocked file."
    echo
    echo "Blocked path: $1"
    echo
    echo "vision-core does not commit model weights, ONNX graphs, tokenizers,"
    echo "or sample media. If you genuinely need to track this file, add an"
    echo "explicit exception to .gitignore and update this hook."
    echo "To stage regardless (not recommended):"
    echo "  VISION_CORE_ALLOW_BINARIES=1 <your command>"
  } >&2
  exit 2
}

# Global opt-out for intentional exceptions.
[[ "${VISION_CORE_ALLOW_BINARIES:-}" == "1" ]] && exit 0

# --- 1. Scan `git add` arguments ---------------------------------------
if [[ "$command" == *"git add"* ]]; then
  # Extract tokens after 'git add' until a '&&', ';', or end of line.
  args="$(printf '%s' "$command" | \
    sed -n 's/.*git[[:space:]]\+add[[:space:]]\+\([^;&|]*\).*/\1/p')"
  for tok in $args; do
    # Skip flags.
    [[ "$tok" == -* ]] && continue
    # If it's a glob, expand it against the working tree.
    mapfile -t expanded < <(compgen -G "$tok" 2>/dev/null || printf '%s\n' "$tok")
    for f in "${expanded[@]}"; do
      if is_blocked "$f"; then
        reject "$f"
      fi
    done
  done
fi

# --- 2. Scan staged changes on `git commit` -----------------------------
if [[ "$command" == *"git commit"* ]] && command -v git >/dev/null 2>&1; then
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    if is_blocked "$f"; then
      reject "$f (already staged — run: git restore --staged \"$f\")"
    fi
  done < <(git diff --cached --name-only 2>/dev/null || true)
fi

exit 0
