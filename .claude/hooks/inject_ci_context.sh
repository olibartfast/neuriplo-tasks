#!/usr/bin/env bash
# .claude/hooks/inject_ci_context.sh
#
# UserPromptSubmit hook. Injects a short, stable reminder about vision-core's
# CI rules so they stay in the active context across long sessions. Without
# this, the agent drifts: re-introduces formatter violations, forgets that
# -DWERROR=ON is a gate, or tries to commit model weights.
#
# Whatever this script prints to stdout is prepended to the user's prompt
# inside Claude Code. Keep it short — every token here is a token not
# available for the task.

cat <<'EOF'
<vision-core-ci-reminder>
Before editing any C/C++ under src/, include/, or tests/:
  - C++17 only. Column limit: 120. 4-space indent. Pointer left (T* p).
  - clang-format-18 MUST be clean; the format_on_edit hook re-checks every edit.
  - clang-tidy-18 runs over src/*.cpp against compile_commands.json in build/.
  - cppcheck runs --enable=warning --std=c++17 -I include src/.
  - -DWERROR=ON is a gate; do not introduce new compiler warnings.
  - Valgrind runs Debug test binaries in CI; do not hide real leaks.
  - Do NOT commit model weights, ONNX graphs, tokenizers, or sample media.
    Extensions blocked: .weights .onnx .pt .pth .bin .safetensors .engine
    .trt .mp4 .avi .mov .jpg .jpeg .png (outside docs/).
    Names blocked: vocab.json, merges.txt, labels.txt, tokenizer.json.
  - Do NOT edit 3rdparty/ to fix lint — suppress in tool config instead.
  - Do NOT add docstrings/comments to code you did not change.
Playbook: .claude/skills/ci-guardian/SKILL.md
</vision-core-ci-reminder>
EOF
