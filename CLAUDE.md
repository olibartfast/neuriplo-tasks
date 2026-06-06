# neuriplo-tasks — Claude instructions

Primary source of truth: **[AGENTS.md](./AGENTS.md)** — read that file before
making changes.

Key points:

- C++17 static library (`libneuriplo-tasks.a`); source roots `src/` + `include/neuriplo/tasks/`
- Build: `cmake -S . -B build -DBUILD_TESTS=ON && cmake --build build --parallel`
- Tests: `ctest --test-dir build --output-on-failure` (GoogleTest)
- Lint: `clang-format-18` (120 col), `clang-tidy-18` (against `build/` compile DB), `cppcheck`
- CI will fail on: unformatted code, clang-tidy warnings on `src/**/*.cpp`, cppcheck warnings, `-DWERROR=ON` build errors, failing `ctest`
- Do **not** commit build artefacts, model weights, or sample media (`*.weights`, `*.onnx`, `*.mp4`, `*.jpg`, `vocab.json`, `merges.txt`, `labels.txt`) — the `block_large_binaries.sh` hook will reject them
- Do **not** edit `3rdparty/` to fix lint — suppress in tool config instead

Playbook for any CI failure: **[.claude/skills/ci-guardian/SKILL.md](./.claude/skills/ci-guardian/SKILL.md)**.
