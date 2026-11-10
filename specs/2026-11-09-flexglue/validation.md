# Validation — FlexGlue keypoint matching

Written before implementation. Each requirement maps to an observable check;
the Results section records what was actually run, and a check that could not
be run is **not executed**, not a pass.

## Checks

| # | Requirement | Check | Result |
|---|-------------|-------|--------|
| V-1 | R-1 routing | `test_keypoint_matching` asserts `flexglue`, `FlexGlue`, `flexglue-fp32`, ` flex-glue ` construct `KeypointMatchingTask` with `getRequiredFrames() == 2`; `sift` still rejected | **pass** |
| V-2 | R-1 README contract | `test_readme_model_types` passes both directions with the new aliases | **pass** |
| V-3 | R-2 preprocessing | Task `preprocess` on a 2-image pair returns 2 non-empty buffers of `3×S×S×4` bytes each (S from `ModelInfo`, default 512); RGB planes differ (red→R=1/G=B=0); BGR→RGB swap verified; empty image rejected | **pass** |
| V-4 | R-3 decode vs reference | Full matcher outputs (785/727 SIFT kpts, berlin pair) decoded in C++ equal the Python `match_pair.py` decode: 5/5 matches, identical indices and scores to 1e-9 | **pass** |
| V-5 | R-3 threshold | Scores at or below `TaskConfig::confidence_threshold` dropped; threshold above every score yields zero matches but keeps keypoints | **pass** |
| V-6 | R-3 forwarding errors | Missing `kpts0/kpts1`/`matches0`/`mscores0`, N0 mismatch between matches/scores, out-of-range indices skipped, non-finite coords → `std::invalid_argument`; matcher-only tensors without forwarded kpts never silently succeed | **pass** |
| V-7 | R-3 covisibility | Pair gated by the graph (`min_covisibility`) decodes to zero matches; `covisibility_score` tensor present and finite | **pass** (probe: `min_covisibility` above score → 0 valid) |
| V-8 | R-4 task wiring | `preprocess` records both original sizes; `postprocess` with forwarded kpts returns one `KeypointMatching` in original corner-pixel space (no rescaling); full-output scratch check preserved all 785/727 keypoints | **pass** |
| V-9 | R-6 docs | `export/keypoint_matching/README.md` FlexGlue section present; `export/README.md` links resolve; batch matrix has the family row; CHANGELOG + roadmap updated | **pass** |
| V-10 | Constitution | No new runtime deps; no OpenCV use in `src/`/`include/`; existing suites green (31/31) | **pass** |
| V-11 | Gate (static) | `cppcheck --error-exitcode=1` clean (62 files); new/edited sources clean under local clang-format-21 (18 unavailable locally — CI job covers the pin) | **pass*** |
| V-12 | Gate (build/test) | `-DWERROR=ON` build + `ctest --output-on-failure` green (31/31) | **pass** |
| V-13 | Gate (memory) | Valgrind unavailable on this machine | **not executed** (CI `valgrind` job) |

**Explicitly not executed in this packet:**

- End-to-end accuracy against labelled pairs (no dataset/eval harness).
- Early-stop / fp16 / dense / refine variants (out of scope by D-6).
- Multi-pair batching (out of scope).
- Matching quality vs SuperGlue/LightGlue (no benchmark harness).

## Probe contract (verified 2026-11-10, Group 2)

Probed `flexglue_descriptor.onnx` + `flexglue_matcher.onnx` (v0.1 release;
SHA256 `93fab8a4…` / `a0baf84f…`, matching upstream `onnx/match_pair.py`
`MODELS`) with `export/keypoint_matching/probe_flexglue.py` on the upstream
berlin pair (1100×757 / 960×640, 785 / 727 SIFT keypoints, CPU ORT 1.31.1).
Assumed table **confirmed, no diff**:

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `image` (descriptor in) | `[1, 3, 512, 512]` float32 | RGB `[0,1]`, squashed bilinear |
| `features` (descriptor out) | `[1, 256, 64, 64]` float32 | dense map, once per image |
| `kpts0/1` (matcher in) | `[1, N, 2]` float32, dynamic `num_keypoints0/1` | corner-origin pixels of original images |
| `image_size0/1` (matcher in) | `[1, 2]` float32 | `(w, h)` of originals |
| `filter_threshold` / `max_distance_px` / `min_covisibility` | scalar float32 | runtime gates |
| `matches0` | `[1, N0]` int64 | index into kpts1 or -1 |
| `mscores0` | `[1, N0]` float32 | match confidence, 0.0 where unmatched |
| `covisibility_score` | `[1]` float32 | pair overlap probability |

Observed: 5 valid of 785 at `filter_threshold=0.1`, covisibility 0.9932, all
valid indices in `[0, N1)`; `max_distance_px=0` rejects all;
`min_covisibility` above the score rejects all (gating inside the graph).

## Evidence Log

| ID | Command/Check | Result | Date | Notes |
|----|---------------|--------|------|-------|
| probe | `probe_flexglue.py --descriptor … --matcher … --source berlin1.jpg --target berlin2.jpg` | pass | 2026-11-10 | Contract confirmed; SHAs match upstream; unmatched scores 0.0 |
| full-decode | scratch C++ driver vs Python on full berlin outputs (785/727 kpts) | pass | 2026-11-10 | 5/5 matches, identical `(src, tgt, score)` to 1e-9; driver + data in /tmp (scratch, uncommitted) |
| gate | `-DWERROR=ON` build + `ctest` (31/31) + `cppcheck` clean | pass | 2026-11-10 | clang-format-18/clang-tidy-18/valgrind unavailable locally; new files verified clean under clang-format-21, CI covers the rest |

## Deviations

- The upstream berlin pair yields only 5 valid matches at
  `filter_threshold=0.1` (covisibility 0.9932) — a hard doppelganger-style
  pair, not a decoder defect. Full-output C++ vs Python equality holds 5/5.
- `clang-format-18`, `clang-tidy-18`, and Valgrind are not installed on this
  machine (no root); new/edited sources were verified clean under the local
  clang-format-21 and `cppcheck` is green. The CI jobs remain the gate for the
  pinned checks.
- Subagent delegation (orchestrator/worker) failed with `Model unavailable:
  deepseek/deepseek-v4-flash`, so the orchestration ran inline in the
  worktree following the same phase plan.

## Definition of Done (integration)

- [ ] Every automated and manual check executed, with evidence recorded
- [ ] Evidence traced to [R-n]; deviations documented
- [ ] Spec, code, roadmap, and changelog agree — merge as one coherent change
- [ ] Durable discoveries propagated to `tech-stack.md` / `mission.md`
