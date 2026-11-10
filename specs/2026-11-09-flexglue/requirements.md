# Requirements — FlexGlue keypoint matching

Spec: `specs/2026-11-09-flexglue/requirements.md` · Branch: `feature/flexglue-keypoint-matching` → `develop`
Roadmap: keypoint-matching extension (follows 2026-10-15-keypoint-matching)

## Goal

Support `cvg/FlexGlue` (sparse-to-dense feature matching with visual
disambiguation) in `KeypointMatchingTask`, so consumers stop reimplementing
FlexGlue descriptor preprocessing and sparse-matcher decoding per project.
First slice covers the ONNX **sparse mode** (`flexglue_descriptor.onnx` +
`flexglue_matcher.onnx`, fp32, no early-stop) only.

## Context

FlexGlue (https://github.com/cvg/FlexGlue, paper arXiv:2610.32976) matches
through dense features: a frozen DINOv3-ConvNeXt encodes each image to a
stride-8 feature map, a transformer refines both maps jointly, and each query
keypoint finds its correspondence by correlation. It also predicts pair
covisibility to reject look-alike (doppelganger) pairs.

Findings from the upstream repo (README + `onnx/export.py` + `onnx/match_pair.py`,
read 2026-11-09 — **inference, to be confirmed by probing a real export before
any decoder lands**):

- Descriptor: `image [1, 3, S, S]` float32 in `[0, 1]`, squashed to `S×S`
  (default 512) with bilinear resize, no antialiasing → `features [1, 256,
  S/8, S/8]`. Runs once per image.
- Sparse matcher (batch 1): `features0/1 [1, 256, S/8, S/8]`, `kpts0/1 [1, N,
  2]` float32 **corner-origin pixels of the original images** (COLMAP
  convention: top-left pixel center at 0.5, 0.5), `image_size0/1 [1, 2]`
  float32 `(w, h)`, scalar `filter_threshold`, `max_distance_px`,
  `min_covisibility` → `matches0 [1, N0]` int64 (index into kpts1 or -1),
  `mscores0 [1, N0]` float32, `covisibility_score [1]` float32. Both images
  need ≥ 1 keypoint. `min_covisibility` gating happens **inside** the graph
  (all matches rejected below it). ONNX `max_distance_px = 0` rejects all
  matches (unlike PyTorch where 0 disables the check); `inf` disables it.
- Keypoints come from **any external detector** (example: SIFT + 0.5 offset;
  LightGlue detectors need `origin="corner"`). FlexGlue ships no detector.
- Variants out of scope for this slice: `*_early_stop*.onnx`
  (`early_stop_decay` input), fp16 transformer layers, `match_dense` /
  sampled modes, RoMa-v2 `refine=True`.

Consequences that shape the contract (existing code):

- `KeypointMatchingTask` (`include/neuriplo/tasks/keypoint_matching/keypoint_matching_task.hpp`)
  consumes exactly 2 images per call (`getRequiredFrames() == 2`), records the
  pair's original sizes, and returns one `KeypointMatching` per pair.
- Existing families are end-to-end (images in, keypoints + matches out):
  SuperGlue packs the pair into one rank-5 buffer; LightGlue packs `[2, C, H,
  W]`. Their postprocessors recover keypoints from **output** tensors.
- FlexGlue is **not** end-to-end in ONNX form: the matcher needs external
  keypoints as **inputs** and returns no coordinates — only indices. The
  decoder cannot reconstruct `source/target_keypoints` from matcher outputs
  alone. Matcher-only SuperGlue exports were declared out of scope for exactly
  this reason (2026-10-15 packet, decision 4); FlexGlue needs an explicit
  forwarding contract instead of silent guessing.
- `Result` already carries `KeypointMatching {source_keypoints,
  target_keypoints, matches}` — no schema change needed.

## In Scope

- [R-1] Factory routing: `flexglue*` prefix (case/hyphen/space normalized)
  constructs a `KeypointMatchingTask`; `ModelType::FLEXGLUE` added (appended,
  existing order untouched).
- [R-2] `FlexGluePreprocessor`: per-image RGB `[0, 1]` NCHW float32 buffers
  squashed to the descriptor size (default 512×512, from `ModelInfo` 4-D input
  shape), BGR→RGB swap, bilinear, no grayscale, no mean/std. Two buffers —
  one per image — because the descriptor runs once per image (not a packed
  pair).
- [R-3] `FlexGluePostprocessor`: decodes matcher outputs **plus forwarded
  keypoint tensors**. Accepted tensor set (classified by shape, order-free):
  `kpts0/1 [1, N, 2]` float (original-image corner pixels, passed through
  unscaled), `matches0 [1, N0]` int, `mscores0 [1, N0]` float,
  `covisibility_score [1]` float (informational; graph already gated on
  `min_covisibility`). Keeps matches with `score > TaskConfig::
  confidence_threshold`; drops rows pointing outside the keypoint arrays.
- [R-4] Task wiring: `KeypointMatchingTask` selects the FlexGlue
  preprocessor/postprocessor for the new family; input-size extraction reuses
  the 4-D `[N, C, H, W]` path (descriptor shape); documents the 3-step
  consumer flow (descriptor → external detector + matcher → postprocess with
  forwarded kpts).
- [R-5] Tests: routing/aliases, pair validation, buffer shapes/values,
  verified-export decode, threshold filtering, malformed tensors, keypoint
  forwarding errors.
- [R-6] Docs + contract: README `## Features` bullet and
  `TASKFACTORY_MODEL_LIST` block, `export/keypoint_matching/README.md`
  FlexGlue section, batch-matrix row, CHANGELOG `[Unreleased]`, roadmap
  update. No new `TaskConfig` fields.

## Out of Scope

- Early-stop matchers (`early_stop_decay`), fp16 matcher, dense/sampled
  keypoint-less modes, RoMa-v2 refinement. Each is a separate export contract;
  the stub for them is "not implemented", not a guess.
- Bundling or downloading a detector (SIFT/ALIKED), weights, or ONNX files.
  The export guide points at the v0.1 release; weights stay out of the repo.
- Batched multi-pair matching (one pair per call, per decision D-2 of the
  parent packet). The export supports B>1 only with equal keypoint counts —
  a later packet if a consumer needs it.
- Any change to `TaskInterface` or the `Result` variant.
- MatchAnything decoding (still `std::logic_error`; untouched).

## Decisions

- [D-1] Two buffers, not one (descriptor runs per image). Rationale: every
  buffer maps 1:1 onto a model input (parent packet consequence 1); the
  descriptor input is per-image, so packing a pair would misrepresent the
  graph. The single-pair packing rule stays per-family.
- [D-2] Keypoints forwarded as tensors into `postprocess`, not smuggled via
  task state. Rationale: keypoints are matcher **inputs** produced by the
  consumer's detector between the two inference steps; the task never sees
  them otherwise. Reusing the `postprocess(tensors)` channel keeps
  `TaskInterface` stable and makes the data flow explicit and testable.
  Alternative rejected: caching keypoints in the task during `preprocess` —
  impossible, they don't exist yet at preprocess time.
- [D-3] No coordinate rescaling in the FlexGlue decoder. Rationale: `kpts0/1`
  are already original-image corner pixels (upstream contract); SuperGlue
  needed normalize-then-scale and LightGlue needed input-space rescaling, but
  FlexGlue needs neither. The decoder validates finiteness and index bounds
  instead.
- [D-4] Reuse `TaskConfig::confidence_threshold` as the match-score gate
  (strictly-greater, consistent with SuperGlue/LightGlue). Rationale: no new
  knob for the same concept; `min_covisibility` stays a graph input owned by
  the consumer's matcher session, not a library threshold. The covisibility
  score output is decoded but only informational in this slice.
- [D-5] Stubs that throw, decoders that are probed. Rationale: same rule as
  the parent packet (decision 4) — the postprocessor throws
  `std::logic_error` until its layout is verified against a real ONNX file;
  this packet's decoder lands only with the probe evidence in validation.
- [D-6] Sparse fp32 first. Rationale: smallest correct behavior (walk-before-run);
  early-stop/fp16/dense are export variants with extra inputs, not free.

## Constraints

- [mission.md](../mission.md): pixels-in/results-out, no runtime calls, no
  weights in repo, no plugin system (compile-time registry only).
- [tech-stack.md](../tech-stack.md): routing is two matchers — factory entry
  **and** `KeypointMatchingTask::detectModelType` both updated; README block
  is a build contract (`tests/test_readme_model_types.cpp` fails otherwise);
  C++17, 120-col, `clang-format-18` clean, `-DWERROR=ON` green.
- [REPO_META.yaml](../../REPO_META.yaml): no new runtime deps; OpenCV
  untouched (SIFT lives in the consumer/export script, not the library).

## Dependencies

- Codebase: `KeypointMatchingTask`, `KeypointMatchingPreprocessor`,
  `FlexGlue`-less `ModelInfo` 4-D shape path, `TaskFactory` descriptor table,
  `tests/test_keypoint_matching.cpp`, `export/keypoint_matching/` guides.
- External: `flexglue_descriptor.onnx` + `flexglue_matcher.onnx` v0.1 release
  for probing (downloaded at verification time, never committed); ONNX
  Runtime + numpy + OpenCV only in the probe script environment, not the lib.

## Assumptions & Open Questions

- [A-1] "flexgue" in the request is a typo for FlexGlue (`cvg/FlexGlue`).
  Basis: "cvg account" + "keypoints model" matches only that repo. If a
  different model was meant, this packet is the wrong scope.
- [A-2] Upstream ONNX I/O table ~~above is accurate; will be confirmed by probe
  before the decoder merges~~ **confirmed by probe 2026-11-10 (v0.1 SHAs match
  upstream `MODELS`; `max_distance_px=0` rejects all; gating inside the graph)**.
  Basis: `onnx/export.py` + `onnx/match_pair.py` docstrings read 2026-11-09.
- [Q-1] Should the covisibility score be surfaced to consumers (e.g. task
  log, result annotation) once gated-pairs need distinguishing from
  no-match pairs? Owner: maintainer. Deferred — graph already gates; decoder
  returns empty matches either way.
- [Q-2] Does any consumer need multi-pair batching or the early-stop
  variant (COLMAP-scale SfM)? If yes, new packet with its own probe.
- [Q-3] Should `describe_resize` other than 512 (e.g. 640) be supported via
  `ModelInfo` alone? Assumed yes (shape-driven) but only 512 verified here.

## Definition of Done (requirements level)

- [ ] Every [R-n] is implemented or explicitly deferred with a tracked location
- [ ] No In Scope behavior silently dropped; no Out of Scope work smuggled in
