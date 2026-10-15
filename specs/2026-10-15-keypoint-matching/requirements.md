# Keypoint matching (SuperGlue family) — requirements

**Date:** 2026-10-15
**Branch:** `feature/keypoint-matching` → `develop`
**Status:** scaffold phase — decoding is deliberately deferred (see Decisions).

## Goal

A `KeypointMatchingTask` that finds correspondences between the two images of a
pair, so consumers stop reimplementing SuperGlue-family preprocessing and
decoding per project. This packet delivers the **scaffold**: the task contract,
the result type, factory routing, the preprocessing seam, and the tests that
lock them. Per-family decoders land once each export's tensor layout has been
probed against a real ONNX file.

## Context

Requested families: `superglue`, `lightglue`, `match-anything` and similar. What
their exports actually look like was checked before designing the contract:

| Family | Common export | Shape |
|--------|---------------|-------|
| SuperGlue | matcher only — `keypoints_0`, `scores_0`, `descriptors_0`, `keypoints_1`, `scores_1`, `descriptors_1` → `matches0`, `matching_scores0` ([jadehh/SuperPoint-SuperGlue-ONNX](https://github.com/jadehh/SuperPoint-SuperGlue-ONNX)) | no images in |
| LightGlue | end-to-end pipeline — interleaved pair `(2B, 1, H, W)` → `keypoints (2B, K, 2)`, `matches (M, 3)`, `match_scores (M,)` ([LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX)) | images in, matches out |
| MatchAnything | HF / Optimum export of `AutoModelForKeypointMatching` (`zju-community/matchanything_eloftr`); contract not yet pinned | images in |

Two consequences shape the scope:

1. **Matcher-only exports do not fit this library's contract.**
   `TaskInterface::preprocess` takes images and produces inference inputs. A
   graph that needs keypoints and descriptors cannot be driven through it
   without a front-end extractor, which is a different task orchestration
   problem, not a decoder. Only end-to-end (extractor + matcher) exports are
   supported here.
2. **Input pair layout differs per family** (SuperGlue-style two inputs vs
   LightGlue's interleaved `(2B, 1, H, W)`), so the preprocessing seam must be
   per family before a decoder claims anything.

## In scope

- `Result` schema: `KeypointMatch` and `KeypointMatching`, added to the variant;
  `TaskType::KeypointMatching` (additive, end of both).
- `KeypointMatchingTask`: `getRequiredFrames() == 2`, exactly one image pair per
  call, `postprocess` returns one `KeypointMatching` per pair.
- `KeypointMatchingPreprocessor`: pair preprocessing (resize, BGR→RGB, `[0,1]`,
  NCHW float32), one buffer per image — mirrors `RaftPreprocessor`.
- Per-family postprocessor seams: `SuperGluePostprocessor`,
  `LightGluePostprocessor`, `MatchAnythingPostprocessor`.
- `TaskFactory` routing: `superglue*`, `lightglue*`, `matchanything*` by prefix.
- Tests: routing and alias normalization, pair validation, preprocessing
  buffers, postprocess seams, result-variant coverage.
- README `## Features` bullet and `TASKFACTORY_MODEL_LIST` block;
  `export/keypoint_matching/README.md`; batch support matrix row; changelog;
  roadmap.

## Out of scope

- **Decoding matches from real tensors.** The postprocessors throw
  `std::logic_error` until an export's layout is verified against a real file.
  This is the next packet; writing a decoder from assumed layouts is how the
  RT-DETR/EdgeCrafter normalization bugs happened.
- **Matcher-only exports** (keypoints/descriptors in — the original SuperGlue
  export). Documented as unsupported; a front-end orchestration design would be
  its own packet.
- **Multi-pair / batched matching.** One pair per call; consumers loop. The
  LightGlue `(2B, ...)` batch axis is a later concern.
- Auto-downloads of weights or export scripts; the export guide documents where
  to get an ONNX file and how to verify it.
- Any change to existing task contracts or result alternatives.

## Decisions

1. **Additive result-schema change.** `KeypointMatch {source_index,
   target_index, score}` and `KeypointMatching {source_keypoints,
   target_keypoints, matches}` are appended to `Result` and `TaskType`. Appending
   keeps existing variant indices stable. The change is review-mandatory per
   [`REPO_META.yaml`](../../REPO_META.yaml) (`preserve_result_schema`), so it is
   called out here and in the changelog rather than slipped in.
2. **Exactly two images per call.** A pair is one unit of work; odd counts are a
   caller bug, not something to silently pair. Multi-pair semantics need batch
   metadata the tensor contract does not carry.
3. **Prefix aliases.** `superglue`, `lightglue`, `matchanything` match by prefix
   so `lightglue_v2`, `matchanything-eloftr`, and later revisions resolve
   without another patch — the `deimv2` failure repeated by design would be a
   regression. A generic `keypointmatching` alias was considered and **dropped**:
   routing a family-less name creates a task that cannot select a decoder, so
   `detectModelType` would reject it one layer later. A generic alias only
   becomes meaningful once output tensor names can be matched to a family (open
   question 5).
4. **Stubs that throw, not decoders that guess.** Each family postprocessor
   documents the expected tensor layout it will implement and throws
   `std::logic_error` until that layout is verified against a real ONNX export.
   The repo's history (two coordinate-convention bugs, one normalization bug)
   is the reason: conventions are detected, not assumed.
5. **One pair preprocessor for now.** `KeypointMatchingPreprocessor` is shared
   by all three families with RGB, `[0,1]`, NCHW. It is split per family when a
   real export says the normalization or channel count differs (LightGlue's
   SuperPoint pipeline expects grayscale — open question 1). Sharing one
   parameterized preprocessor across families is the pattern the roadmap flags
   as a known source of bugs; the split is the follow-up, not the initial
   abstraction.
6. **`postprocess` receives both original sizes.** `TaskInterface::postprocess`
   carries one `frame_size`, but matches must be rescaled to two images that can
   differ in size. The task records the pair's sizes during `preprocess()` and
   passes them to the decoder; tensor-only callers that never preprocessed fall
   back to `frame_size` for both. This avoids a contract change now and unblocks
   correct rescaling later.
7. **No `TaskConfig` additions yet.** A match-score threshold
   (`confidence_threshold`) and keypoint-cap tuning arrive with the decoders
   that can honor them.

## Open questions

1. LightGlue's SuperPoint pipeline takes **grayscale** `(2B, 1, H, W)`; the
   shared preprocessor emits the source channel count. Confirm channels and
   normalization (`[0,1]`, `mean/std 0.5`?) against the export when the decoder
   lands.
2. Do pipeline exports return keypoints in **input space** or original-image
   space? Decides the rescale policy; probe, do not assume.
3. MatchAnything's ONNX export (HF Optimum? custom?) — no verified contract yet;
   its guide documents how to check `input_names` / output shapes.
4. Does any consumer need **multiple pairs per call** (`neuriplo-track` pairs
   frames)? If yes, a batch-aware design is a new packet, not a patch.
5. Could a generic `keypointmatching` alias pick a family from
   `ModelInfo::output_names` once at least one decoder is implemented? Deferred
   until there is something real to dispatch to.

## Constraint check against the constitution

- [mission.md](../mission.md): pixels-in / results-out, no runtime calls, no
  model zoo — the scaffold adds a task contract and an export guide, no
  weights or engine code.
- [tech-stack.md](../tech-stack.md): routing is two matchers — the factory entry
  **and** `KeypointMatchingTask::detectModelType` are both updated here; result
  schema change is additive.
- [REPO_META.yaml](../../REPO_META.yaml): no new runtime dependencies; OpenCV
  untouched.
