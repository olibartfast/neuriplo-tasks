# Keypoint matching (SuperGlue family) — requirements

**Date:** 2026-10-15
**Branch:** `feature/keypoint-matching` → `develop`
**Status:** SuperGlue decoder delivered and verified against a real ONNX export;
LightGlue and MatchAnything decoding still deferred (see Decisions).

## SuperGlue contract (verified 2026-11-02)

Probed against `KnightsAnalytics/superglue_outdoor` (an Optimum export of
`magic-leap-community/superglue_outdoor`, transformers 5.20, opset 17), whose
repo also ships `reference_*.npz` inputs/outputs and an `export_verification.json`.
Reproduce with [`export/keypoint_matching/probe_superglue.py`](../../export/keypoint_matching/probe_superglue.py).

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `pixel_values` (in) | `[1, 2, 3, 480, 640]` float32 | **one** input; pair on axis 1; grayscale replicated to 3 channels; `[0, 1]`; no mean/std |
| `keypoints` | `[1, 2, N, 2]` float32 | normalized `x / width`, `y / height` |
| `matches` | `[1, 2, N]` int64 | side 0 → index into side 1's keypoints, or `-1`; side 1 is the inverse map |
| `matching_scores` | `[1, 2, N]` float32 | mutual-match probability |
| `mask` | `[1, 2, N]` int64 | `1` = detected keypoint, `0` = padding (packed at the front) |

`N` is dynamic — 568 for the reference pair's blank image, 903 for the
phototourism Capitol pair — so decoders must read it from the tensor, never
assume it. The export carries no SuperPoint detector score, so
`Keypoint::confidence` cannot be filled. Preprocessing is resize-to-`640x480`
bilinear, BGR→RGB, torchvision grayscale weights (`0.2989, 0.5870, 0.1140`),
`[0, 1]`, three replicated channels.

Two consequences that changed the scaffold design:

1. **One buffer, not two.** `preprocess()` returns buffers that map 1:1 onto
   model inputs. This export has a single input holding the pair, so
   `SuperGluePreprocessor` overrides `preprocess_pair()` and emits one packed
   `[2, 3, 480, 640]` buffer (source first). The two-input families keep the
   one-buffer-per-image behaviour.
2. **The shared preprocessor was wrong for SuperGlue.** It emitted RGB at
   `[0, 1]`; the verified export needs grayscale. This is the split that
   decision 5 anticipated, executed on evidence rather than assumption.

## Goal

A `KeypointMatchingTask` that finds correspondences between the two images of a
pair, so consumers stop reimplementing SuperGlue-family preprocessing and
decoding per project. This packet delivered the **scaffold** — the task
contract, the result type, factory routing, the preprocessing seam, and the
tests that lock them — and now adds the **SuperGlue decoder**, built only after
the export's tensor layout was probed against a real ONNX file (see the
contract section above). LightGlue and MatchAnything decoders land the same way.

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
- `SuperGluePreprocessor`: the same resize path plus grayscale and pair packing
  into the single `pixel_values` tensor, one buffer total.
- `SuperGluePostprocessor`: mask-aware decode of `keypoints` / `matches` /
  `matching_scores`, normalized-to-original coordinate rescaling, and score
  thresholding from `TaskConfig::confidence_threshold`.
- Per-family postprocessor seams: `LightGluePostprocessor`,
  `MatchAnythingPostprocessor` (still throwing).
- `TaskFactory` routing: `superglue*`, `lightglue*`, `matchanything*` by prefix.
- Tests: routing and alias normalization, pair validation, preprocessing
  buffers, postprocess seams, result-variant coverage.
- README `## Features` bullet and `TASKFACTORY_MODEL_LIST` block;
  `export/keypoint_matching/README.md`; batch support matrix row; changelog;
  roadmap.

## Out of scope

- **Decoding LightGlue and MatchAnything tensors.** Those postprocessors throw
  `std::logic_error` until an export's layout is verified against a real file.
  Writing a decoder from assumed layouts is how the RT-DETR/EdgeCrafter
  normalization bugs happened.
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
5. **The shared preprocessor was split on evidence.** The scaffold shipped one
   RGB `[0,1]` preprocessor for all three families and recorded grayscale as an
   open question. The verified SuperGlue export needs grayscale and a single
   packed pair tensor, so `SuperGluePreprocessor` overrides `preprocess_pair()`
   while `KeypointMatchingPreprocessor` keeps the two-buffer behaviour for
   LightGlue and MatchAnything. One shared parameterized preprocessor across
   families is the pattern the roadmap flags as a bug source; the split is now
   per family, each awaiting its own export.
6. **`postprocess` receives both original sizes.** `TaskInterface::postprocess`
   carries one `frame_size`, but matches must be rescaled to two images that can
   differ in size. The task records the pair's sizes during `preprocess()` and
   passes them to the decoder; tensor-only callers that never preprocessed fall
   back to `frame_size` for both. This avoids a contract change now and unblocks
   correct rescaling later.
7. **The score threshold uses the existing knob.** `SuperGluePostprocessor` takes
   a match-score threshold that the task fills from
   `TaskConfig::confidence_threshold` (default `0.25`), so no new field was
   added and consumers configure matches like detections. `KeypointMatch::score`
   carries the raw probability, so a caller who wants every match can set the
   threshold to `0`. Keypoint-cap tuning stays deferred.

## Open questions

1. ~~LightGlue's SuperPoint pipeline takes **grayscale** `(2B, 1, H, W)`.~~
   **SuperGlue answered (2026-11-02):** grayscale, three replicated channels,
   `[0, 1]`, no mean/std. LightGlue's own channel count and normalization stay
   open until its export is probed.
2. ~~Do pipeline exports return keypoints in **input space** or original-image
   space?~~ **Answered (2026-11-02):** SuperGlue returns normalized
   `x / width` and `y / height`, so each image's original size rescales it
   directly; no letterbox offset is involved because the export stretches
   rather than pads.
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
