# Video model families (V-JEPA 2, VideoPrism, PE Video) — requirements

**Date:** 2026-11-03
**Branch:** `feature/video-models-pe-vjepa2-videoprism` → `develop` (merged in
[PR #23](https://github.com/olibartfast/neuriplo-tasks/pull/23), commit `2ab2c4b`)
**Status:** preprocessors and routing delivered; V-JEPA 2 verified against a real
ONNX export, VideoPrism and PE Video verified against their processor contracts only.

## Goal

Route the three newer video-classification families through the existing
`VideoClassificationTask` so consumers do not reimplement per-checkpoint
preprocessing, without adding a task contract or a new result type.

## Contracts (from the checkpoints and transformers processors)

| Family | Input | Resize | Normalization | Frames |
|--------|-------|--------|---------------|--------|
| V-JEPA 2 (`facebook/vjepa2-*`) | `[1,T,3,256,256]` | shortest edge `crop*256/224` = 292, center crop | ImageNet | `frames_per_clip` (16/32/64) |
| VideoPrism (`google/videoprism-*`) | `[1,16,3,288,288]` | direct resize, bicubic | `1/255` only | 16 base / 8 large |
| PE Video (`facebook/pe-av-large`) | `[1,T,3,336,336]` | direct resize | `1/255`, mean/std 0.5 | 8–16 |

Sources: `VJEPA2VideoProcessor` defaults (`size.shortest_edge=292`,
`crop_size=256`, ImageNet mean/std), `VideoPrismProcessor` defaults
(`size 288x288`, `do_normalize=False`, `resample=3`), and
`facebook/pe-av-large`'s `video_preprocessor_config.json`
(`size 336x336`, `image_mean/std=[0.5]`).

## In scope

- `PeVideoPreprocessor`, `Vjepa2Preprocessor`, `VideoPrismPreprocessor`.
- `VideoClassificationTask::ModelType` gains `PE_VIDEO`, `VJEPA2`, `VIDEOPRISM`,
  with family detection by name and per-family default input sizes.
- Factory routing: `pevideo*`, `vjepa*`, `videoprism*` by prefix.
- Tests: routing (including revision suffixes), one-buffer-per-clip sizing at the
  family resolution, and per-family normalization values.
- README Features bullet and alias list, export guide, batch matrix, changelog.

## Out of scope

- **A new task or result type.** These families emit `logits [1, num_classes]`, so
  the existing softmax top-k postprocessor applies unchanged.
- **PE Video as a raw export.** `PeVideoModel` is video–text contrastive and takes
  tokenized text; classification needs a fixed-vocabulary export with baked text
  embeddings, documented in the export guide. The encoder/embedding path is not a
  classification task.
- **VideoPrism / PE Video ONNX verification.** Neither has a usable pre-exported
  ONNX file: `google/videoprism-*` publishes Flax weights only, and PE Video has no
  classification export. Their preprocessing is verified against the processor
  configuration and a reference reimplementation, not a model run.
- Frame sampling from a video file — the task still receives a frame list.

## Decisions

1. **No postprocessor change.** All three emit logits, so `VideoClassificationPostprocessor`
   (softmax + top-k) stays as it is; only preprocessing and routing differ.
2. **Frames come from the input shape.** `extractVideoInputSize` already reads
   `shape[1]` of a 5D tensor, so `getRequiredFrames()` follows the checkpoint's
   `frames_per_clip` without a config field. Defaults (16) only apply to models
   whose `ModelInfo` carries no 5D shape.
3. **Family detection by substring** (`videoprism`, `vjepa`, `pevideo`) rather than
   equality, so revision suffixes (`vjepa2.1-vitl`, `videoprism-base-f16r288`)
   resolve without another patch.
