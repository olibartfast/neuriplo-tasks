# Depth Anything 3 — Requirements

## Goal

Support Depth Anything 3 (DA3) monocular relative-depth models in
`DepthEstimationTask`: preprocess (504 default, multiples of 14, ImageNet),
decode the `[1,H,W]` depth output, route factory names, and document the export
contract. No new `Result` type, no new task class.

## Decisions

- v1 covers **relative-depth** DA3 only (`da3-small/base/large/giant`,
  `da3mono-large`, `depth-anything-3` naming). Metric (`da3metric-*`), Nested
  (already meters), and camera-depth (4-channel input) models are explicitly
  rejected with an unsupported-model error: metric scaling needs a focal length
  the task contract (`ModelInfo`/`TaskConfig`) does not carry. Focal plumbing
  is a follow-up slice, not this one.
- **Reuse, don't duplicate**: DA3 relative output (`depth [1,H,W]`, optional
  `conf [1,H,W]`, names `depth`/`conf` per the DA3 `export_onnx.py` contract)
  is layout-compatible with `DepthAnythingV2Postprocessor` (`[B,H,W]` +
  resize-to-frame + min/max normalize), so DA3 reuses it directly. No new
  postprocessor class in v1.
- **New preprocessor, same config**: `DepthAnything3Preprocessor` keeps the V2
  preprocessing (RGB, `[0,1]`, ImageNet, NCHW float32) with a 504 default
  instead of 518. `extractInputSize` falls back to 504 for DA3 when
  `ModelInfo` carries no shapes.
- **Factory**: `contains "depthanything3"` or `startsWith "da3"` routes to
  `DepthEstimationTask`; the task constructor throws for metric/nested names.
  Descriptor sits with the other depth guards (after YOLO-depth).
- The optional `conf` output tensor is ignored in v1: `DepthEstimation` has no
  confidence field and decode reads tensor 0 only.

## Scope

In:

- `ModelType::DEPTH_ANYTHING_3`, detection (metric/nested reject), preprocessor,
  factory descriptor.
- Unit tests: DA3 preprocess (explicit + 504 fallback shapes), synthetic
  `[1,H,W]` postprocess through the task, metric-name rejection, factory
  routing + alias normalization entries.
- README `TASKFACTORY_MODEL_LIST` block (auto-checked by
  `test_readme_model_types`), Features bullet if depth line lists models,
  `export/depth_estimation/depth_anything_3/README.md` setup guide,
  `export/README.md` tree entry, CHANGELOG `[Unreleased]`, roadmap link.

Out:

- Metric/nested/camera-depth support, `conf` map decoding, focal-length
  plumbing, real weights/exports (guides only, no binaries).
- New `Result` variant members, `TaskType` changes, CMake changes.

## Contracts

- Input: RGB float32 NCHW `[B,3,H,W]` (from `ModelInfo`), BGR→RGB, resize,
  `[0,1]` + ImageNet mean/std. Default geometry 504x504; dims multiple of 14
  per the DA3 RoPE constraint (enforced at export, not at runtime).
- Output tensor 0: `[B,H,W]` (also `[B,1,H,W]`/`[H,W]` via the shared layout
  parser) float disparity/depth; resized to frame, min/max normalized.
- Result: one `DepthEstimation` per batch row (`depth`, `normalized_depth`,
  `min_depth`, `max_depth`).
