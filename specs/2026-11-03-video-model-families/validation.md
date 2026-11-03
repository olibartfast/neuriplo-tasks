# Video model families — validation

Checks are defined against the requirements; a check that was not run is recorded
as such, not as a pass.

## Checks

| # | Requirement | Check | Result |
|---|-------------|-------|--------|
| V-1 | Routing | `test_video_classification` and `test_task_factory` route `pevideo`, `pe-video`, `PE Video`, `vjepa2`, `vjepa2-vitl-fpc16-256-ssv2`, `vjepa2.1-vitl`, `videoprism`, `videoprism-base-f16r288` to `VideoClassification`; `slowfast` is rejected | **pass** |
| V-2 | Frames | `getRequiredFrames()` equals the 5D input's time axis (16 in the test's `ModelInfo`) | **pass** |
| V-3 | Buffer shape | One buffer per clip, `frames × 3 × H × W × 4` bytes, at each family's resolution (336 PE Video, 256 V-JEPA 2, 288 VideoPrism) | **pass** |
| V-4 | Normalization | Reference pixel: VideoPrism `[0,0,1]` (no mean/std), PE Video `[-1,-1,1]` (mean/std 0.5), V-JEPA 2 `[-2.1179,-2.0357,2.64]` (ImageNet) | **pass** |
| V-5 | Postprocess | Logits `{0.1,2.0,0.5,1.5}` → top class 1 with softmax 0.5052, second class 3 | **pass** |
| V-6 | V-JEPA 2 real export | ONNX run on a 32-frame clip built from a real image: input `pixel_values_videos [batch, num_frames, 3, H, W]`, output `logits [batch, 48]`; top-5 identical to the transformers recipe | **pass** |
| V-7 | VideoPrism / PE Video real export | No usable export exists (Flax-only weights; PE Video has no classification graph) — preprocessing verified against the processor configuration only | **not executed** |
| V-8 | Gate | clang-format-18 `--dry-run --Werror`, cppcheck, `-DWERROR=ON` build, `ctest` (31 suites), Valgrind on the new suite | **pass** |

## Results (2026-11-03)

**Merged:** PR [#23](https://github.com/olibartfast/neuriplo-tasks/pull/23) into
`develop` (`2ab2c4b`).

**V-JEPA 2 export:** `onnx-community/vjepa2-vitl-fpc32-256-diving48-ONNX`
(`onnx/model.onnx`, 1.4 GB). Frame: the magicleap phototourism Capitol image
(1016x762), repeated 32 times to fill the clip.

| Check | Evidence |
|-------|----------|
| V-1..V-2 | `test_video_classification` 7/7, `test_task_factory` 24/24. |
| V-3 | `PreprocessesOneBufferPerClipAtFamilySize`: 25,165,824 bytes for `1 × 32 × 3 × 256 × 256` float32. |
| V-4 | The three normalization tests pass to `1e-5` (ImageNet within `1e-4`). |
| V-6 | ONNX Runtime reports `pixel_values_videos [batch_size, num_frames, 3, height, width]` → `logits [batch_size, 48]`. The C++ buffer and a Python reimplementation of the transformers recipe produced **the same top-5** (`36, 29, 47, 16, 45`) and the same top-1; logits differ by at most `1.02` on a scale of ~6.8 because the two resize kernels differ. |
| V-8 | `clang-format-18 --dry-run --Werror` clean; `cppcheck` exit 0; clang-tidy-18 clean on the touched sources; `-DWERROR=ON` build OK; `ctest` 31/31; Valgrind on `test_video_classification` 0 errors, 0 leaks. |

## Deviations recorded

- **Preprocessing matches on predictions, not on pixels.** PIL's bilinear resize is
  antialiased on downscale; this library's `Linear` kernel is plain bilinear. On a
  1016→389 downscale the pixel difference reaches `0.98` in normalized units at
  sharp edges (`mean 0.057`), yet the classifier's top-5 is unchanged. Same finding
  as the SuperGlue work; recorded rather than hidden.
- **VideoPrism is bicubic and unnormalized.** It is the first family here that adds
  no mean/std, so its preprocessor deliberately has no `kMean`/`kStd` members.
- **PE Video cannot be classified from a raw export.** `PeVideoModel` emits
  `logits_video_text` and needs tokenized text, so the task requires a
  fixed-vocabulary export with baked text embeddings. Routing and preprocessing are
  in place; the export recipe is documented and the C++ path is untested against a
  model because none exists yet.
