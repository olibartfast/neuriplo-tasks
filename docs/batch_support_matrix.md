# Batch support matrix (B6 audit)

Hand-maintained audit of built-in `TaskFactory` families as of **Track B6**
(all batch utilities landed; domain adoption complete for classification,
detection, depth, and pose). Consumer adoption flow: [batch_processing.md](./batch_processing.md).

## How to read this table

| Column | Meaning |
|--------|---------|
| **Preprocess packs batch?** | Does `preprocess(vector<cv::Mat>)` produce a **single** inference-ready tensor with leading batch `N`? |
| **Postprocess splits batch?** | Does `postprocess` return **one `Result` per batch index** when output tensors carry `N > 1`? |
| **Status** | **Ready** — both sides workable today; **Partial** — one side or tensor-only; **N/A** — not image-batch semantics. |

**Important distinction:** `vector<cv::Mat>` often means **multi-input** (views, frame pairs, clip frames), not **batch N**.
`ModelInfo.batch_size_` / `max_batch_size_` are consumer hints; tasks do not enforce them today.

---

## Summary matrix

| Task family | Example `model_type` strings | Input layout (`N` axis) | Preprocess packs batch? | Postprocess splits batch? | Status | Notes |
|-------------|------------------------------|-------------------------|----------------------|---------------------------|--------|-------|
| Classification | `torchvisionclassifier`, `vitclassifier`, `tensorflowclassifier`, `resnet*` | `[N,C,H,W]` NCHW or `[N,H,W,C]` NHWC | No — one buffer per `Mat` | **Yes** — one top-1 per `shape[0]` when `N>1` | **Ready** | Full `top_k` when `N=1`; batched `[N,C]` via `postprocessClassificationLogits`. Covered by `tests/test_classification_batch.cpp`. |
| Object detection — YOLO | `yolo*`, `yolov10`, `yolo26`, `yolonas`, `yolov7e2e` | `[N,anchors,C]` or `[N,C,anchors]` | No — one buffer per `Mat` | **Yes** — `YoloPostprocessor` iterates `shape[0]`; per-slice NMS | **Ready** | Batch loop in `postprocessYoloStandard`. Covered by `tests/test_object_detection_batch.cpp`. |
| Object detection — RT-DETR / DEIM / DFINE | `rtdetr`, `rtdetrv2`, `dfine`, `deim` | `[N,Q,*]` typical | No — **single image** + `orig_target_sizes` side input | No — indexes `scores.shape[1]` as query count | Partial | `preprocess` uses `imgs[0]` only; multi-input ≠ batch. |
| Object detection — RT-DETR Ultralytics | `rtdetrul`, `rtdetrultralytics` | `[N,C,H,W]` | No — one buffer per `Mat` | No — single-output decode | Partial | Same as YOLO path for preprocess. |
| Object detection — RF-DETR | `rfdetr` | `[N,C,H,W]` | No — one buffer per `Mat` | No — query-axis decode, no batch loop | Partial | |
| Object detection — EdgeCrafter | `ecdet*`, `edgecrafter` (det) | `[N,C,H,W]` + `orig_size` | No — **single image** + orig size tensor | No | Partial | Same multi-input pattern as RT-DETR style. |
| Instance segmentation — YOLO | `yoloseg`, `yolo*seg*` | `[N,…]` det + proto heads | No — one buffer per `Mat` | No — instance list, no batch index | Partial | Proto/det heads assume `N=1` in practice. |
| Instance segmentation — RF-DETR | `rfdetrseg` | `[N,C,H,W]` | No — one buffer per `Mat` | No | Partial | |
| Instance segmentation — EdgeCrafter | `ecseg*`, `edgecrafter*seg*` | Multi-input | No — **single image** | No | Partial | |
| Pose — YOLO | `yolo*pose*` | `[N,anchors,C]` or `[N,C,anchors]` | Partial — `Preprocessor::preprocess(imgs)` loops per `Mat` | No — anchor loop, batch dim unused | Partial | |
| Pose — ViTPose | `vitpose` | `[N,J,H,W]` heatmaps | Partial — per-`Mat` buffers | **Yes** — one `PoseEstimation` per `shape[0]` | **Ready** | Covered by `tests/test_pose_estimation_batch.cpp`. |
| Pose — EdgeCrafter | `ecpose*`, `edgecrafter*pose*` | Multi-input | No — **single image** | No | Partial | |
| Depth estimation | `*depthanythingv2*` | `[N,H,W]` or `[N,1,H,W]` / NCHW | No — one buffer per `Mat` | **Yes** — one `DepthEstimation` per batch index | **Ready** | Covered by `tests/test_depth_estimation_batch.cpp`. |
| Gaussian splatting | `lgm`, `grm`, `*splat*` | `[N,C,H,W]` views; output `[N,G,14]` | No — **one buffer per view** (`getRequiredFrames()=4`) | Partial — reads `[N,G,14]` but **one** `Result` | N/A | Multi-view ≠ image batch; merged gaussian list. |
| Video classification | `videomae`, `vivit`, `timesformer` | `[N,T,C,H,W]` clip | Temporal — **one** concatenated buffer | No — single clip `top_k` | N/A | `vector<Mat>` = frame list; batch axis is clip, not images. |
| Optical flow | `raft` | `[N,2,H,W]` flow field | Pair — even `Mat` count, one buffer per pair | No — assumes `[1,2,H,W]` | N/A | `getRequiredFrames()=2`; not `N` independent images. |
| Open-vocab detection | `owlv2`, `owlvit`, `groundingdino` | `[N,C,H,W]` + text | No — **single image** + token buffers | No — flat detection list | Partial | Grounding DINO comments `[batch,…]`; no batch loop in decode. |
| Image understanding | `gemma4`, `imageunderstanding`, `llama*` | Text + raw RGB payload | No — prompt + **one** image | No — single `ImageUnderstanding` string | N/A | Generative; single request semantics. |

**Coverage:** every `TaskFactory` registration row has an entry above (det/seg/pose EdgeCrafter grouped by routing).

---

## Preprocess behaviour (by pattern)

### A — Per-image loop (one `vector<uint8_t>` per `Mat`)

**Families:** classification, depth, YOLO detection/seg (standard), RF-DETR det/seg, YOLO pose (via `Preprocessor::preprocess(imgs)`), RT-DETR UL.

- Contract: `results.size() == imgs.size()`.
- Consumer must stack buffers into `[N,C,H,W]` (or engine-specific layout) before inference.
- Does **not** validate `imgs.size() <= ModelInfo.max_batch_size_`.

### B — Multi-input, single spatial image

**Families:** RT-DETR style, EdgeCrafter det/seg/pose, open-vocab.

- Uses `imgs[0]` only; extra inputs (`orig_target_sizes`, `input_ids`, …) derived from that image.
- `results.size() == model_info.input_names.size()`, not `imgs.size()`.

### C — Temporal / multi-view (not batch N)

| Family | `getRequiredFrames()` | Behaviour |
|--------|----------------------|-----------|
| Video classification | `num_frames_` (from 5D shape, default 16) | Concatenates frames into **one** buffer; pads by repeating last frame. |
| Optical flow | `2` | Requires even count; preprocesses **pairs** sequentially. |
| Gaussian splatting | `4` (LGM/GRM) | One buffer per view image; trained for 4 views, not `N` arbitrary images. |

### D — Generative dual payload

**Image understanding:** `[prompt_bytes, image_bytes]` or prompt-only; never batching images.

---

## Postprocess behaviour (by pattern)

### Splits batch → `vector<Result>` size `N`

| Component | Layout handled | Source |
|-----------|----------------|--------|
| `DepthAnythingV2Postprocessor` | `[N,H,W]`, `[N,1,H,W]`, NCHW 4D | `src/depth_estimation/depth_anything_v2_postprocessor.cpp` |
| `ViTPosePostprocessor` | `[N,J,H,W]` | `src/pose_estimation/vit_pose_postprocessor.cpp` |
| `YoloPostprocessor` (standard) | `[N,anchors,cls+4]` or `[N,cls+4,anchors]` | `src/object_detection/yolo_postprocessor.cpp` |

### Tensor-aware but single aggregate `Result`

| Component | Behaviour |
|-----------|-----------|
| `LgmPostprocessor` | Accepts `[G,14]` or `[N,G,14]`; flattens to **one** `GaussianSplatting` with `num_gaussians = N×G`. |
| Classification / video postprocessors | Flat or `[1,C]` logits → `top_k` entries (not per-image). |
| RAFT | Expects `[1,2,H,W]`; no batch iteration. |

### Ignores leading batch dimension

| Component | Evidence |
|-----------|----------|
| `YoloPosePostprocessor` | Anchor loop only; no `shape[0]` loop |
| `RtDetrPostprocessor` | `num_dets = scores.shape[1]` |
| `GroundingDinoPostprocessor` | Queries over `num_queries`; batch in comment only |

---

## Domains that must stay `N = 1` (or non-image-batch)

| Domain | Why |
|--------|-----|
| Video classification | Batch axis is **clip batch**; input is a **frame sequence** in one tensor (`[N,T,C,H,W]`). Running `N=2` clips needs different `ModelInfo` and frame sourcing, not `N` unrelated `Mat`s. |
| Optical flow | Input is **frame pairs**; output is one flow field per pair. Batching is pair-wise, not independent images. |
| Gaussian splatting | Input is **fixed multi-view** (4 views); output is one splat asset. `N` in `[N,G,14]` is view-batch for the model, not consumer image batch. |
| Image understanding | Single prompt + single image (VLM decode). |
| RT-DETR / EdgeCrafter / open-vocab (current code) | Preprocess hard-codes first image; text prompts are global per request. |

---

## `ModelInfo` batch fields (baseline)

| Field | Used by tasks today? |
|-------|---------------------|
| `batch_size_`, `max_batch_size_` | Set by `addInput`; **not** read in task `preprocess`/`postprocess` |
| `input_batch_sizes`, `output_batch_sizes` | Stored; **not** enforced in C++ tasks |

Tests default `batch_size_ = 1` (`tests/test_task_factory.cpp`, `tests/test_readme_model_types.cpp`).

---

## Current batch test coverage

| Test file | What it proves |
|-----------|----------------|
| `tests/test_batch_types.cpp` | `BatchRequest`/`BatchPreprocessOutput`/`BatchPostprocessOutput` construction and invariants |
| `tests/test_batch_preprocess.cpp` | `batchPreprocess()` packs N images with per-image buffers and batch size metadata |
| `tests/test_batch_postprocess.cpp` | `batchPostprocess()` splits results by batch index |
| `tests/test_batch_integration.cpp` | Full round-trip: preprocess N=2 → inference → postprocess, results aligned to batch indices |
| `tests/test_classification_batch.cpp` | Classification `[N,C]` logits → N top-1 results |
| `tests/test_depth_estimation_batch.cpp` | Depth `[N,H,W]` / `[N,1,H,W]` → N depth maps |
| `tests/test_pose_estimation_batch.cpp` | ViTPose `[N,J,H,W]` → N pose results |
| `tests/test_object_detection_batch.cpp` | YOLO `[N,anchors,cls+4]` → N detection lists, per-slice NMS |
| `tests/test_yolo_postprocessor.cpp` | `N=1` YOLO decode (baseline) |
| `tests/test_depth_estimation.cpp` | `N=1` depth map round-trip (baseline) |
| `tests/test_pose_estimation.cpp` | ViTPose `N=1` heatmaps (baseline) |
| `tests/test_gaussian_splatting.cpp` | `[N,G,14]` layout with `N=1` |
| `tests/test_optical_flow.cpp` | `[1,2,H,W]` RAFT output |

---

## Domain adoption summary (B4/B5)

| Domain | Batch preprocess | Batch postprocess | Integration test | Status |
|--------|-----------------|-------------------|-----------------|--------|
| Classification | `batchPreprocess` → stacked `[N,C,H,W]` | `batchPostprocess` → N `Classification` results | `test_classification_batch.cpp` | **Done** |
| Object detection (YOLO) | `batchPreprocess` → one buffer per `Mat` | `YoloPostprocessor` batch loop → N `Detection` lists | `test_object_detection_batch.cpp` | **Done** |
| Depth estimation | `batchPreprocess` → one buffer per `Mat` | `batchPostprocess` → N `DepthEstimation` results | `test_depth_estimation_batch.cpp` | **Done** |
| Pose (ViTPose) | `batchPreprocess` → per-`Mat` buffers | `batchPostprocess` → N `PoseEstimation` results | `test_pose_estimation_batch.cpp` | **Done** |

---

## Maintenance

Update this file when:

- A task gains true `N`-image preprocess packing or per-index `Result` output.
- A new `TaskFactory` family is registered.
- A domain moves from **Partial** to **Ready** (both preprocess and postprocess columns become Yes).

**B6 stop criteria:** met — B0 audit complete, B1–B3 utilities landed, B4 domain adoption complete, B5 integration tests pass, B6 consumer guide published.
