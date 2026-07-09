# Batch support matrix

Hand-maintained audit of built-in `TaskFactory` families. Batch utilities are implemented,
with domain adoption complete for classification, detection, depth, and pose. Consumer
adoption flow: [batch_processing.md](./batch_processing.md).

## How to read this table

| Column | Meaning |
|--------|---------|
| **Preprocess packs batch?** | Does `preprocess(vector<vision::Image>)` produce a **single** inference-ready tensor with leading batch `N`? |
| **Postprocess splits batch?** | Does `postprocess` return **one `Result` per batch index** when output tensors carry `N > 1`? |
| **Status** | **Ready** — both sides workable today; **Partial** — one side or tensor-only; **N/A** — not image-batch semantics. |

**Important distinction:** `vector<vision::Image>` often means **multi-input** (views, frame pairs, clip frames), not **batch N**.
`ModelInfo.batch_size_` / `max_batch_size_` are consumer hints; tasks do not enforce them today.

---

## Summary matrix

| Task family | Example `model_type` strings | Input layout (`N` axis) | Preprocess packs batch? | Postprocess splits batch? | Status | Notes |
|-------------|------------------------------|-------------------------|----------------------|---------------------------|--------|-------|
| Classification | `torchvisionclassifier`, `vitclassifier`, `tensorflowclassifier`, `resnet*` | `[N,C,H,W]` NCHW or `[N,H,W,C]` NHWC | No — one buffer per image | **Yes** — one top-1 per `shape[0]` when `N>1` | **Ready** | Full `top_k` when `N=1`; batched `[N,C]` via `postprocessClassificationLogits`. Covered by `tests/test_classification_batch.cpp`. |
| Object detection — YOLO | `yolo*`, `yolov10`, `yolo26`, `yolonas`, `yolov7e2e` | `[N,anchors,C]` or `[N,C,anchors]` | No — one buffer per image | **Yes** — `YoloPostprocessor` iterates `shape[0]`; per-slice NMS | **Ready** | Batch loop in `postprocessYoloStandard`. Covered by `tests/test_object_detection_batch.cpp`. |
| Object detection — RT-DETR / DEIM / DFINE | `rtdetr`, `rtdetrv2`, `dfine`, `deim` | `[N,Q,*]` typical | **Yes** — stacked image buffers + batched `orig_target_sizes` | **Yes** — `postprocessRTDETR` iterates `shape[0]` | **Ready** | |
| Object detection — RT-DETR Ultralytics | `rtdetrul`, `rtdetrultralytics` | `[N,C,H,W]` | No — one buffer per image | **Yes** — `postprocessRTDETRUL` iterates `shape[0]` | **Ready** | |
| Object detection — RF-DETR | `rfdetr` | `[N,C,H,W]` | No — one buffer per image | **Yes** — batch loop over `shape[0]` for boxes + labels | **Ready** | |
| Object detection — EdgeCrafter | `ecdet*`, `edgecrafter` (det) | `[N,C,H,W]` + `orig_size` | **Yes** — stacked image buffers + batched `orig_size` | **Yes** — batch loop over `shape[0]` for scores, boxes, labels | **Ready** | |
| Instance segmentation — YOLO | `yoloseg`, `yolo*seg*` | `[N,…]` det + proto heads | No — one buffer per image | **Yes** — per-batch NMS + mask gen via batch proto offset | **Ready** | |
| Instance segmentation — RF-DETR | `rfdetrseg` | `[N,C,H,W]` | No — one buffer per image | **Yes** — batch loop over `shape[0]` for boxes, labels, masks | **Ready** | |
| Instance segmentation — EdgeCrafter | `ecseg*`, `edgecrafter*seg*` | Multi-input | **Yes** — stacked image buffers + batched `orig_size` | **Yes** — batch loop over `shape[0]` for scores, boxes, labels, masks | **Ready** | |
| Pose — YOLO | `yolo*pose*` | `[N,anchors,C]` or `[N,C,anchors]` | No — one buffer per image | **Yes** — batch loop over `shape[0]`; per-batch body NMS | **Ready** | |
| Pose — ViTPose | `vitpose` | `[N,J,H,W]` heatmaps | No — one buffer per image | **Yes** — one `PoseEstimation` per `shape[0]` | **Ready** | Covered by `tests/test_pose_estimation_batch.cpp`. |
| Pose — EdgeCrafter | `ecpose*`, `edgecrafter*pose*` | Multi-input | **Yes** — stacked image buffers + batched `orig_size` | **Yes** — batch loop over `shape[0]` for scores, keypoints, labels | **Ready** | |
| Depth estimation | `*depthanythingv2*` | `[N,H,W]` or `[N,1,H,W]` / NCHW | No — one buffer per image | **Yes** — one `DepthEstimation` per batch index | **Ready** | Covered by `tests/test_depth_estimation_batch.cpp`. |
| Gaussian splatting | `lgm`, `grm`, `*splat*` | `[N,C,H,W]` views; output `[N,G,14]` | No — **one buffer per view** (`getRequiredFrames()=4`) | Partial — reads `[N,G,14]` but **one** `Result` | N/A | Multi-view ≠ image batch; merged gaussian list. |
| Video classification | `videomae`, `vivit`, `timesformer` | `[N,T,C,H,W]` clip | Temporal — **one** concatenated buffer | No — single clip `top_k` | N/A | `vector<Image>` = frame list; batch axis is clip, not images. |
| Optical flow | `raft` | `[N,2,H,W]` flow field | Pair — even `Mat` count, one buffer per pair | No — assumes `[1,2,H,W]` | N/A | `getRequiredFrames()=2`; not `N` independent images. |
| Open-vocab detection | `owlv2`, `owlvit`, `groundingdino` | `[N,C,H,W]` + text | **Yes** — stacked image buffers; text tokens shared across batch | **Yes** — batch loop over `shape[0]` for boxes + logits | **Ready** | |
| Image understanding | `gemma4`, `imageunderstanding`, `llama*` | Text + raw RGB payload | No — prompt + **one** image | No — single `ImageUnderstanding` string | N/A | Generative; single request semantics. |

**Coverage:** every `TaskFactory` registration row has an entry above (det/seg/pose EdgeCrafter grouped by routing).

---

## Preprocess behaviour (by pattern)

### A — Per-image loop (one `vector<uint8_t>` per image)

**Families:** classification, depth, YOLO detection/seg (standard), RF-DETR det/seg, YOLO pose (via `Preprocessor::preprocess(imgs)`), RT-DETR UL.

- Contract: `results.size() == imgs.size()`.
- Consumer must stack buffers into `[N,C,H,W]` (or engine-specific layout) before inference.
- Does **not** validate `imgs.size() <= ModelInfo.max_batch_size_`.

### B — Multi-input with stacking (batched image + batched side inputs)

**Families:** RT-DETR style, EdgeCrafter det/seg/pose, open-vocab.

- Preprocesses all `imgs`; concatenates per-image buffers into one batched image tensor.
- Side inputs (`orig_target_sizes`, `orig_size`) are stacked: `[N,2]` int64.
- Text token buffers are shared across the batch (encoded once, not per-image).
- `results.size() == model_info.input_names.size()` (one buffer per model input node).

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
| `YoloPosePostprocessor` | `[N,anchors,kpts+C]` or `[N,kpts+C,anchors]` | `src/pose_estimation/yolo_pose_postprocessor.cpp` |
| `YoloSegmentationPostprocessor` | `[N,…]` det + proto heads, per-batch NMS | `src/instance_segmentation/yolo_segmentation_postprocessor.cpp` |
| `RfDetrPostprocessor` | `[N,Q,4]` boxes + `[N,Q,C]` labels | `src/object_detection/rfdetr_postprocessor.cpp` |
| `RfDetrSegmentationPostprocessor` | `[N,Q,4]` + `[N,Q,C]` + `[N,Q,H,W]` masks | `src/instance_segmentation/rfdetr_segmentation_postprocessor.cpp` |
| `RtDetrPostprocessor` (UL) | `[N,Q,4+C]` combined output | `src/object_detection/rtdetr_postprocessor.cpp` |
| `RtDetrPostprocessor` (standard) | `[N,Q]` scores/labels + `[N,Q,4]` boxes | `src/object_detection/rtdetr_postprocessor.cpp` |
| `EdgeCrafterPostprocessor` | `[N,Q]` scores/labels + `[N,Q,4]` boxes | `src/object_detection/edgecrafter_postprocessor.cpp` |
| `EdgeCrafterSegmentationPostprocessor` | `[N,Q]` scores/labels + `[N,Q,4]` boxes + `[N,Q,H,W]` masks | `src/instance_segmentation/edgecrafter_segmentation_postprocessor.cpp` |
| `EdgeCrafterPosePostprocessor` | `[N,Q]` scores/labels + `[N,Q,K,D]` keypoints | `src/pose_estimation/edgecrafter_pose_postprocessor.cpp` |
| `GroundingDinoPostprocessor` | `[N,Q,4]` boxes + `[N,Q,seq_len]` logits | `src/open_vocab_detection/grounding_dino_postprocessor.cpp` |
| `OWLv2Postprocessor` | `[N,Q,4]` boxes + `[N,Q,num_prompts]` logits (+ objectness) | `src/open_vocab_detection/owlv2_postprocessor.cpp` |

### Tensor-aware but single aggregate `Result`

| Component | Behaviour |
|-----------|-----------|
| `LgmPostprocessor` | Accepts `[G,14]` or `[N,G,14]`; flattens to **one** `GaussianSplatting` with `num_gaussians = N×G`. |
| Classification / video postprocessors | Flat or `[1,C]` logits → `top_k` entries (not per-image). |
| RAFT | Expects `[1,2,H,W]`; no batch iteration. |

### Ignores leading batch dimension

_No components remain in this category — all postprocessors now iterate `shape[0]`._

---

## Domains that must stay `N = 1` (or non-image-batch)

| Domain | Why |
|--------|-----|
| Video classification | Batch axis is **clip batch**; input is a **frame sequence** in one tensor (`[N,T,C,H,W]`). Running `N=2` clips needs different `ModelInfo` and frame sourcing, not `N` unrelated `Mat`s. |
| Optical flow | Input is **frame pairs**; output is one flow field per pair. Batching is pair-wise, not independent images. |
| Gaussian splatting | Input is **fixed multi-view** (4 views); output is one splat asset. `N` in `[N,G,14]` is view-batch for the model, not consumer image batch. |
| Image understanding | Single prompt + single image (VLM decode). |

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

## Domain adoption summary

| Domain | Batch preprocess | Batch postprocess | Integration test | Status |
|--------|-----------------|-------------------|-----------------|--------|
| Classification | `batchPreprocess` → stacked `[N,C,H,W]` | `batchPostprocess` → N `Classification` results | `test_classification_batch.cpp` | **Done** |
| Object detection (YOLO) | `batchPreprocess` → one buffer per image | `YoloPostprocessor` batch loop → N `Detection` lists | `test_object_detection_batch.cpp` | **Done** |
| Object detection (RF-DETR) | `batchPreprocess` → one buffer per image | `RfDetrPostprocessor` batch loop → N `Detection` lists | — | **Done** |
| Object detection (RT-DETR UL) | `batchPreprocess` → one buffer per image | `RtDetrPostprocessor::postprocessRTDETRUL` batch loop → N `Detection` lists | — | **Done** |
| Object detection (RT-DETR standard) | Stacked image buffers + batched `orig_target_sizes` | `RtDetrPostprocessor::postprocessRTDETR` batch loop → N `Detection` lists | — | **Done** |
| Object detection (EdgeCrafter) | Stacked image buffers + batched `orig_size` | `EdgeCrafterPostprocessor` batch loop → N `Detection` lists | — | **Done** |
| Depth estimation | `batchPreprocess` → one buffer per image | `batchPostprocess` → N `DepthEstimation` results | `test_depth_estimation_batch.cpp` | **Done** |
| Pose (ViTPose) | `batchPreprocess` → per-`Mat` buffers | `batchPostprocess` → N `PoseEstimation` results | `test_pose_estimation_batch.cpp` | **Done** |
| Pose (YOLO) | `batchPreprocess` → one buffer per image | `YoloPosePostprocessor` batch loop → N `PoseEstimation` lists | — | **Done** |
| Pose (EdgeCrafter) | Stacked image buffers + batched `orig_size` | `EdgeCrafterPosePostprocessor` batch loop → N `PoseEstimation` lists | — | **Done** |
| Instance segmentation (YOLO) | `batchPreprocess` → one buffer per image | `YoloSegmentationPostprocessor` per-batch NMS + mask gen → N `InstanceSegmentation` lists | — | **Done** |
| Instance segmentation (RF-DETR) | `batchPreprocess` → one buffer per image | `RfDetrSegmentationPostprocessor` batch loop → N `InstanceSegmentation` lists | — | **Done** |
| Instance segmentation (EdgeCrafter) | Stacked image buffers + batched `orig_size` | `EdgeCrafterSegmentationPostprocessor` batch loop → N `InstanceSegmentation` lists | — | **Done** |
| Open-vocab detection | Stacked image buffers; shared text tokens | `GroundingDinoPostprocessor` / `OWLv2Postprocessor` batch loop → N `OpenVocabDetection` lists | — | **Done** |

---

## Maintenance

Update this file when:

- A task gains true `N`-image preprocess packing or per-index `Result` output.
- A new `TaskFactory` family is registered.
- A domain moves from **Partial** to **Ready** (both preprocess and postprocess columns become Yes).

All planned batch utilities, their integration coverage, and the consumer guide are complete. Keep this matrix current as task families evolve.
