# Batch processing — consumer guide

How to run **N independent images** through a single `TaskInterface` using the
Track B helpers (`batchPreprocess`, `batchPostprocess`). Intended for downstream
repos such as [tritonic](https://github.com/olibartfast/tritonic) and
[neuriplo-infer](https://github.com/olibartfast/neuriplo-infer).

**Related**

| Document | Use when |
|----------|----------|
| [batch_support_matrix.md](./batch_support_matrix.md) | Per-task-family readiness (Ready / Partial / N/A) |
| [ROADMAP.md](./ROADMAP.md) | Track B step history and guardrails |
| [README.md](../README.md) § Batch Processing Utilities | Short API snippet |

---

## Responsibilities

| Layer | Owns |
|-------|------|
| **neuriplo-tasks** | Per-image preprocess (`TaskInterface::preprocess`), batched-tensor postprocess split where implemented, `ModelInfo` validation in batch helpers |
| **Inference engine** | Stack `pre.buffers` into `[N,C,H,W]` (or engine layout), run ONNX/Triton/TensorRT, return `vector<Tensor>` with leading batch `N` |
| **Consumer app** | Image sourcing, dynamic batch sizing policy, mapping `post.results` back to request IDs |

neuriplo-tasks does **not** schedule GPU queues, allocate device memory, or call an
inference runtime. Helpers only wrap existing task code and attach `batch_size`
metadata.

---

## `ModelInfo` batch fields

Set these when you build `ModelInfo` from your model config (ONNX shapes, Triton
`config.pbtxt`, etc.):

| Field | Meaning |
|-------|---------|
| `input_shapes` | Include leading `N` when the engine runs batched inference, e.g. `{2, 3, 224, 224}` |
| `batch_size_` | Hint for the **current** inference request (often equals `request.images.size()`) |
| `max_batch_size_` | Upper bound; `batchPreprocess` / `batchPostprocess` throw if `N` exceeds this when `> 0` |
| `input_batch_sizes` / `output_batch_sizes` | Optional per-I/O batch dims (from tooling); not enforced inside tasks today |

`batch_size_` does not change preprocess output layout: you still get **one buffer
per `cv::Mat`**. The engine must concatenate or bind those buffers into a single
batched input tensor.

---

## Worked example — classification, N = 2

Matches `tests/test_batch_integration.cpp` (`ClassificationPreprocessPostprocessPipeline`).

### 1. Configure the task

```cpp
#include <neuriplo/tasks/core/batch_postprocess.hpp>
#include <neuriplo/tasks/core/batch_preprocess.hpp>
#include <neuriplo/tasks/core/task_config.hpp>
#include <neuriplo/tasks/core/task_factory.hpp>

using namespace neuriplo_tasks;

ModelInfo model_info;
model_info.input_shapes = {{2, 3, 224, 224}};   // NCHW, N = 2
model_info.input_formats = {"FORMAT_NCHW"};
model_info.input_names = {"input"};
model_info.output_names = {"output"};
model_info.input_types = {CV_32F};
model_info.batch_size_ = 2;
model_info.max_batch_size_ = 2;

TaskConfig config;
config.apply_softmax = false;

auto task = TaskFactory::createTaskInstance("resnet50", model_info, config);
```

### 2. Preprocess (library)

```cpp
BatchRequest request;
request.images = {image_a, image_b};   // two independent cv::Mat frames

BatchPreprocessOutput pre = batchPreprocess(*task, request);
// pre.batch_size == 2
// pre.buffers.size() == 2  (one preprocessed buffer per image)
```

`imageBatchSizeMatches(request, pre.batch_size)` should be true for standard
image-batch tasks.

### 3. Inference (engine — not in neuriplo-tasks)

Stack `pre.buffers` into the layout your runtime expects (example: single input
`[2, 3, 224, 224]` float tensor). Run the model. Build output tensors for
postprocess, e.g. logits shape `[2, num_classes]`.

### 4. Postprocess (library)

```cpp
cv::Size frame_size = request.images[0].size();   // original frame for letterbox undo
std::vector<Tensor> output_tensors = { /* engine output */ };

BatchPostprocessOutput post =
    batchPostprocess(*task, frame_size, output_tensors, pre.batch_size);
```

For classification with `N > 1`:

- `post.batch_size == 2`
- `post.results.size() == 2` — one `Classification` per image (top-1 per batch index)
- `postprocessResultsMatchBatchSize(post)` is true

With `N == 1`, classification still returns up to `TaskConfig::top_k` entries in
one or more `Classification` results as before.

---

## Worked example — YOLO detection, N = 2

Same integration test pattern (`ObjectDetectionPreprocessPostprocessPipeline`).

```cpp
ModelInfo model_info;
model_info.input_shapes = {{2, 3, 640, 640}};
model_info.input_formats = {"FORMAT_NCHW"};
model_info.input_names = {"images"};
model_info.output_names = {"output0"};
model_info.batch_size_ = 2;
model_info.max_batch_size_ = 2;

auto task = TaskFactory::createTaskInstance("yolov8", model_info);

BatchRequest request;
request.images = {frame_a, frame_b};

auto pre = batchPreprocess(*task, request);
// Engine: batched input [2,3,640,640], output e.g. [2, 4+num_classes, anchors]

auto post = batchPostprocess(*task, cv::Size(640, 640), output_tensors, pre.batch_size);
```

**Detection result shape**

- `post.batch_size` stays **2** (image count).
- `post.results.size()` may be **greater than 2** — one `Detection` per box, not
  per image. Filter by decoding order / slice if you need per-image lists (see
  YOLO batch slice loop in `yolo_postprocessor.cpp`).
- `postprocessResultsMatchBatchSize(post)` is **false** for detection; that is expected.

Export batched YOLO ONNX with a dynamic batch axis; see
[export/detection/ObjectDetection.md](../export/detection/ObjectDetection.md).

---

## Migrating from batch_size = 1

| Before (N = 1) | After (N > 1) |
|----------------|---------------|
| `task->preprocess({mat})` | `batchPreprocess(*task, {mat_a, mat_b, ...})` |
| `task->postprocess(size, tensors)` | `batchPostprocess(*task, size, tensors, pre.batch_size)` |
| `ModelInfo` shape `[1, C, H, W]` | Shape `[N, C, H, W]` and `max_batch_size_ >= N` |
| Single `Classification` / flat detections | See [batch_support_matrix.md](./batch_support_matrix.md) for per-family split rules |

**Checklist**

1. Confirm task family is **Ready** or **Partial** in the support matrix.
2. Export or configure the engine model with dynamic (or fixed) batch on axis 0.
3. Set `max_batch_size_` on `ModelInfo` to match the engine cap.
4. Keep `frame_size` as the **original** image size used for coordinate remap (first
   image is typical when all frames share the same letterbox policy).
5. Add engine-side tests that `[N, …]` outputs match what neuriplo-tasks postprocessors expect.

**Do not** treat these as image batches without reading the matrix:

- Video classification (`vector<Mat>` = frame list for one clip)
- Optical flow (pairs of frames)
- Gaussian splatting (multi-view set)
- RT-DETR / EdgeCrafter / open-vocab (multi-input, often `imgs[0]` only today)

---

## `batchPostprocess` behaviour summary

| Task pattern | `results.size()` vs `batch_size` |
|--------------|----------------------------------|
| Classification, depth, ViTPose (batched tensor) | `results.size() == batch_size` |
| Detection / instance segmentation | `results.size()` ≥ 1, variable; `batch_size` = image count |
| Gaussian splatting | Often `results.size() == 1` for multi-view |
| Depth / pose (tensor-led) | If postprocessor emits a different count, helper updates `output.batch_size` to match |

Throws `std::invalid_argument` when `batch_size <= 0`, exceeds `max_batch_size_`,
or result count is inconsistent for tasks that require a strict match.

---

## Export and engine alignment

When adding batch to ONNX exports:

| Domain | Export note |
|--------|-------------|
| Classification | Dynamic batch on input/output; `[N,C]` logits — [TorchVisionClassification.md](../export/classification/TorchVisionClassification.md) |
| YOLO v8/v11/v12 | `[N, 4+C, anchors]` — [ObjectDetection.md](../export/detection/ObjectDetection.md) |
| Depth / ViTPose | Postprocessors already split `shape[0]`; ensure export keeps leading `N` |

See [export/README.md](../export/README.md) for the export tree and batch-axis pointers.

---

## Verification in this repo

```bash
ctest --test-dir build --output-on-failure -R 'Batch|classification_batch|object_detection_batch|depth_estimation_batch|pose_estimation_batch'
```

Key tests: `test_batch_integration`, `test_batch_preprocess`, `test_batch_postprocess`,
`test_classification_batch`, `test_object_detection_batch`.
