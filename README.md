# neuriplo-tasks

> 🚧 Status: Under Development — expect frequent updates.

A set of framework-agnostic computer vision algorithms including common pre-processing and post-processing steps designed to be reused across multiple inference engine projects such as:
* [tritonic](https://github.com/olibartfast/tritonic)
* [neuriplo-infer](https://github.com/olibartfast/neuriplo-infer)
* [neuriplo-track](https://github.com/olibartfast/neuriplo-track)

## Features

- **Object Detection**: YOLO (v4-v12, yolo26), RT-DETR family (RT-DETR v1/v2/v4, D-FINE, DEIM v1/v2), YOLO-NAS, RF-DETR, EdgeCrafter
- **Instance Segmentation**: YOLOv5/v8/v11-seg, YOLOv10-seg, YOLO26-seg, RF-DETR-Seg, EdgeCrafter
- **Classification**: Torchvision (ResNet, EfficientNet, etc.), TensorFlow/Keras Models, Vision Transformers (ViT)
- **Video Classification**: VideoMAE, ViViT, TimeSformer
- **Optical Flow**: RAFT
- **Pose Estimation**: YOLO pose (v5/v8/v11/v26), ViTPose, EdgeCrafter, RF-DETR keypoint pose
- **Depth Estimation**: Depth Anything V2
- **Open-Vocabulary Detection**: OWLv2 / OWL-ViT style text-conditioned detection; Grounding DINO
- **Gaussian Splatting**: LGM, LGM-mini, GRM (feed-forward image → 3D Gaussians)
- **Image Understanding (VLM)**: Gemma 4 and compatible vision-language models via llama.cpp (image captioning, visual Q&A)
- **Unified Task Interface**: Factory pattern for creating task instances with integrated preprocessing and postprocessing
- **Composite Task Pipelines**: Ordered `Result` stages for multi-task flows such as detection → pose or detection → segmentation
- **Unified Tensor Interface**: Simplified API using `Tensor` struct that encapsulates data and shape information


## Two Ways to Use neuriplo-tasks

### 1. Direct Preprocessor/Postprocessor Usage (Flexible)

Use individual preprocessors and postprocessors for maximum flexibility:

```cpp
#include <neuriplo/tasks/object_detection/yolo_postprocessor.hpp>
#include <neuriplo/tasks/object_detection/detection_preprocessor.hpp>
#include <neuriplo/tasks/object_detection/object_detection_task.hpp>
#include <neuriplo/tasks/core/vision/stb_io.hpp>

using namespace neuriplo_tasks;

// Object Detection with Preprocessing Example
DetectionPreprocessor yolo_prep(vision::Size(640, 640));
vision::Image image = vision::loadImage("image.jpg");

// Preprocess
auto preprocessed = yolo_prep.preprocess({image});
// ... run inference with your engine ...

// Postprocess
// Create postprocessor instance (reusable)
YoloPostprocessor postprocessor(
    ObjectDetectionTask::ModelType::YOLO_STANDARD,
    0.25f, // confidence threshold
    0.45f  // NMS threshold
);

// Convert inference outputs to Tensor format
std::vector<Tensor> tensors;
for (size_t i = 0; i < inference_outputs.size(); ++i) {
    tensors.emplace_back(inference_outputs[i], output_shapes[i]);
}

std::vector<Detection> detections = postprocessor.postprocess(
    tensors,       // std::vector<Tensor>
    image.size()
);
```

### 2. TaskInterface/TaskFactory Usage (Unified)

Use the unified task interface for integrated preprocessing and postprocessing:

```cpp
#include <neuriplo/tasks/core/task_factory.hpp>
#include <neuriplo/tasks/core/task_interface.hpp>
#include <neuriplo/tasks/core/model_info.hpp>
#include <neuriplo/tasks/core/vision/stb_io.hpp>

using namespace neuriplo_tasks;

// Setup model info
ModelInfo model_info;
model_info.input_shapes = {{1, 3, 640, 640}};
model_info.input_formats = {"FORMAT_NCHW"};
model_info.input_names = {"images"};
model_info.output_names = {"output0"};

// Create task instance via factory
auto task = TaskFactory::createTaskInstance("yolov8", model_info);

// Preprocess
std::vector<vision::Image> images = {vision::loadImage("image.jpg")};
auto preprocessed = task->preprocess(images);

// ... run inference ...

// Convert inference outputs to Tensor format
std::vector<Tensor> tensors;
for (size_t i = 0; i < inference_outputs.size(); ++i) {
    tensors.emplace_back(inference_outputs[i], output_shapes[i]);
}

// Postprocess
auto results = task->postprocess(
    images[0].size(),
    tensors  // std::vector<Tensor>
);

// Access results via std::variant
for (const auto& result : results) {
    if (std::holds_alternative<Detection>(result)) {
        const auto& det = std::get<Detection>(result);
        std::cout << "Class: " << det.class_id
                  << " Confidence: " << det.class_confidence << std::endl;
    }
}
```

### 3. Batch Processing Utilities

Helpers for running **N independent images** through the same task without reimplementing
batch metadata or per-domain split logic.

- **Consumer guide:** [docs/batch_processing.md](docs/batch_processing.md) — worked examples, engine vs library responsibilities, migration from `N=1`
- **Per-family readiness:** [docs/batch_support_matrix.md](docs/batch_support_matrix.md)

**Headers:** `batch_types.hpp`, `batch_preprocess.hpp`, `batch_postprocess.hpp`

```cpp
#include <neuriplo/tasks/core/batch_preprocess.hpp>
#include <neuriplo/tasks/core/batch_postprocess.hpp>
#include <neuriplo/tasks/core/task_factory.hpp>

using namespace neuriplo_tasks;

ModelInfo model_info;
model_info.input_shapes = {{2, 3, 224, 224}};
model_info.input_formats = {"FORMAT_NCHW"};
model_info.max_batch_size_ = 2;
model_info.batch_size_ = 2;

auto task = TaskFactory::createTaskInstance("resnet50", model_info);

BatchRequest request;
request.images = {vision::loadImage("a.jpg"), vision::loadImage("b.jpg")};

// 1. Preprocess — one buffer per image; batch_size = N
BatchPreprocessOutput pre = batchPreprocess(*task, request);

// 2. Run inference (consumer responsibility): stack pre.buffers into [N,C,H,W]
//    or feed separate inputs per your engine contract.

// 3. Postprocess — align vector<Result> with batch indices
std::vector<Tensor> output_tensors = { /* engine output, e.g. shape [N, num_classes] */ };
BatchPostprocessOutput post =
    batchPostprocess(*task, request.images[0].size(), output_tensors, pre.batch_size);

// Classification with N>1: one top-1 Result per image (post.results.size() == N).
// Detection: variable detections per image; post.batch_size stays N.
```

Set `ModelInfo.max_batch_size_` so `batchPreprocess` / `batchPostprocess` reject oversized
batches. `batch_size_` is a consumer hint for the inference request.

### 4. Task Pipeline Utilities

`TaskPipeline` composes explicit `Result` stages without hiding inference engine boundaries.
Use it for workflows such as detection → pose or detection → segmentation after each model
has produced its own `Result` vector.

```cpp
#include <neuriplo/tasks/core/task_pipeline.hpp>

using namespace neuriplo_tasks;

SequentialTaskPipeline pipeline;
pipeline.addStage([](const std::vector<Result>& detections) {
    std::vector<Result> pose_inputs;
    // Convert/filter detection results for the next model boundary.
    return pose_inputs;
});

auto next_results = pipeline.run(detection_results);
```

## Usage

### As CMake Submodule

```cmake
add_subdirectory(neuriplo-tasks)
target_link_libraries(your_target neuriplo-tasks::neuriplo-tasks)
```

### As Installed Package

```bash
# Install
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
cmake --build .
sudo cmake --install .
```

```cmake
# In your CMakeLists.txt
find_package(neuriplo-tasks REQUIRED)
target_link_libraries(your_target neuriplo-tasks::neuriplo-tasks)
```

### Supported Model Types (TaskFactory)

`TaskFactory` routes model type strings through a **built-in, compile-time**
registration table in `src/core/task_factory.cpp`. New built-in tasks require
editing that table and the README list below. **Third-party or runtime task
plugins are not supported**; if plugin extension becomes a product goal, add a
separate explicit extension registry rather than growing the internal table
indefinitely.

<!-- TASKFACTORY_MODEL_LIST:START -->
The TaskFactory supports the following model type strings. Matching normalizes strings by lowercasing and stripping whitespace, hyphens, and underscores, so `YOLO-V8`, `yolo_v8`, and ` yolo v8 ` route identically. Specific segmentation and pose aliases are checked before generic detection aliases.

**Object Detection:**

- `"yolo"`, `"yolov7e2e"`, `"yolov10"`, `"yolo26"`, `"yolov4"` - YOLO-based variants
- `"yolonas"` - YOLO-NAS
- `"rtdetr"` - RT-DETR family (RT-DETR v1, v2, and v4; excludes v3; includes D-FINE and DEIM v1/v2)
- `"rtdetrul"`, `"rtdetrultralytics"` - RT-DETR (Ultralytics implementation)
- `"rfdetr"` - RF-DETR
- `"ecdet"` - EdgeCrafter detection (any string starting with `ecdet`)
- `"edgecrafter"`, `"edgecrafter-det"` - EdgeCrafter detection unless the normalized string contains `seg` or `pose`

**Instance Segmentation:**
- `"ecseg"` - EdgeCrafter segmentation (any string starting with `ecseg` or `edgecrafter` and containing `seg`)
- `"yoloseg"`, `"yolo-seg"`, `"yolov8-seg"` - YOLOv5/YOLOv8/YOLO11-style segmentation
- `"yolov10seg"`- YOLOv10
- `"yolo26seg"` - YOLO26
- `"rfdetrseg"` - RF-DETR

**Classification:**
- `"torchvision-classifier"` - Torchvision models (ResNet, EfficientNet, etc.)
- `"tensorflow-classifier"` - TensorFlow/Keras models
- `"vit-classifier"` - Vision Transformers

Any model type starting with `resnet` (e.g. `resnet50`) or containing `tensorflow` also routes to classification.

**Video Classification:**
- `"videomae"` - VideoMAE
- `"vivit"` - ViViT
- `"timesformer"` - TimeSformer

**Optical Flow:**
- `"raft"` - RAFT optical flow

**Pose Estimation:**
- `"yolov8pose"`, `"yolov8-pose"` - YOLOv8 pose (single-stage, returns bbox + keypoints)
- `"yolo11pose"`, `"yolo11-pose"` - YOLO11 pose
- `"yolo26pose"`, `"yolo26-pose"` - YOLO26 pose
- `"yolov5pose"`, `"yolov5-pose"` - YOLOv5 pose
- `"rfdetrpose"`, `"rfdetr-pose"`, `"rfdetrkeypoint"`, `"rfdetr-keypoint"`, `"rfdetrkpt"`, `"rfdetr-kpt"` - RF-DETR keypoint pose (single-stage, returns bbox + 17 coco keypoints with visibility and per-keypoint covariance)
- `"vitpose"` - ViTPose (top-down, heatmap-based)
- `"ecpose"` - EdgeCrafter pose estimation (any string starting with `ecpose`, or `edgecrafter` and containing `pose`)

RF-DETR keypoint models output per-keypoint visibility and 2×2 pixel covariance (decoded from Cholesky L via the ONNX `log_l11`, `l21`, `log_l22` channels). Keypoints are filtered by an uncertainty-weighted score fusion that discounts high-covariance predictions.

**Depth Estimation:**
- `"depth_anything_v2"`, `"depth-anything-v2"` - Depth Anything V2

**Open-Vocabulary Detection:**
- `"owlv2"` - OWLv2 open-vocabulary detection
- `"owlvit"` - OWL-ViT compatible open-vocabulary detection
- `"groundingdino"` - Grounding DINO text-conditioned detection

Open-vocabulary models use text prompts supplied at runtime through `TaskConfig::text_prompts`. Tokenizer assets can be passed either as file paths (`tokenizer_vocab_path`, `tokenizer_merges_path`) or preloaded text blobs (`tokenizer_vocab_json`, `tokenizer_merges_text`).

The expected ONNX contract is:
- Inputs: `pixel_values`, `input_ids`, `attention_mask`
- Outputs: `logits`, `pred_boxes`, and optional `objectness_logits`

Results are returned as `OpenVocabDetection` entries containing `bbox`, `score`, `prompt_index`, and resolved `label`.

For export details, see [export/open_vocab_detection/OWLv2.md](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/open_vocab_detection/OWLv2.md).

**Image Understanding (VLM):**
- `"gemma4"`, `"gemma"`, `"llama"`, `"llamacpp"`, `"imageunderstanding"` - Vision-language model image captioning / Q&A via llama.cpp backend

Input contract: `preprocess()` returns two tensors — `[0]` UTF-8 prompt bytes, `[1]` raw RGB pixels with an 8-byte header `[uint32 width LE][uint32 height LE][H×W×3 bytes]`. When no image is provided only tensor `[0]` is returned (text-only mode). Output is a UTF-8 string returned as float-encoded bytes (one `float` per byte value).

Requires the llama.cpp `LLAMACPP` backend with an mmproj (vision projector) GGUF.

For model download and setup details, see [export/image_understanding/ImageUnderstanding.md](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/image_understanding/ImageUnderstanding.md).

**Gaussian Splatting:**
- `"lgm"`, `"lgm-mini"` - LGM (Large Gaussian Model)
- `"grm"` - GRM
- `"gaussiansplatting"`, any string containing `"splat"` - generic alias

<!-- TASKFACTORY_MODEL_LIST:END -->

## Building

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --parallel
```

The core `neuriplo-tasks` target has no image-library dependency. Link `neuriplo-tasks::vision-stb` for file loading and saving. Enable and link `neuriplo-tasks::vision-opencv` only at an OpenCV consumer boundary. Public task contracts use `vision::Image`, `vision::Size`, and `vision::PixelType`.

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `-DBUILD_TESTS=ON/OFF` | `OFF` | Build unit tests |
| `-DWERROR=ON/OFF` | `OFF` | Treat compiler warnings as errors (used in CI) |
| `-DSANITIZERS=ON/OFF` | `OFF` | Enable AddressSanitizer + UndefinedBehaviorSanitizer |
| `-DNEURIPLO_TASKS_WITH_STB=ON/OFF` | `ON` | Build the optional `neuriplo-tasks::vision-stb` image I/O target |
| `-DNEURIPLO_TASKS_WITH_OPENCV=ON/OFF` | `OFF` | Build the optional `neuriplo-tasks::vision-opencv` interop target |

### Format Code (Optional)

Install clang-format if needed:
```bash
sudo apt-get install -y clang-format-18
```

Check for formatting issues without modifying files:
```bash
find src include tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 --dry-run --Werror
```

Apply formatting in place:
```bash
find src include tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format-18 -i
```

Configuration: `.clang-format` at the project root.

### Static Analysis — clang-tidy (Optional)

Install clang-tidy if needed:
```bash
sudo apt-get install -y clang-tidy-18
```

`compile_commands.json` is generated automatically in the build directory. Run clang-tidy on all project sources:
```bash
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
find src -name '*.cpp' | xargs clang-tidy-18 -p build
```

Configuration: `.clang-tidy` at the project root.

> **Note:** Never blindly apply `--fix` — some checks (e.g. `modernize-pass-by-value`) change function signatures and break link-time compatibility with already-compiled objects. Always do a clean build after applying fixes.

### Static Analysis — cppcheck (Optional)

Install cppcheck if needed:
```bash
sudo apt-get install -y cppcheck
```

Run flow-based static analysis:
```bash
cppcheck --enable=warning --std=c++17 \
  --suppress=missingIncludeSystem \
  --suppress=unmatchedSuppression \
  --error-exitcode=1 \
  -I include src/
```

### Sanitizers (Optional)

Build with AddressSanitizer and UndefinedBehaviorSanitizer enabled. Use a separate build directory to keep sanitizer and release builds independent:

```bash
cmake -S . -B build-san \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTS=ON \
  -DSANITIZERS=ON

cmake --build build-san --parallel

# Run tests under sanitizers:
ctest --test-dir build-san --output-on-failure
```

Sanitizers catch memory errors, use-after-free, undefined behaviour, and integer overflow at runtime with minimal code changes.

### Valgrind (Optional)

Build tests in Debug mode and run every test binary under Valgrind:

```bash
sudo apt-get install -y valgrind

cmake -S . -B build-valgrind -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
cmake --build build-valgrind --parallel

for test_bin in build-valgrind/tests/test_*; do
  [ -x "$test_bin" ] || continue
  valgrind \
    --error-exitcode=1 \
    --leak-check=full \
    --show-leak-kinds=definite,indirect \
    --errors-for-leak-kinds=definite,indirect \
    --track-origins=yes \
    --num-callers=25 \
    "$test_bin"
done
```

CI runs this as a separate code-quality job.

### Pre-commit (Optional)

[pre-commit](https://pre-commit.com/) runs `clang-format` and `cppcheck` automatically on every commit:

```bash
pip install pre-commit
pre-commit install          # install the git hook
pre-commit run --all-files  # run manually on all files
```

### Strict Compilation (Optional)

To treat all compiler warnings as errors (as CI does), pass `-DWERROR=ON`:

```bash
cmake -S . -B build -DWERROR=ON
cmake --build build
```

## Code Quality Tools

| Tool | Purpose | How to run |
|------|---------|------------|
| `clang-format-18` | Code formatting | `find src include tests -name '*.cpp' -o -name '*.hpp' \| xargs clang-format-18 -i` |
| `clang-tidy-18` | Static analysis (AST-based) | `find src -name '*.cpp' \| xargs clang-tidy-18 -p build` |
| `cppcheck` | Static analysis (flow-based) | `cppcheck --enable=warning --std=c++17 -I include src/` |
| ASan + UBSan | Runtime memory/UB detection | `-DSANITIZERS=ON` at configure time |
| Valgrind | Runtime leak/error detection | `build-valgrind` + `valgrind --error-exitcode=1 ...` |
| pre-commit | Automates format + cppcheck on commit | `pre-commit install` |

## License

MIT License
