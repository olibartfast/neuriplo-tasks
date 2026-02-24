# vision-core

> 🚧 Status: Under Development — expect frequent updates.

A set of framework-agnostic computer vision algorithms including common pre-processing and post-processing steps designed to be reused across multiple inference engine projects such as:
* [tritonic](https://github.com/olibartfast/tritonic)
* [vision-inference](https://github.com/olibartfast/vision-inference)
* [vision-tracking](https://github.com/olibartfast/vision-tracking)

## Features

- **Object Detection**: YOLO (v5-v12, yolo26), RT-DETR family (RT-DETR v1/v2/v4, D-FINE, DEIM v1/v2), YOLO-NAS
- **Instance Segmentation**: YOLOv5/v8/v11-seg, RF-DETR-Seg
- **Classification**: Torchvision (ResNet, EfficientNet, etc.), TensorFlow/Keras Models, Vision Transformers (ViT)
- **Video Classification**: VideoMAE, ViViT, TimeSformer
- **Optical Flow**: RAFT
- **Pose Estimation**: ViTPose
- **Depth Estimation**: Depth Anything V2
- **Unified Task Interface**: Factory pattern for creating task instances with integrated preprocessing and postprocessing
- **Unified Tensor Interface**: Simplified API using `Tensor` struct that encapsulates data and shape information


## Two Ways to Use vision-core

### 1. Direct Preprocessor/Postprocessor Usage (Flexible)

Use individual preprocessors and postprocessors for maximum flexibility:

```cpp
#include <vision-core/object_detection/yolo_postprocessor.hpp>
#include <vision-core/object_detection/detection_preprocessor.hpp>
#include <vision-core/object_detection/object_detection_task.hpp>

using namespace vision_core;

// Object Detection with Preprocessing Example
DetectionPreprocessor yolo_prep(cv::Size(640, 640));
cv::Mat image = cv::imread("image.jpg");

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
#include <vision-core/core/task_factory.hpp>
#include <vision-core/core/task_interface.hpp>
#include <vision-core/core/model_info.hpp>

using namespace vision_core;

// Setup model info
ModelInfo model_info;
model_info.input_shapes = {{1, 3, 640, 640}};
model_info.input_formats = {"FORMAT_NCHW"};
model_info.input_names = {"images"};
model_info.output_names = {"output0"};

// Create task instance via factory
auto task = TaskFactory::createTaskInstance("yolov8", model_info);

// Preprocess
std::vector<cv::Mat> images = {cv::imread("image.jpg")};
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

## Usage

### As CMake Submodule

```cmake
add_subdirectory(vision-core)
target_link_libraries(your_target vision-core::vision-core)
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
find_package(vision-core REQUIRED)
target_link_libraries(your_target vision-core::vision-core)
```

### Supported Model Types (TaskFactory)

The TaskFactory supports the following model type strings:

**Object Detection:**

- `"yolo"`, `"yolov7e2e"`, `"yolov10"`, `"yolo26"`, `"yolov4"` - YOLO-based variants
- `"yolonas"` - YOLO-NAS
- `"rtdetr"` - RT-DETR family (RT-DETR v1, v2, and v4; excludes v3; includes D-FINE and DEIM v1/v2)
- `"rtdetrul"` - RT-DETR (Ultralytics implementation)
- `"rfdetr"` - RF-DETR

**Instance Segmentation:**
- `"yoloseg"` - YOLOv5/YOLOv8/YOLO11
- `"yolov10seg"`- YOLOv10
- `"yolo26seg"` - YOLO26
- `"rfdetrseg"` - RF-DETR

**Classification:**
- `"torchvision-classifier"` - Torchvision models (ResNet, EfficientNet, etc.)
- `"tensorflow-classifier"` - TensorFlow/Keras models
- `"vit-classifier"` - Vision Transformers

**Video Classification:**
- `"videomae"` - VideoMAE
- `"vivit"` - ViViT
- `"timesformer"` - TimeSformer

**Optical Flow:**
- `"raft"` - RAFT optical flow

**Pose Estimation:**
- `"vitpose"` - ViTPose

**Depth Estimation:**
- `"depth_anything_v2"`, `"depth-anything-v2"` - Depth Anything V2

## Building

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --parallel
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `-DBUILD_TESTS=ON/OFF` | `OFF` | Build unit tests |
| `-DWERROR=ON/OFF` | `OFF` | Treat compiler warnings as errors (used in CI) |
| `-DSANITIZERS=ON/OFF` | `OFF` | Enable AddressSanitizer + UndefinedBehaviorSanitizer |

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
| pre-commit | Automates format + cppcheck on commit | `pre-commit install` |

## License

MIT License

## Roadmap
### In Progress 🚧
- [ ] Migration of tritonic/vision-inference/deep-stream-infer-lab to use vision-core

### Planned 📋
- [ ] Batch processing utilities
- [ ] Performance benchmarks and optimizations
