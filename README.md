# vision-core

> 🚧 Status: Under Development — expect frequent updates.

A set of framework-agnostic computer vision algorithms including common pre-processing and post-processing steps designed to be reused across multiple inference engine projects such as:
* [tritonic](https://github.com/olibartfast/tritonic)
* [object-detection-inference](https://github.com/olibartfast/object-detection-inference)
* [deep-stream-infer-lab](https://github.com/olibartfast/deepstream-infer-lab)

## Features

- **Object Detection**: YOLO (v5-v12), YOLOv10, RT-DETR family (v1/v2), YOLO-NAS, D-FINE, RF-DETR
- **Instance Segmentation**: YOLOv5-seg, YOLOv8-seg, YOLO11-seg, RF-DETR-Seg
- **Classification**: Torchvision (ResNet, EfficientNet, etc.), TensorFlow/Keras Models, Vision Transformers (ViT)
- **Video Classification**: TimeSformer
- **Optical Flow**: RAFT
- **Unified Task Interface**: Factory pattern for creating task instances with integrated preprocessing and postprocessing
- **Thread-Safe TaskFactory**: Custom task registration with thread-safe registry access

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

std::vector<Detection> detections = postprocessor.postprocess(
    inference_outputs, // std::vector<std::vector<TensorElement>>
    output_shapes,     // std::vector<std::vector<int64_t>>
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

// Postprocess
auto results = task->postprocess(
    images[0].size(), 
    inference_outputs, 
    output_shapes
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
- `"yolov5 to v12"` - YOLO family
- `"yolonas"`, `"yolo-nas"` - YOLO-NAS
- `"rtdetr"`, `"rtdetrv2"` - RT-DETR family
- `"rtdetrul"`, `"rtdetr-ultralytics"` - RT-DETR Ultralytics variant
- `"rfdetr"`, `"rf-detr"` - RF-DETR
- `"dfine"`, `"deim"` - D-FINE and variants

**Instance Segmentation:**
- `"yoloseg"`, `"yolov8-seg"` - YOLO segmentation
- `"rfdetrseg"`, `"rf-detr-seg"` - RF-DETR segmentation

**Classification:**
- `"torchvision-classifier"` - Torchvision models (ResNet, EfficientNet, etc.)
- `"tensorflow-classifier"` - TensorFlow/Keras models
- `"vit-classifier"` - Vision Transformers
- `"resnet50"`, `"resnet"` - ResNet variants

**Optical Flow:**
- `"raft"` - RAFT optical flow

## Building

```bash
mkdir build && cd build
cmake -DBUILD_TESTS=ON ..
cmake --build .
```

## License

MIT License

## Roadmap
### In Progress 🚧
- [ ] Extend Unit Tests
- [ ] CI/CD pipeline setup
- [ ] Migration of tritonic/object-detection-inference/deep-stream-infer-lab to use vision-core

### Planned 📋
- [ ] Batch processing utilities
- [ ] Performance benchmarks and optimizations
- [ ] Additional video classification models
- [ ] Pose estimation support
