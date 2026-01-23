# vision-core

> 🚧 Status: Under Development — expect frequent updates.  

A set of framework-agnostic computer vision algorithms including common pre-processing and post-processing steps designed to be reused across multiple inference engine projects such as:
* [tritonic](https://github.com/olibartfast/tritonic)
* [object-detection-inference](https://github.com/olibartfast/object-detection-inference)
* [vision-tracking](https://github.com/olibartfast/vision-tracking)

## Features

- **Object Detection**: YOLO (v5-v12, yolo26), RT-DETR family (RT-DETR v1/v2/v4, D-FINE, DEIM v1/v2), YOLO-NAS
- **Instance Segmentation**: YOLOv5/v8/v11-seg, RF-DETR-Seg
- **Classification**: Torchvision (ResNet, EfficientNet, etc.), TensorFlow/Keras Models, Vision Transformers (ViT)
- **Video Classification**: VideoMAE, ViViT, TimeSformer
- **Optical Flow**: RAFT
- **Pose Estimation**: ViTPose
- **Unified Task Interface**: Factory pattern for creating task instances with integrated preprocessing and postprocessing
- **Unified Tensor Interface**: Simplified API using `Tensor` struct that encapsulates data and shape information

## Tensor Interface

Vision-core uses a unified `Tensor` struct to simplify API usage and improve type safety:

```cpp
struct Tensor {
    std::vector<TensorElement> data;    // Tensor values (variant: float, int32_t, int64_t, uint8_t)
    std::vector<int64_t> shape;         // Tensor dimensions
    
    // Constructors
    Tensor() = default;
    Tensor(std::vector<TensorElement> data_, std::vector<int64_t> shape_);
};

// Example usage:
Tensor output_tensor(inference_data, {1, 25200, 85});  // YOLO output
Tensor scores_tensor(scores_data, {1, 8400});          // Confidence scores
```

**Benefits:**
- **Type Safety**: Data and shape are always paired together, preventing mismatched parameters
- **Cleaner API**: Reduces parameter count in postprocessing functions
- **Better Encapsulation**: Related tensor information is grouped in a single structure
- **Consistency**: All postprocessors use the same tensor interface

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
- [x] Additional video classification models (VideoMAE, ViViT)
- [x] Pose estimation support (ViTPose)
