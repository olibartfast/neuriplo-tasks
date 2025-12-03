# vision-core (TODO / In Progress)
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

using namespace vision_core;

// Object Detection with Preprocessing Example
YoloPreprocessor yolo_prep(cv::Size(640, 640));
cv::Mat image = cv::imread("image.jpg");

// Preprocess
auto preprocessed = yolo_prep.preprocess(image);
// ... run inference with your engine ...

// Postprocess
std::vector<Detection> detections = YoloPostprocessor::postprocess(
    output_data, shape, image.size(), 640, 640, 0.25f, 0.45f
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
- `"yolo"`, `"yolov5"` to `"yolov12"`, `"yolo11"` - YOLO family (v5-v12)
- `"yolonas"`, `"yolo-nas"` - YOLO-NAS
- `"rtdetr"`, `"rtdetrv2"`, `"rtdetr-ultralytics"` - RT-DETR family
- `"rfdetr"`, `"rf-detr"`, `"rfdetr-s/m/l"` - RF-DETR variants
- `"dfine"`, `"d-fine"`, `"dfine-s/m/l/x"` - D-FINE variants

**Instance Segmentation:**
- `"yolo-seg"`, `"yolov5-seg"`, `"yolov8-seg"`, `"yolov11-seg"` - YOLO segmentation
- `"rfdetr-seg"`, `"rf-detr-seg"` - RF-DETR segmentation

**Classification:**
- `"torchvision-classifier"`, `"resnet"`, `"resnet50"` - Torchvision models (ImageNet norm)
- `"tensorflow-classifier"` - TensorFlow/Keras models
- `"vit-classifier"` - Vision Transformers

**Optical Flow:**
- `"raft"`, `"optical-flow"`, `"raft-small/large"` - RAFT optical flow

**Video Classification:**
- `"timesformer"`, `"video-classification"`, `"action-recognition"` - TimeSformer


## Building

```bash
mkdir build && cd build
cmake -DBUILD_TESTS=ON ..
cmake --build .
```


## Directory Structure

```
vision-core/
├── include/vision-core/        # Public headers
│   ├── core/                   # Core data structures and task interface
│   │   ├── result_types.hpp    # Result structures (Detection, Classification, etc.)
│   │   ├── model_info.hpp      # Model metadata structure
│   │   ├── task_interface.hpp  # Base task interface
│   │   ├── task_factory.hpp    # Task factory pattern
│   │   ├── bbox_processor.hpp  # Bounding box utilities
│   │   └── preprocessor.hpp    # Base preprocessor
│   ├── object_detection/       # Object detection algorithms
│   │   ├── yolo_postprocessor.hpp
│   │   ├── rtdetr_postprocessor.hpp
│   │   ├── detection_preprocessor.hpp
│   │   └── ...
│   ├── instance_segmentation/  # Instance segmentation
│   ├── classification/         # Classification
│   ├── video_classification/   # Video classification
│   └── optical_flow/          # Optical flow
├── src/                        # Implementation files
│   ├── core/
│   │   ├── task_interface.cpp
│   │   ├── task_factory.cpp
│   │   └── ...
│   ├── object_detection/
│   └── ...
├── docs/                       # Documentation
│   └── NAMING_CONVENTIONS.md
├── tests/                      # Unit tests
├── CMakeLists.txt
└── README.md
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
