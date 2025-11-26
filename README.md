# vision-core

Framework-agnostic computer vision algorithms (pre and post processing steps) planned to be used in other inference engine projects like [tritonic](https://github.com/olibartfast/tritonic), [object-detection-inference](https://github.com/olibartfast/object-detection-inference) and [deep-stream-infer-lab](https://github.com/olibartfast/deepstream-infer-lab)

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

**Video Classification:**
- `"timesformer"` - TimeSformer video classification

## Projects Using vision-core

- [tritonic](https://github.com/olibartfast/tritonic) - Triton Inference Server Client
- [object-detection-inference](https://github.com/olibartfast/object-detection-inference) - Multi-backend object detection (ONNX Runtime, TensorRT, OpenVINO)
- [deepstream-infer-lab](https://github.com/olibartfast/deepstream-infer-lab) - NVIDIA DeepStream inference experiments


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

### Completed ✅
- [x] Core data structures (Detection, Classification, InstanceSegmentation, etc.)
- [x] Unified Result variant type
- [x] TaskInterface base class
- [x] TaskFactory with model type registration
- [x] ModelInfo for framework-agnostic model metadata
- [x] Bounding box utilities with letterbox support (XYWH and XYXY)
- [x] YOLO postprocessor (v5-v12, auto-format detection)
- [x] YOLOv10 postprocessor (end-to-end, no NMS)
- [x] RT-DETR postprocessor
- [x] YOLO-NAS postprocessor
- [x] D-FINE postprocessor
- [x] RF-DETR postprocessor
- [x] Classification postprocessor (top-k, softmax)
- [x] Instance segmentation (YOLO-seg with mask decoding)
- [x] RF-DETR segmentation postprocessor
- [x] Video classification (TimeSformer)
- [x] Optical flow (RAFT)
- [x] Modern C++ conventions and documentation
- [x] Integration guide for tritonic
- [x] Concrete TaskInterface implementations (YOLO, RT-DETR, Classification)
- [x] Comprehensive unit tests for TaskFactory
- [x] Thread-safe TaskFactory with custom task registration
- [x] Instance segmentation task implementation (YOLO-Seg, RF-DETR-Seg)
- [x] Transformer detection task implementation (RT-DETR, D-FINE, RF-DETR)
- [x] Optical flow task implementation (RAFT)
- [x] Video classification task implementation (TimeSformer)

### In Progress 🚧
- [ ] CI/CD pipeline setup
- [ ] Migration of tritonic to use vision-core

### Planned 📋
- [ ] Batch processing utilities
- [ ] Performance benchmarks and optimizations
- [ ] Additional video classification models
- [ ] Pose estimation support
