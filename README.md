# vision-core

Framework-agnostic computer vision algorithms (pre and post processing steps) to be used in other inference engine projects like [tritonic](https://github.com/olibartfast/tritonic) and [object-detection-inference](https://github.com/olibartfast/object-detection-inference)

## Features

- **Object Detection**: YOLO (v5-v12), YOLOv10, RT-DETR family (v1/v2), YOLO-NAS, D-FINE, RF-DETR
- **Instance Segmentation**: YOLOv5-seg, YOLOv8-seg, YOLO11-seg, RF-DETR-Seg
- **Classification**: Torchvision (ResNet, EfficientNet, etc.), TensorFlow/Keras Models, Vision Transformers (ViT)
- **Video Classification**: TimeSformer
- **Optical Flow**: RAFT


Example:
```cpp
struct Detection {
    cv::Rect bbox;
    float confidence;  // not 'score'
    int class_id;      // not 'label'
};

class YoloPostprocessor {
    [[nodiscard]] static std::vector<Detection> postprocess(...);
};
```

## Usage

### As CMake Submodule

```cmake
add_subdirectory(vision-core)
target_link_libraries(your_target vision-core::vision-core)
```

### Component-Based Building

```cmake
# Build only object detection
set(BUILD_OBJECT_DETECTION ON CACHE BOOL "")
set(BUILD_CLASSIFICATION OFF CACHE BOOL "")
add_subdirectory(vision-core)
```

### Example

```cpp
#include <vision-core/object_detection/yolo_postprocessor.hpp>
#include <vision-core/object_detection/detection_preprocessor.hpp>
#include <vision-core/classification/classification_preprocessor.hpp>
#include <vision-core/core/detection.hpp>

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

for (const auto& det : detections) {
    std::cout << "Class: " << det.class_id 
              << " Confidence: " << det.confidence
              << " BBox: " << det.bbox << std::endl;
}

// Classification with Preprocessing Example
ViTPreprocessor vit_prep(cv::Size(224, 224));
auto preprocessed_cls = vit_prep.preprocess(image);
// ... run inference ...

std::vector<ClassificationResult> predictions = ClassifierPostprocessor::postprocess(
    output_data, shape, /*top_k=*/5, /*apply_softmax=*/true
);

// Optical Flow Example
RaftPreprocessor raft_prep(cv::Size(960, 520));
cv::Mat frame1 = cv::imread("frame1.jpg");
cv::Mat frame2 = cv::imread("frame2.jpg");

auto preprocessed_frames = raft_prep.preprocess_pair(frame1, frame2);
// ... run inference ...

OpticalFlowResult flow = RaftPostprocessor::postprocess(
    output_data, shape, frame1.size()
);
cv::imshow("Flow", flow.flow_visualization);
```

## Projects Using vision-core

- [tritonic](https://github.com/olibartfast/tritonic) - Triton Inference Server Client
- [object-detection-inference](https://github.com/olibartfast/object-detection-inference) - Multi-backend object detection

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
│   ├── core/                   # Core data structures
│   │   ├── detection.hpp
│   │   └── bbox_processor.hpp
│   ├── object_detection/       # Object detection algorithms
│   │   ├── yolo_postprocessor.hpp
│   │   └── rtdetr_postprocessor.hpp
│   └── ...
├── src/                        # Implementation files
│   ├── core/
│   ├── object_detection/
│   └── ...
├── tests/                      # Unit tests
├── CMakeLists.txt
└── README.md
```

## Contributing

When contributing, please follow:
- Modern C++ naming conventions (see [NAMING_CONVENTIONS.md](docs/NAMING_CONVENTIONS.md))
- Include comprehensive documentation
- Add unit tests for new features
- Use `clang-format` with the project style

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed development guidelines.

## License

MIT License

## Roadmap

### Completed ✅
- [x] Core data structures (Detection, ClassificationResult)
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

### In Progress 🚧
- [ ] Comprehensive unit tests
- [ ] CI/CD pipeline setup
- [ ] Migration in tritonic and object-detection-inference

### Planned 📋
- [ ] Batch processing utilities
- [ ] Python bindings (pybind11)
- [ ] Performance benchmarks and optimizations
- [ ] Additional video classification models
