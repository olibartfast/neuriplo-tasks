# vision-core

Framework-agnostic computer vision algorithms: object detection, instance segmentation, classification, and optical flow

## Features

- **Object Detection**: YOLO (v5-v12), RT-DETR, RT-DETRv2, YOLO-NAS, D-FINE, RF-DETR
  - ✅ Auto-detects YOLO format (v5-v7 vs v8+)
  - ✅ NMS and confidence filtering
  - ✅ Letterbox coordinate transformation
  - ✅ Support for anchor-free and transformer-based detectors
  
- **Instance Segmentation**: YOLOv5-seg, YOLOv8-seg, YOLO11-seg
  - ✅ Mask prototype decoding
  - ✅ Mask resizing and thresholding
  - ✅ Integrated with detection pipeline
  
- **Classification**: Generic top-k classifier
  - ✅ Softmax activation
  - ✅ Top-k predictions
  - ✅ Works with any classification model
  
- **Optical Flow**: RAFT (planned)

## Design Philosophy

Pure algorithm implementations with:
- ✅ No inference engine dependencies
- ✅ Framework-agnostic interfaces
- ✅ Reusable across different projects
- ✅ Only depends on OpenCV
- ✅ Modern C++17 with clear conventions

## Modern C++ Conventions

This library follows strict modern C++ naming conventions:

- **Types (classes, structs, enums)**: `PascalCase`
- **Functions and methods**: `snake_case`
- **Variables and parameters**: `snake_case`
- **Member variables**: `snake_case` (public) or `snake_case_` (private)
- **Constants**: `kConstantName` (static const) or inline variables
- **Attributes**: `[[nodiscard]]`, `noexcept` where appropriate

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
#include <vision-core/classification/classifier_postprocessor.hpp>
#include <vision-core/instance_segmentation/yolo_segmentation_postprocessor.hpp>
#include <vision-core/core/detection.hpp>

using namespace vision_core;

// Object Detection Example
std::vector<Detection> detections = YoloPostprocessor::postprocess(
    output_data, shape, frame_size, 640, 640, 0.25f, 0.45f
);

for (const auto& det : detections) {
    std::cout << "Class: " << det.class_id 
              << " Confidence: " << det.confidence
              << " BBox: " << det.bbox << std::endl;
}

// Classification Example
std::vector<ClassificationResult> predictions = ClassifierPostprocessor::postprocess(
    output_data, shape, /*top_k=*/5, /*apply_softmax=*/true
);

for (const auto& pred : predictions) {
    std::cout << "Class: " << pred.class_id 
              << " Confidence: " << pred.confidence << std::endl;
}

// Instance Segmentation Example
std::vector<InstanceSegmentation> segments = YoloSegmentationPostprocessor::postprocess(
    detection_output, mask_output, 
    detection_shape, mask_shape,
    frame_size, 640, 640, 0.25f, 0.45f, 0.5f
);

for (const auto& seg : segments) {
    std::cout << "Class: " << seg.class_id 
              << " Mask size: " << seg.mask_width << "x" << seg.mask_height << std::endl;
}
```

## Projects Using vision-core

- [tritonic](https://github.com/olibartfast/tritonic) - Triton Inference Server Client
- [object-detection-inference](https://github.com/olibartfast/object-detection-inference) - Multi-backend object detection

## Building

```bash
mkdir build && cd build
cmake -DBUILD_OBJECT_DETECTION=ON -DBUILD_TESTS=ON ..
cmake --build .
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_OBJECT_DETECTION` | ON | Build object detection components |
| `BUILD_INSTANCE_SEGMENTATION` | OFF | Build instance segmentation |
| `BUILD_CLASSIFICATION` | OFF | Build classification components |
| `BUILD_OPTICAL_FLOW` | OFF | Build optical flow algorithms |
| `BUILD_TESTS` | OFF | Build unit tests |

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

## API Design Principles

1. **Header-only where possible** for templates and small utilities
2. **Static factory methods** for postprocessors (stateless operations)
3. **Clear ownership** using smart pointers when needed
4. **Exception safety** with strong exception guarantee
5. **Const correctness** throughout the API

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
- [x] Bounding box utilities with letterbox support
- [x] YOLO postprocessor (v5-v12, auto-format detection)
- [x] RT-DETR postprocessor
- [x] YOLO-NAS postprocessor
- [x] D-FINE postprocessor
- [x] RF-DETR postprocessor
- [x] Classification postprocessor (top-k, softmax)
- [x] Instance segmentation (YOLO-seg with mask decoding)
- [x] Modern C++ conventions and documentation

### In Progress 🚧
- [ ] Comprehensive unit tests
- [ ] CI/CD pipeline setup
- [ ] Migration guides for tritonic and object-detection-inference

### Planned 📋
- [ ] Additional detector variants (YOLO-NAS, RF-DETR, D-FINE)
- [ ] RAFT optical flow implementation
- [ ] Batch processing utilities
- [ ] Python bindings (pybind11)
- [ ] Performance benchmarks and optimizations
