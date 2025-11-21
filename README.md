# vision-core

Framework-agnostic computer vision algorithms: object detection, instance segmentation, classification, and optical flow

## Features

- **Object Detection**: YOLO (v5-v12), RT-DETR, RT-DETRv2, YOLO-NAS, RF-DETR, D-FINE, DEIM
- **Instance Segmentation**: YOLOv5-seg, YOLOv8-seg, YOLO11-seg
- **Classification**: Generic classifier postprocessing
- **Optical Flow**: RAFT

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
#include <vision-core/core/detection.hpp>

using namespace vision_core;

// Process YOLO output
std::vector<Detection> detections = YoloPostprocessor::postprocess(
    output_data, shape, frame_size, 640, 640, 0.25f, 0.45f
);

// Access results with modern naming
for (const auto& det : detections) {
    std::cout << "Class: " << det.class_id 
              << " Confidence: " << det.confidence
              << " BBox: " << det.bbox << std::endl;
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

- [ ] YOLOv8 instance segmentation support
- [ ] RAFT optical flow implementation
- [ ] Generic classification postprocessor
- [ ] Batch processing utilities
- [ ] Python bindings (pybind11)
