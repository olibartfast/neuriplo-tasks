# Vision-Core Development Guide

## 1. Project Overview and Architecture

The `vision-core` library serves as a framework-agnostic foundation for computer vision algorithms. It provides pure algorithm implementations without inference engine dependencies, making it reusable across different projects like `tritonic` and `object-detection-inference`.

### Key Design Principles

- **Framework Agnostic**: No dependencies on specific inference engines (ONNX Runtime, TensorRT, Triton, etc.)
- **Pure Algorithms**: Only postprocessing and utility functions
- **OpenCV Only**: Single dependency for maximum portability
- **Modern C++17**: Leveraging modern language features for safety and clarity
- **Component-Based**: Modular design allowing selective inclusion of features

## 2. Repository Structure

```
vision-core/
├── include/vision-core/          # Public API headers
│   ├── core/                     # Core data structures and utilities
│   │   ├── detection.hpp         # Detection and classification result structs
│   │   └── bbox_processor.hpp    # Bounding box utilities
│   ├── object_detection/         # Object detection postprocessors
│   │   ├── yolo_postprocessor.hpp
│   │   └── rtdetr_postprocessor.hpp
│   ├── instance_segmentation/    # Instance segmentation (planned)
│   ├── classification/           # Classification postprocessors (planned)
│   └── optical_flow/             # Optical flow algorithms (planned)
├── src/                          # Implementation files
│   ├── core/
│   │   └── bbox_processor.cpp
│   ├── object_detection/
│   │   ├── yolo_postprocessor.cpp
│   │   └── rtdetr_postprocessor.cpp
│   └── ...
├── tests/                        # Unit and integration tests
├── docs/                         # Documentation
│   ├── DEVELOPMENT.md            # This file
│   └── NAMING_CONVENTIONS.md     # Coding standards
├── cmake/                        # CMake configuration files
│   └── vision-core-config.cmake.in
├── CMakeLists.txt                # Main build configuration
└── README.md                     # User-facing documentation
```

## 3. Development Environment Setup

### Prerequisites

- **C++17 compatible compiler**: GCC 7+, Clang 5+, MSVC 2017+
- **CMake**: Version 3.15 or higher
- **OpenCV**: Version 4.x recommended
- **Git**: For version control

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/olibartfast/vision-core.git
cd vision-core

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake -DBUILD_OBJECT_DETECTION=ON -DBUILD_TESTS=ON ..

# Build the library
cmake --build .

# Run tests (if enabled)
ctest --output-on-failure
```

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Follow naming conventions** (see [NAMING_CONVENTIONS.md](NAMING_CONVENTIONS.md))

3. **Build incrementally**
   ```bash
   cmake --build . --target vision-core
   ```

4. **Run tests frequently**
   ```bash
   ctest -R my_test_name
   ```

5. **Format code** (if clang-format is configured)
   ```bash
   clang-format -i include/vision-core/**/*.hpp src/**/*.cpp
   ```

## 4. Implementation Guide

### Adding a New Postprocessor

#### Step 1: Create Header File

Create `include/vision-core/object_detection/new_detector_postprocessor.hpp`:

```cpp
#pragma once

#include "vision-core/core/detection.hpp"
#include <vector>

namespace vision_core {

class NewDetectorPostprocessor {
public:
    [[nodiscard]] static std::vector<Detection> postprocess(
        const float* output,
        const std::vector<int64_t>& shape,
        const cv::Size& frame_size,
        int network_width,
        int network_height,
        float confidence_threshold = 0.5f
    );
};

} // namespace vision_core
```

#### Step 2: Implement in Source File

Create `src/object_detection/new_detector_postprocessor.cpp`:

```cpp
#include "vision-core/object_detection/new_detector_postprocessor.hpp"
#include "vision-core/core/bbox_processor.hpp"
#include <opencv2/dnn.hpp>

namespace vision_core {

std::vector<Detection> NewDetectorPostprocessor::postprocess(
    const float* output,
    const std::vector<int64_t>& shape,
    const cv::Size& frame_size,
    int network_width,
    int network_height,
    float confidence_threshold)
{
    std::vector<Detection> detections;
    
    // Parse output tensor
    // Apply confidence filtering
    // Perform NMS if needed
    // Scale boxes using BBoxProcessor
    
    return detections;
}

} // namespace vision_core
```

#### Step 3: Update CMakeLists.txt

Add to the `BUILD_OBJECT_DETECTION` section:

```cmake
if(BUILD_OBJECT_DETECTION)
    list(APPEND CORE_SOURCES
        src/object_detection/new_detector_postprocessor.cpp
    )
    list(APPEND CORE_HEADERS
        include/vision-core/object_detection/new_detector_postprocessor.hpp
    )
endif()
```

#### Step 4: Add Unit Tests

Create `tests/test_new_detector_postprocessor.cpp`:

```cpp
#include <gtest/gtest.h>
#include "vision-core/object_detection/new_detector_postprocessor.hpp"

TEST(NewDetectorPostprocessorTest, BasicProcessing) {
    // Arrange
    std::vector<float> mock_output = {/* test data */};
    std::vector<int64_t> shape = {1, 100, 6};
    cv::Size frame_size(1920, 1080);
    
    // Act
    auto detections = NewDetectorPostprocessor::postprocess(
        mock_output.data(), shape, frame_size, 640, 640, 0.5f
    );
    
    // Assert
    EXPECT_GT(detections.size(), 0);
}
```

## 5. Build Configuration Options

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_OBJECT_DETECTION` | `ON` | Build object detection postprocessors |
| `BUILD_INSTANCE_SEGMENTATION` | `OFF` | Build instance segmentation components |
| `BUILD_CLASSIFICATION` | `OFF` | Build classification postprocessors |
| `BUILD_OPTICAL_FLOW` | `OFF` | Build optical flow algorithms |
| `BUILD_TESTS` | `OFF` | Build unit tests with Google Test |

### Build Configurations

**Development Build** (with debugging):
```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON ..
cmake --build .
```

**Release Build** (optimized):
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_OBJECT_DETECTION=ON ..
cmake --build . -- -j$(nproc)
```

**Minimal Build** (core only):
```bash
cmake -DBUILD_OBJECT_DETECTION=OFF \
      -DBUILD_INSTANCE_SEGMENTATION=OFF \
      -DBUILD_CLASSIFICATION=OFF \
      -DBUILD_OPTICAL_FLOW=OFF ..
```

## 6. Testing Guidelines

### Running Tests

```bash
# Run all tests
ctest --output-on-failure

# Run specific test
ctest -R BBoxProcessorTest

# Run with verbose output
ctest -V

# Run in parallel
ctest -j$(nproc)
```

### Writing Tests

- Use **Google Test** framework
- Place tests in `tests/` directory
- Name test files: `test_<component_name>.cpp`
- Use descriptive test names: `TEST(ComponentName, DescriptiveTestName)`
- Follow AAA pattern: Arrange, Act, Assert

Example:
```cpp
TEST(YoloPostprocessorTest, HandlesEmptyOutput) {
    // Arrange
    std::vector<float> empty_output;
    std::vector<int64_t> shape = {1, 0, 85};
    
    // Act
    auto detections = YoloPostprocessor::postprocess(
        empty_output.data(), shape, {640, 640}, 640, 640
    );
    
    // Assert
    EXPECT_TRUE(detections.empty());
}
```

## 7. Integration with Other Projects

### Using as Git Submodule

In your main project (e.g., `tritonic` or `object-detection-inference`):

```bash
# Add as submodule
git submodule add https://github.com/olibartfast/vision-core.git external/vision-core
git submodule update --init --recursive
```

In your project's `CMakeLists.txt`:

```cmake
# Add vision-core as subdirectory
add_subdirectory(external/vision-core)

# Link to your target
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE vision-core::vision-core)
```

### Using as Installed Package

```bash
# Install vision-core
cd vision-core/build
cmake --install . --prefix /usr/local
```

In your project:

```cmake
find_package(vision-core REQUIRED)
target_link_libraries(my_app PRIVATE vision-core::vision-core)
```

### Example Integration Code

```cpp
#include <vision-core/object_detection/yolo_postprocessor.hpp>
#include <vision-core/core/detection.hpp>

// In your inference code
void process_inference_output(
    const float* raw_output,
    const std::vector<int64_t>& output_shape,
    const cv::Mat& original_image)
{
    // Use vision-core postprocessor
    auto detections = vision_core::YoloPostprocessor::postprocess(
        raw_output,
        output_shape,
        original_image.size(),
        640, 640,  // network dimensions
        0.25f,     // confidence threshold
        0.45f      // NMS threshold
    );
    
    // Process detections
    for (const auto& det : detections) {
        cv::rectangle(original_image, det.bbox, cv::Scalar(0, 255, 0), 2);
        std::cout << "Class: " << det.class_id 
                  << " Conf: " << det.confidence << std::endl;
    }
}
```

## 8. Component Architecture

### Core Components

**`core/detection.hpp`**
- Data structures for all vision tasks
- `Detection`: Bounding box + class + confidence
- `ClassificationResult`: Class prediction result

**`core/bbox_processor.hpp`**
- Bounding box transformation utilities
- Letterbox scaling support
- Coordinate system conversions

### Object Detection Components

**YOLO Family** (`object_detection/yolo_postprocessor.hpp`)
- Supports: YOLOv5, v6, v7, v8, v9, v10, v11, v12
- Auto-detects format from tensor shape
- Handles both anchor-based and anchor-free variants

**RT-DETR Family** (`object_detection/rtdetr_postprocessor.hpp`)
- Transformer-based detectors
- No NMS required (pre-filtered by model)
- Dual tensor output (boxes + scores)

### Future Components

- **Instance Segmentation**: YOLOv8-seg, YOLO11-seg
- **Classification**: Generic top-k classifier
- **Optical Flow**: RAFT implementation

## 9. Migration Tasks from Existing Projects

### Task List for Extracting Code from `tritonic` and `object-detection-inference`

- [ ] **Audit existing detector implementations**
  - Identify all YOLO variants in both repos
  - Document RT-DETR implementations
  - List segmentation and classification code
  
- [ ] **Extract shared postprocessing logic**
  - [ ] YOLO postprocessor (v5-v12)
  - [ ] RT-DETR postprocessor
  - [ ] Instance segmentation (YOLOv8-seg)
  - [ ] Classification utilities
  - [ ] NMS implementations
  
- [ ] **Migrate bounding box utilities**
  - [ ] Letterbox scaling functions
  - [ ] Coordinate transformations
  - [ ] Box clamping and validation
  
- [ ] **Update tritonic to use vision-core**
  - [ ] Remove duplicated postprocessor code
  - [ ] Add vision-core as submodule
  - [ ] Update CMake configuration
  - [ ] Update #includes to vision-core namespace
  - [ ] Test all detector backends with new library
  
- [ ] **Update object-detection-inference to use vision-core**
  - [ ] Remove duplicated detector code
  - [ ] Add vision-core as submodule
  - [ ] Refactor backend wrappers to use vision-core
  - [ ] Update build system
  - [ ] Verify all models work correctly
  
- [ ] **Documentation updates**
  - [ ] Update tritonic README with vision-core dependency
  - [ ] Update object-detection-inference README
  - [ ] Add migration guide for contributors
  - [ ] Document breaking changes (if any)
  
- [ ] **CI/CD Integration**
  - [ ] Set up vision-core CI pipeline
  - [ ] Update tritonic CI to test with vision-core
  - [ ] Update object-detection-inference CI
  - [ ] Add cross-repo compatibility tests

## 10. Best Practices

### Code Style

- Follow [NAMING_CONVENTIONS.md](NAMING_CONVENTIONS.md) strictly
- Use `snake_case` for functions and variables
- Use `PascalCase` for types
- Add `[[nodiscard]]` to functions returning important values
- Use `noexcept` where appropriate
- Prefer `const` and `constexpr`

### Error Handling

```cpp
// Validate inputs early
if (bbox.size() < 4) {
    throw std::invalid_argument("Bbox must have at least 4 elements");
}

// Use noexcept for operations that cannot fail
static void clamp_to_bounds(cv::Rect& box, const cv::Size& size) noexcept;
```

### Documentation

```cpp
/**
 * @brief Brief one-line description
 * 
 * Detailed explanation of what the function does,
 * including algorithm details if relevant.
 * 
 * @param param_name Description of parameter
 * @return Description of return value
 * @throws ExceptionType When this exception is thrown
 */
[[nodiscard]] static ReturnType function_name(ParamType param_name);
```

## 11. Performance Considerations

- Minimize memory allocations in hot paths
- Use `reserve()` for vectors when size is known
- Prefer `const&` for large objects in parameters
- Use move semantics for returns when appropriate
- Consider vectorization opportunities with OpenCV

## 12. Roadmap and Next Steps

### Short-term (Current Phase)

- [x] Set up repository structure
- [x] Implement core data structures
- [x] Implement YOLO postprocessor interface
- [x] Implement RT-DETR postprocessor interface
- [x] Create comprehensive documentation
- [ ] Implement YOLO postprocessor logic
- [ ] Implement RT-DETR postprocessor logic
- [ ] Add unit tests for all components
- [ ] Set up CI/CD pipeline

### Medium-term

- [ ] Extract code from tritonic
- [ ] Extract code from object-detection-inference
- [ ] Add instance segmentation support
- [ ] Add classification postprocessor
- [ ] Add benchmark suite
- [ ] Optimize performance critical paths

### Long-term

- [ ] RAFT optical flow implementation
- [ ] Python bindings (pybind11)
- [ ] Additional detector architectures (DETR, Faster R-CNN, etc.)
- [ ] Support for 3D bounding boxes
- [ ] Tracking utilities (SORT, DeepSORT)

## 13. Getting Help

- **Issues**: Open an issue on GitHub for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Contributing**: See CONTRIBUTING.md (when created)
- **Contact**: @olibartfast

## 14. Resources

- [Modern C++ Guidelines](https://isocpp.github.io/CppCoreGuidelines/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [CMake Documentation](https://cmake.org/documentation/)
- [Google Test Documentation](https://google.github.io/googletest/)