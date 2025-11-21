# DEVELOPMENT.md

## 1. Project Overview and Architecture

The vision-core library serves as the foundational framework for computer vision applications. It is designed to facilitate various computer vision tasks such as object detection, classification, segmentation, and optical flow algorithms. This document provides comprehensive guidelines for developers to contribute to the library effectively.

## 2. Repository Structure

- **src/**: Contains all the source code of the library.
  - **components/**: Subdirectory for modular components like object detection, classification, etc.
- **include/**: Header files for public interfaces.
- **tests/**: Unit tests and integration tests for validating functionality.
- **CMakeLists.txt**: Build configuration file for managing builds with CMake.

## 3. Step-by-step Implementation Guide

1. **Setting Up Your Development Environment**  
   - Clone the repository: `git clone https://github.com/olibartfast/vision-core.git`
   - Navigate to the cloned directory: `cd vision-core`
   - Open your codespace.

2. **Installing Dependencies**  
   - Ensure CMake is installed.
   - Install any additional libraries or tools required by the project as defined in the `README.md`.
  
3. **Implement Core Functions**: Each component will have its implementation guide.

## 4. Code Examples

### Object Detection Example
```cpp
class ObjectDetector {
public:
    void detectObjects(const Image& img);
};
```

### Classification Example
```cpp
class Classifier {
public:
    std::string classify(const Image& img);
};
```

## 5. Build Instructions

1. Open your terminal and navigate to the project root.
2. Create a build directory: `mkdir build && cd build`
3. Run CMake to configure the project: `cmake ..`
4. Compile the project: `make`

## 6. Testing Guidelines

- Use Google Test framework for unit testing.
- Run tests using: `make test`

## 7. Integration Instructions

To integrate the vision-core library with Tritonic and object-detection-inference, follow these steps:
1. Link the libraries in your CMake configuration.
2. Ensure all necessary headers are included correctly.

## 8. Component Breakdown

- **Object Detection**: Describes the logic for identifying objects in images.
- **Classification**: Involves methods for determining the categories of detected objects.
- **Segmentation**: Provides techniques for delineating objects within images.
- **Optical Flow**: Explains the tracking of moving objects across multiple frames.

## 9. CMake Configuration Details

Make sure to properly configure your `CMakeLists.txt` to include all relevant source files and establish link libraries:
```cmake
add_executable(vision-core main.cpp)
target_link_libraries(vision-core ${OpenCV_LIBS})
```

## 10. Next Steps and Roadmap

- **Feature Development**: Focus on implementing more features based on user feedback.
- **Documentation Enhancements**: Continuous improvement on documentation.
- **Performance Optimization**: Review current implementations for potential speed-ups.