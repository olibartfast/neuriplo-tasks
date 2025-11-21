# Modern C++ Naming Conventions - Applied Changes

## Overview

This document details the modern C++ naming conventions applied to the `vision-core` library to ensure consistency, readability, and adherence to industry best practices.

## Naming Convention Rules

### 1. Types (Classes, Structs, Enums, Type Aliases)
**Convention**: `PascalCase`

**Examples**:
```cpp
class YoloPostprocessor { };
struct Detection { };
struct ClassificationResult { };
using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;
```

### 2. Functions and Methods
**Convention**: `snake_case`

**Before**:
```cpp
static cv::Rect calculateBoundingBox(...);
static cv::Rect scaleToOriginal(...);
static void clampBox(...);
```

**After**:
```cpp
static cv::Rect calculate_bounding_box(...);
static cv::Rect scale_to_original(...);
static void clamp_to_bounds(...);
```

### 3. Variables and Parameters
**Convention**: `snake_case`

**Before**:
```cpp
const cv::Size& img_size
int network_width
float conf_threshold
```

**After** (improved for clarity):
```cpp
const cv::Size& image_size          // More descriptive
int network_width                    // Already good
float confidence_threshold           // Full word instead of abbreviation
```

### 4. Struct/Class Members
**Convention**: `snake_case` for public members

**Before**:
```cpp
struct Detection {
    cv::Rect bbox;
    float score;              // Ambiguous
    int label;                // Generic
    std::vector<cv::Point> mask;
};
```

**After**:
```cpp
struct Detection {
    cv::Rect bbox;
    float confidence{0.0f};   // Clearer intent + default value
    int class_id{-1};         // More specific + default value
    std::vector<cv::Point> mask;
};
```

### 5. Private Member Variables
**Convention**: `snake_case_` (trailing underscore)

```cpp
class MyClass {
private:
    int count_;                    // Private member
    std::string name_;             // Private member
};
```

### 6. Constants
**Convention**: `kConstantName` or `CONSTANT_NAME`

```cpp
static constexpr float kDefaultConfidence = 0.25f;
static constexpr int kMaxDetections = 300;

// Or for macros/legacy:
#define MAX_CLASSES 80
```

## Modern C++ Features Applied

### 1. Default Member Initializers
**Before**:
```cpp
struct Detection {
    float score;
    int label;
    Detection() : score(0.0f), label(-1) {}
};
```

**After**:
```cpp
struct Detection {
    float confidence{0.0f};
    int class_id{-1};
    Detection() = default;
};
```

### 2. `[[nodiscard]]` Attribute
Applied to functions where ignoring the return value is likely a bug:

```cpp
[[nodiscard]] static cv::Rect calculate_bounding_box(...);
[[nodiscard]] static std::vector<Detection> postprocess(...);
```

### 3. `noexcept` Specifier
Applied to functions guaranteed not to throw:

```cpp
static cv::Rect scale_to_original(...) noexcept;
static void clamp_to_bounds(cv::Rect& box, const cv::Size& image_size) noexcept;
```

### 4. `constexpr` for Compile-Time Constants
```cpp
static constexpr int kDefaultNetworkSize = 640;
```

### 5. Use of `std::clamp` (C++17)
**Before**:
```cpp
box.x = std::max(0, std::min(box.x, image_size.width - 1));
```

**After**:
```cpp
box.x = std::clamp(box.x, 0, image_size.width - 1);
```

### 6. Designated Initializers and Brace Initialization
```cpp
Detection det{bbox, 0.95f, 3};  // Constructor
auto size = cv::Size{640, 640}; // Brace init
```

## File Naming Conventions

**Convention**: `snake_case.hpp` and `snake_case.cpp`

**Applied**:
- `detection.hpp` (not `Detection.hpp`)
- `bbox_processor.hpp` (not `BBoxProcessor.hpp`)
- `yolo_postprocessor.hpp` (not `YoloPostprocessor.hpp`)

## Documentation Improvements

### Enhanced Doxygen Comments

**Before**:
```cpp
/**
 * @brief Calculate bounding box with letterbox scaling
 */
static cv::Rect calculateBoundingBox(...);
```

**After**:
```cpp
/**
 * @brief Calculate bounding box with letterbox scaling
 * 
 * Converts bounding box coordinates from network space to original image space,
 * accounting for letterbox padding used during preprocessing.
 * 
 * @param image_size Original image size
 * @param bbox Raw bbox coordinates [x_center, y_center, width, height]
 * @param network_width Network input width
 * @param network_height Network input height
 * @return Scaled bounding box in original image coordinates
 * @throws std::invalid_argument if bbox has fewer than 4 elements
 */
[[nodiscard]] static cv::Rect calculate_bounding_box(...);
```

## Summary of Changes

### Core Files Updated

1. **`include/vision-core/core/detection.hpp`**
   - `score` → `confidence`
   - `label` → `class_id`
   - Added default member initializers
   - Added constructors with parameters

2. **`include/vision-core/core/bbox_processor.hpp`**
   - `calculateBoundingBox` → `calculate_bounding_box`
   - `scaleToOriginal` → `scale_to_original`
   - `clampBox` → `clamp_to_bounds`
   - `img_size` → `image_size`
   - Added `[[nodiscard]]` and `noexcept`

3. **`src/core/bbox_processor.cpp`**
   - Updated all function names to match header
   - Used `std::clamp` instead of nested `std::max`/`std::min`
   - Added `const` for intermediate calculations
   - Improved variable naming (`r_w` → `ratio_width`)

4. **`include/vision-core/object_detection/yolo_postprocessor.hpp`**
   - `conf_threshold` → `confidence_threshold`
   - `processV567Format` → `process_v567_format`
   - `processUltralyticsFormat` → `process_ultralytics_format`
   - Added comprehensive documentation
   - Added `[[nodiscard]]` attribute

## Benefits of These Changes

1. **Consistency**: All code follows the same naming patterns
2. **Readability**: Clear, descriptive names improve code understanding
3. **Modern**: Uses C++17 features and best practices
4. **Safety**: `[[nodiscard]]` and `noexcept` prevent common errors
5. **Documentation**: Enhanced comments improve API usability
6. **Industry Standard**: Follows conventions used by major C++ projects

## References

- Google C++ Style Guide
- C++ Core Guidelines (by Bjarne Stroustrup and Herb Sutter)
- LLVM Coding Standards
- Effective Modern C++ (Scott Meyers)
