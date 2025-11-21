# Design Decision: Tritonic Architecture Pattern

## Analysis Summary

After analyzing both `object-detection-inference` and `tritonic` repositories, we selected **tritonic's task-based architecture** as the foundation for `vision-core`.

## Comparison

### object-detection-inference Architecture
```cpp
// Inheritance-based detector pattern
class Detector {
    virtual std::vector<Detection> postprocess(...) = 0;
    virtual cv::Mat preprocess_image(...) = 0;
};

class YoloVn : public Detector { };
class RtDetr : public Detector { };
```

**Pros:**
- Simple inheritance hierarchy
- Each detector is self-contained
- Clear separation of detector types

**Cons:**
- Mixes preprocessing and postprocessing concerns
- Tied to specific inference backends
- Harder to share common utilities across detector families
- Less flexible for multi-task scenarios

### tritonic Architecture ✅ (Selected)
```cpp
// Task-based interface pattern
class TaskInterface {
    virtual std::vector<Result> postprocess(...) = 0;
    virtual std::vector<uint8_t> preprocess(...) = 0;
    virtual TaskType getTaskType() = 0;
};

using Result = std::variant<Classification, Detection, InstanceSegmentation, OpticalFlow>;

class YOLO : public TaskInterface { };
class YOLOSeg : public YOLO { };  // Extends for segmentation
class RFDetr : public TaskInterface { };
class RFDetrSeg : public RFDetr { };  // Extends for segmentation
```

**Pros:**
- ✅ **Task-oriented design**: Organizes by computer vision task (detection, segmentation, classification)
- ✅ **Polymorphic results**: `std::variant` allows type-safe multi-task support
- ✅ **Better code reuse**: `YOLOSeg extends YOLO`, `RFDetrSeg extends RFDetr`
- ✅ **Framework agnostic**: No dependency on specific inference engines
- ✅ **Cleaner separation**: Pure algorithms, no backend coupling
- ✅ **Consistent naming**: `class_id`, `class_confidence` throughout
- ✅ **Modern C++**: Uses variants, optionals, structured bindings

**Cons:**
- Slightly more complex type system (variant handling)
- Requires understanding of visitor pattern for result access

## Applied Architecture in vision-core

We adapted tritonic's pattern for a **pure postprocessing library**:

### Core Abstractions

```cpp
// Data Structures
struct Detection {
    cv::Rect bbox;
    float confidence;
    int class_id;
};

struct InstanceSegmentation : public Detection {
    std::vector<uint8_t> mask_data;
    int mask_width, mask_height;
};

struct ClassificationResult {
    int class_id;
    float confidence;
    std::string class_name;
};
```

### Postprocessor Pattern

```cpp
// Static methods - no state, pure functions
class YoloPostprocessor {
public:
    [[nodiscard]] static std::vector<Detection> postprocess(
        const TensorElement* output,
        const std::vector<int64_t>& shape,
        const cv::Size& frame_size,
        int network_width,
        int network_height,
        float confidence_threshold,
        float nms_threshold
    );
private:
    static std::tuple<...> process_v567_format(...);
    static std::tuple<...> process_ultralytics_format(...);
};
```

## Key Design Decisions

### 1. **Static Methods Over Inheritance**
- No base class hierarchy for postprocessors
- Each postprocessor is independent
- **Rationale**: Postprocessing is stateless; no need for virtual dispatch

### 2. **Separation of Concerns**
```
vision-core (algorithms only)
    ├── No preprocessing
    ├── No inference
    └── Pure postprocessing + utilities

tritonic/object-detection-inference (application logic)
    ├── Preprocessing
    ├── Inference engine integration
    └── Uses vision-core for postprocessing
```

### 3. **Component Organization**
```
core/                    # Shared utilities
├── detection.hpp        # Common data structures
└── bbox_processor.hpp   # Coordinate transformations

object_detection/        # Detection algorithms
├── yolo_postprocessor.hpp
└── rtdetr_postprocessor.hpp

instance_segmentation/   # Segmentation algorithms
└── yolo_segmentation_postprocessor.hpp

classification/          # Classification algorithms
└── classifier_postprocessor.hpp
```

### 4. **Auto-Format Detection**
From tritonic's YOLO implementation:
```cpp
// Auto-detect YOLO format from tensor shape
const bool is_v567_format = shape[1] > shape[2];
const auto [boxes, scores, class_ids] = is_v567_format 
    ? process_v567_format(...)
    : process_ultralytics_format(...);
```

### 5. **Inheritance Where It Makes Sense**
Following tritonic's `YOLOSeg extends YOLO` pattern:
```cpp
struct InstanceSegmentation : public Detection {
    // Adds mask-specific fields to detection
};
```

## Benefits of This Approach

1. **Single Source of Truth**: All detector algorithms in one place
2. **Framework Agnostic**: Works with ONNX, TensorRT, Triton, OpenVINO, etc.
3. **Easy Migration**: Both tritonic and object-detection-inference can use it
4. **Testable**: Pure functions are easy to unit test
5. **Performance**: No virtual dispatch overhead, compiler can inline
6. **Modern C++**: Uses C++17 features properly

## Migration Path

### For tritonic
```cpp
// Before
class YOLO : public TaskInterface {
    std::vector<Result> postprocess(...) override {
        // Complex postprocessing logic here
    }
};

// After
class YOLO : public TaskInterface {
    std::vector<Result> postprocess(...) override {
        auto detections = vision_core::YoloPostprocessor::postprocess(...);
        std::vector<Result> results;
        for (const auto& det : detections) {
            results.push_back(Detection{det.bbox, det.confidence, det.class_id});
        }
        return results;
    }
};
```

### For object-detection-inference
```cpp
// Before
class YoloVn : public Detector {
    std::vector<Detection> postprocess(...) override {
        // Duplicated logic
    }
};

// After
class YoloVn : public Detector {
    std::vector<Detection> postprocess(...) override {
        return vision_core::YoloPostprocessor::postprocess(...);
    }
};
```

## Conclusion

The tritonic architecture was selected because it:
- Better aligns with computer vision task organization
- Provides cleaner code reuse through strategic inheritance
- Uses modern C++ idioms effectively
- Separates concerns cleanly

We simplified it further for vision-core by:
- Removing preprocessing (not needed in pure algorithm library)
- Using static methods (no state needed)
- Keeping only the algorithmic core
- Maintaining the excellent naming conventions

This creates a clean, reusable foundation that both projects can depend on.
