# Using vision-core as a Dependency in TritonIC

This guide explains how to integrate `vision-core` into the TritonIC project to eliminate code duplication.

## Overview

The `vision-core` library now provides:
- **TaskInterface**: Base class for all CV tasks (detection, classification, segmentation, etc.)
- **TaskFactory**: Factory pattern for creating task instances by model type
- **Result structures**: Unified result types (Detection, Classification, InstanceSegmentation, OpticalFlow, VideoClassification) in `result_types.hpp`
- **ModelInfo**: Framework-agnostic model metadata structure
- All preprocessing and postprocessing algorithms

## Integration Steps

### 1. Add vision-core as a Git Submodule

```bash
cd tritonic
git submodule add https://github.com/olibartfast/vision-core.git external/vision-core
git submodule update --init --recursive
```

### 2. Update tritonic/CMakeLists.txt

Add vision-core as a subdirectory and link against it:

```cmake
# After project() declaration
add_subdirectory(external/vision-core)

# When defining your executable or library
target_link_libraries(tritonic-client
    PRIVATE
        vision-core::vision-core
        # ... other dependencies
)
```

### 3. Remove Duplicated Code from tritonic

The following can be removed from tritonic as they're now in vision-core:

**Headers to remove:**
- `include/TaskInterface.hpp` → use `vision-core/core/task_interface.hpp`
- `src/tasks/task_factory.hpp` → use `vision-core/core/task_factory.hpp`

**Source files to remove:**
- All files in `src/tasks/object_detection/`
- All files in `src/tasks/classification/`
- All files in `src/tasks/instance_segmentation/`
- All files in `src/tasks/optical_flow/`
- All files in `src/tasks/video_classification/`
- `src/tasks/task_factory.cpp`

### 4. Update Include Statements

Replace tritonic-specific includes with vision-core includes:

**Before:**
```cpp
#include "TaskInterface.hpp"
#include "task_factory.hpp"
```

**After:**
```cpp
#include <vision-core/core/task_interface.hpp>
#include <vision-core/core/task_factory.hpp>
#include <vision-core/core/model_info.hpp>
```

### 5. Update Type Mappings

Map tritonic types to vision-core types:

```cpp
// In your tritonic code:
using vision_core::TaskInterface;
using vision_core::TaskFactory;
using vision_core::ModelInfo;
using vision_core::Result;
using vision_core::Detection;
using vision_core::Classification;
using vision_core::InstanceSegmentation;
using vision_core::OpticalFlow;
using vision_core::VideoClassification;
using vision_core::TaskType;
using vision_core::TensorElement;
```

### 6. Adapt TritonModelInfo to ModelInfo

Create a conversion function:

```cpp
vision_core::ModelInfo convertToVisionCoreModelInfo(const TritonModelInfo& triton_info) {
    vision_core::ModelInfo model_info;
    model_info.input_shapes = triton_info.input_shapes;
    model_info.input_formats = triton_info.input_formats;
    model_info.input_names = triton_info.input_names;
    model_info.output_names = triton_info.output_names;
    model_info.input_types = triton_info.input_types;
    model_info.max_batch_size_ = triton_info.max_batch_size_;
    model_info.batch_size_ = triton_info.batch_size_;
    return model_info;
}
```

### 7. Update Task Creation

**Before:**
```cpp
std::unique_ptr<TaskInterface> task = TaskFactory::createTaskInstance(model_type, tritonModelInfo);
```

**After:**
```cpp
auto visionCoreModelInfo = convertToVisionCoreModelInfo(tritonModelInfo);
std::unique_ptr<vision_core::TaskInterface> task = 
    vision_core::TaskFactory::createTaskInstance(model_type, visionCoreModelInfo);
```

## Example: Updated tritonic/src/main/client.cpp

```cpp
#include <vision-core/core/task_factory.hpp>
#include <vision-core/core/task_interface.hpp>
#include <vision-core/core/model_info.hpp>
#include "ITriton.hpp"
#include "Triton.hpp"
#include "Config.hpp"

// Convert Triton-specific model info to vision-core ModelInfo
vision_core::ModelInfo convertModelInfo(const TritonModelInfo& triton_info) {
    vision_core::ModelInfo info;
    info.input_shapes = triton_info.input_shapes;
    info.input_formats = triton_info.input_formats;
    info.input_names = triton_info.input_names;
    info.output_names = triton_info.output_names;
    info.input_types = triton_info.input_types;
    info.max_batch_size_ = triton_info.max_batch_size_;
    info.batch_size_ = triton_info.batch_size_;
    return info;
}

std::vector<vision_core::Result> processSource(
    const std::vector<cv::Mat>& source,
    const std::unique_ptr<vision_core::TaskInterface>& task,
    const std::unique_ptr<ITriton>& tritonClient) {
    
    // Preprocess
    auto preprocessed = task->preprocess(source);
    
    // Infer
    auto [infer_results, infer_shapes] = tritonClient->infer(preprocessed);
    
    // Postprocess
    cv::Size frame_size = source[0].size();
    return task->postprocess(frame_size, infer_results, infer_shapes);
}

int main(int argc, const char* argv[]) {
    // ... config loading ...
    
    // Create Triton client
    std::unique_ptr<ITriton> tritonClient = 
        std::make_unique<Triton>(url, protocol, model_name, model_version, verbose);
    
    // Get model info from Triton
    TritonModelInfo tritonModelInfo = 
        tritonClient->getModelInfo(model_name, server_address, input_sizes);
    
    // Convert to vision-core format
    auto visionCoreModelInfo = convertModelInfo(tritonModelInfo);
    
    // Create task using vision-core factory
    std::unique_ptr<vision_core::TaskInterface> task = 
        vision_core::TaskFactory::createTaskInstance(model_type, visionCoreModelInfo);
    
    // Process images/video
    // ... rest of processing logic ...
}
```

## Benefits

1. **No Code Duplication**: All CV algorithms maintained in one place (vision-core)
2. **Framework Agnostic**: vision-core can be used with any inference framework (ONNX Runtime, TensorRT, Triton, etc.)
3. **Easier Testing**: vision-core has its own test suite
4. **Cleaner Separation**: tritonic focuses on Triton integration, vision-core focuses on CV algorithms
5. **Reusability**: Other projects can use vision-core directly

## Migration Checklist

- [ ] Add vision-core as submodule
- [ ] Update CMakeLists.txt to link vision-core
- [ ] Remove duplicated task implementations from tritonic
- [ ] Update include paths
- [ ] Create ModelInfo conversion utility
- [ ] Update main.cpp and processing functions
- [ ] Remove old task_factory from tritonic
- [ ] Update tests to use vision-core types
- [ ] Build and verify all functionality works
- [ ] Update tritonic documentation

## Notes

- The `TensorElement` type is compatible between vision-core and tritonic
- Result structures use inheritance (Classification → Detection → InstanceSegmentation)
- TaskType enum is identical between both libraries
- Error handling uses the same InputDimensionError exception
