#include "vision-core/core/task_factory.hpp"
#include <algorithm>
#include <stdexcept>

namespace vision_core {

// Forward declarations of task implementations will be added here
// as we implement each model type

std::map<std::string, TaskFactory::TaskCreator> TaskFactory::task_creators_ = {
    // Detection models - to be implemented
    // {"yolov8", [](const ModelInfo& info) { return std::make_unique<YoloTask>(info); }},
    // {"rtdetr", [](const ModelInfo& info) { return std::make_unique<RtDetrTask>(info); }},
    // ... more models
};

void TaskFactory::validateInputSizes(const std::vector<std::vector<int64_t>>& input_sizes) {
    if (input_sizes.empty()) {
        throw InputDimensionError("Input sizes vector is empty");
    }
    for (const auto& size : input_sizes) {
        if (size.empty()) {
            throw InputDimensionError("An input size vector is empty");
        }
        if (std::any_of(size.begin(), size.end(), [](int64_t s) { return s <= 0; })) {
            throw InputDimensionError("Non-positive input size detected");
        }
    }
}

std::unique_ptr<TaskInterface> TaskFactory::createTaskInstance(
    const std::string& model_type,
    const ModelInfo& model_info) {
    
    validateInputSizes(model_info.input_shapes);
    
    auto it = task_creators_.find(model_type);
    if (it != task_creators_.end()) {
        return it->second(model_info);
    }
    
    throw std::invalid_argument("Unrecognized model type: " + model_type);
}

} // namespace vision_core
