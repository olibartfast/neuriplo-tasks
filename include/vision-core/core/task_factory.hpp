#pragma once

#include "vision-core/core/task_interface.hpp"
#include <memory>
#include <string>
#include <map>
#include <functional>

namespace vision_core {

/**
 * @brief Factory for creating task instances
 * 
 * Creates appropriate task implementations based on model type string.
 * Supports all common CV tasks: detection, classification, segmentation, etc.
 */
class TaskFactory {
public:
    /**
     * @brief Create a task instance for the given model type
     * @param model_type Model type identifier (e.g., "yolov8", "rtdetr", "resnet50")
     * @param model_info Model configuration information
     * @return Unique pointer to task interface
     * @throws std::invalid_argument if model type is invalid
     * @throws InputDimensionError if model info has invalid dimensions
     */
    static std::unique_ptr<TaskInterface> createTaskInstance(
        const std::string& model_type, 
        const ModelInfo& model_info);

private:
    using TaskCreator = std::function<std::unique_ptr<TaskInterface>(const ModelInfo&)>;
    static std::map<std::string, TaskCreator> task_creators_;
    
    /**
     * @brief Validate input sizes
     */
    static void validateInputSizes(const std::vector<std::vector<int64_t>>& input_sizes);
};

} // namespace vision_core
