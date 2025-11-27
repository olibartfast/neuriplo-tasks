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
 * 
 * The factory uses a registration pattern allowing both built-in and custom tasks.
 * Built-in tasks are registered automatically on first use.
 */
class TaskFactory {
public:
    using TaskCreator = std::function<std::unique_ptr<TaskInterface>(const ModelInfo&)>;

    /**
     * @brief Register a custom task creator
     * 
     * Allows downstream projects to register custom task implementations.
     * Registration is thread-safe and can be called at any time.
     * 
     * @param model_type Model type identifier (will be normalized)
     * @param creator Factory function to create the task
     * 
     * Example:
     * @code
     * TaskFactory::registerTask("custom-model", 
     *     [](const ModelInfo& info) { 
     *         return std::make_unique<CustomTask>(info); 
     *     });
     * @endcode
     */
    static void registerTask(const std::string& model_type, TaskCreator creator);

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
    /**
     * @brief Get the task creators registry (lazy initialization)
     */
    static std::map<std::string, TaskCreator>& getRegistry();
    
    /**
     * @brief Validate input sizes
     */
    static void validateInputSizes(const std::vector<std::vector<int64_t>>& input_sizes);

    /**
     * @brief Normalize model type names for lookup
     */
    static std::string normalizeModelType(const std::string& model_type);
};

/**
 * @brief Force registration of all built-in task types
 * 
 * Call this function to ensure all static task registrars are executed.
 * This is mainly needed for testing or if using the library as a static library.
 */
void registerAllTasks();

} // namespace vision_core
