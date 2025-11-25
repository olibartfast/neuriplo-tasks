#include "vision-core/core/task_factory.hpp"
#include <algorithm>
#include <stdexcept>
#include <mutex>

namespace vision_core {

// Shared mutex for registry access
static std::mutex& getRegistryMutex() {
    static std::mutex registry_mutex;
    return registry_mutex;
}

std::map<std::string, TaskFactory::TaskCreator>& TaskFactory::getRegistry() {
    static std::map<std::string, TaskCreator> registry;
    return registry;
}

void TaskFactory::registerTask(const std::string& model_type, TaskCreator creator) {
    if (model_type.empty()) {
        throw std::invalid_argument("Cannot register task with empty model type");
    }
    if (!creator) {
        throw std::invalid_argument("Cannot register null task creator");
    }

    const auto normalized = normalizeModelType(model_type);
    std::lock_guard<std::mutex> lock(getRegistryMutex());
    auto& registry = getRegistry();
    registry[normalized] = std::move(creator);
}

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

    const auto normalized_type = normalizeModelType(model_type);
    if (normalized_type.empty()) {
        throw std::invalid_argument("Model type string is empty");
    }

    std::lock_guard<std::mutex> lock(getRegistryMutex());
    auto& registry = getRegistry();
    auto it = registry.find(normalized_type);
    if (it != registry.end()) {
        return it->second(model_info);
    }
    
    throw std::invalid_argument("Unrecognized model type: " + model_type);
}

std::string TaskFactory::normalizeModelType(const std::string& model_type) {
    std::string normalized;
    normalized.reserve(model_type.size());

    for (char c : model_type) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            continue;
        }
        if (c == '-' || c == '_' || c == '/') {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    // Normalize YOLO variants to base "yolo"
    if (normalized.rfind("yolov", 0) == 0 || normalized.rfind("yolo", 0) == 0) {
        return "yolo";
    }

    return normalized;
}

} // namespace vision_core
