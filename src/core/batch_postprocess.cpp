#include "neuriplo/tasks/core/batch_postprocess.hpp"

#include "neuriplo/tasks/core/result_types.hpp"

#include <stdexcept>

namespace neuriplo_tasks {

namespace {

void validateBatchSize(int batch_size, const ModelInfo& model_info) {
    if (batch_size <= 0) {
        throw std::invalid_argument("batchPostprocess: batch_size must be positive");
    }
    if (model_info.max_batch_size_ > 0 && batch_size > model_info.max_batch_size_) {
        throw std::invalid_argument("batchPostprocess: batch_size exceeds ModelInfo.max_batch_size_");
    }
}

[[nodiscard]] bool isAggregateBatchTask(TaskType task_type) noexcept {
    return task_type == TaskType::GaussianSplatting;
}

[[nodiscard]] bool trustsTensorBatchCount(TaskType task_type) noexcept {
    return task_type == TaskType::DepthEstimation || task_type == TaskType::PoseEstimation;
}

[[nodiscard]] bool allowsVariableResultCount(TaskType task_type) noexcept {
    return task_type == TaskType::Detection || task_type == TaskType::InstanceSegmentation;
}

BatchPostprocessOutput finalizeOutput(std::vector<Result> results, int batch_size, TaskType task_type) {
    BatchPostprocessOutput output;
    output.batch_size = batch_size;

    if (results.empty()) {
        output.results = std::move(results);
        return output;
    }

    if (results.size() == static_cast<size_t>(batch_size)) {
        output.results = std::move(results);
        return output;
    }

    if (results.size() == 1) {
        output.results = std::move(results);
        return output;
    }

    if (allowsVariableResultCount(task_type)) {
        output.results = std::move(results);
        return output;
    }

    if (trustsTensorBatchCount(task_type) || isAggregateBatchTask(task_type)) {
        output.results = std::move(results);
        output.batch_size = static_cast<int>(output.results.size());
        return output;
    }

    throw std::invalid_argument("batchPostprocess: result count does not match batch_size");
}

} // namespace

BatchPostprocessOutput batchPostprocess(TaskInterface& task, const Size& frame_size, const std::vector<Tensor>& tensors,
                                        int batch_size) {
    validateBatchSize(batch_size, task.getModelInfo());

    auto results = task.postprocess(frame_size, tensors);
    return finalizeOutput(std::move(results), batch_size, task.getTaskType());
}

} // namespace neuriplo_tasks
