#include "neuriplo/tasks/core/batch_preprocess.hpp"

#include <stdexcept>

namespace neuriplo_tasks {

namespace {

void validateBatchRequest(const BatchRequest& request, const ModelInfo& model_info) {
    if (request.images.empty()) {
        throw std::invalid_argument("batchPreprocess: empty image batch");
    }
    if (model_info.max_batch_size_ > 0 && request.images.size() > static_cast<size_t>(model_info.max_batch_size_)) {
        throw std::invalid_argument("batchPreprocess: batch size exceeds ModelInfo.max_batch_size_");
    }
}

} // namespace

BatchPreprocessOutput batchPreprocess(TaskInterface& task, const BatchRequest& request) {
    validateBatchRequest(request, task.getModelInfo());

    BatchPreprocessOutput output;
    output.buffers = task.preprocess(request.images);
    output.batch_size = static_cast<int>(request.images.size());
    return output;
}

} // namespace neuriplo_tasks
