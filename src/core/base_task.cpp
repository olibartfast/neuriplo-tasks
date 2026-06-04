#include "vision-core/core/base_task.hpp"

#include <stdexcept>
#include <utility>

namespace vision_core {

BaseTask::BaseTask(const ModelInfo& model_info, std::string empty_input_message)
    : TaskInterface(model_info), empty_input_message_(std::move(empty_input_message)) {}

std::vector<std::vector<uint8_t>> BaseTask::preprocess(const std::vector<cv::Mat>& imgs) {
    std::vector<std::vector<uint8_t>> results;
    results.reserve(imgs.size());

    const auto& preprocessor = getPreprocessor();
    for (const auto& img : imgs) {
        if (img.empty()) {
            throw std::invalid_argument(empty_input_message_);
        }
        results.push_back(preprocessor.preprocess(img));
    }

    return results;
}

std::vector<Result> BaseTask::postprocess(const cv::Size& frame_size, const std::vector<Tensor>& tensors) {
    if (!validateOutputs(tensors)) {
        return {};
    }

    return decode(frame_size, tensors);
}

bool BaseTask::validateOutputs(const std::vector<Tensor>& tensors) const { return !tensors.empty(); }

} // namespace vision_core
