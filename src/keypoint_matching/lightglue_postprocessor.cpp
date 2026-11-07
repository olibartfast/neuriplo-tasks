#include "neuriplo/tasks/keypoint_matching/lightglue_postprocessor.hpp"

#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace neuriplo_tasks {

namespace {

struct LightGlueOutputs {
    const Tensor* keypoints{nullptr};
    const Tensor* matches{nullptr};
    const Tensor* scores{nullptr};
};

std::size_t elementCount(const Tensor& tensor) {
    std::size_t count = 1;
    for (int64_t dim : tensor.shape) {
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

bool holdsInteger(const Tensor& tensor) {
    return !tensor.data.empty() && !std::holds_alternative<float>(tensor.data.front());
}

/**
 * @brief Group the export's outputs by shape
 *
 * keypoints [2, K, 2], matches [M, 3] (int), mscores [M] (float). Unrelated
 * tensors are ignored.
 */
LightGlueOutputs classifyOutputs(const std::vector<Tensor>& tensors) {
    LightGlueOutputs outputs;
    for (const Tensor& tensor : tensors) {
        const std::vector<int64_t>& shape = tensor.shape;
        if (shape.size() == 3 && shape[0] == 2 && shape[2] == 2) {
            if (outputs.keypoints == nullptr) {
                outputs.keypoints = &tensor;
            }
            continue;
        }
        if (shape.size() == 2 && shape[1] == 3 && (tensor.data.empty() || holdsInteger(tensor))) {
            if (outputs.matches == nullptr) {
                outputs.matches = &tensor;
            }
            continue;
        }
        if (shape.size() == 1 && !holdsInteger(tensor) && outputs.scores == nullptr) {
            outputs.scores = &tensor;
        }
    }
    return outputs;
}

void requireShape(const Tensor& tensor, std::size_t expected_elements, const char* name) {
    if (elementCount(tensor) != tensor.data.size() || tensor.data.size() != expected_elements) {
        throw std::invalid_argument(std::string("LightGlue '") + name + "' output expects " +
                                    std::to_string(expected_elements) + " elements but holds " +
                                    std::to_string(tensor.data.size()));
    }
}

} // namespace

LightGluePostprocessor::LightGluePostprocessor(float score_threshold, Size input_size)
    : score_threshold_(score_threshold), input_size_(input_size) {}

KeypointMatching LightGluePostprocessor::postprocess(const std::vector<Tensor>& tensors, const Size& source_size,
                                                     const Size& target_size) {
    if (tensors.empty()) {
        return {};
    }

    const LightGlueOutputs outputs = classifyOutputs(tensors);
    if (outputs.keypoints == nullptr) {
        throw std::invalid_argument("LightGlue: no [2, K, 2] keypoints output found");
    }
    if (outputs.matches == nullptr) {
        throw std::invalid_argument("LightGlue: no [M, 3] integer matches output found");
    }
    if (outputs.scores == nullptr) {
        throw std::invalid_argument("LightGlue: no [M] float mscores output found");
    }

    const std::size_t keypoint_count = static_cast<std::size_t>(outputs.keypoints->shape[1]);
    const std::size_t match_count = static_cast<std::size_t>(outputs.matches->shape[0]);
    requireShape(*outputs.keypoints, keypoint_count * 4, "keypoints");
    requireShape(*outputs.matches, match_count * 3, "matches");
    requireShape(*outputs.scores, match_count, "mscores");
    if (input_size_.width <= 0 || input_size_.height <= 0) {
        throw std::invalid_argument("LightGlue: model input size must be positive");
    }

    const std::vector<TensorElement>& keypoint_data = outputs.keypoints->data;
    const std::vector<TensorElement>& match_data = outputs.matches->data;
    const std::vector<TensorElement>& score_data = outputs.scores->data;

    const float source_scale_x = static_cast<float>(source_size.width) / static_cast<float>(input_size_.width);
    const float source_scale_y = static_cast<float>(source_size.height) / static_cast<float>(input_size_.height);
    const float target_scale_x = static_cast<float>(target_size.width) / static_cast<float>(input_size_.width);
    const float target_scale_y = static_cast<float>(target_size.height) / static_cast<float>(input_size_.height);

    KeypointMatching result;
    result.source_keypoints.reserve(keypoint_count);
    result.target_keypoints.reserve(keypoint_count);

    for (std::size_t i = 0; i < keypoint_count; ++i) {
        Keypoint source;
        source.x = tensorElementToFloat(keypoint_data[i * 2]) * source_scale_x;
        source.y = tensorElementToFloat(keypoint_data[i * 2 + 1]) * source_scale_y;
        result.source_keypoints.push_back(source);

        Keypoint target;
        target.x = tensorElementToFloat(keypoint_data[(keypoint_count + i) * 2]) * target_scale_x;
        target.y = tensorElementToFloat(keypoint_data[(keypoint_count + i) * 2 + 1]) * target_scale_y;
        result.target_keypoints.push_back(target);
    }

    for (std::size_t row = 0; row < match_count; ++row) {
        const int pair_index = tensorElementToInt(match_data[row * 3]);
        if (pair_index != 0) {
            continue;
        }
        const int source_index = tensorElementToInt(match_data[row * 3 + 1]);
        const int target_index = tensorElementToInt(match_data[row * 3 + 2]);
        if (source_index < 0 || target_index < 0 || static_cast<std::size_t>(source_index) >= keypoint_count ||
            static_cast<std::size_t>(target_index) >= keypoint_count) {
            continue;
        }
        const float score = tensorElementToFloat(score_data[row]);
        if (!(score > score_threshold_)) {
            continue;
        }
        result.matches.emplace_back(source_index, target_index, score);
    }

    return result;
}

} // namespace neuriplo_tasks
