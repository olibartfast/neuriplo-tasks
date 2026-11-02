#include "neuriplo/tasks/keypoint_matching/superglue_postprocessor.hpp"

#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace neuriplo_tasks {

namespace {

struct SuperGlueOutputs {
    const Tensor* keypoints{nullptr};
    const Tensor* matches{nullptr};
    const Tensor* scores{nullptr};
    const Tensor* mask{nullptr};
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
 * The transformers SuperGlue export returns keypoints [1, 2, N, 2], matches
 * [1, 2, N] (int), matching_scores [1, 2, N] (float), and mask [1, 2, N] (int).
 * Shapes identify them so a different output order still decodes; unrelated
 * tensors are ignored.
 */
SuperGlueOutputs classifyOutputs(const std::vector<Tensor>& tensors) {
    SuperGlueOutputs outputs;
    for (const Tensor& tensor : tensors) {
        const std::vector<int64_t>& shape = tensor.shape;
        if (shape.size() == 4 && shape[0] == 1 && shape[1] == 2 && shape[3] == 2) {
            if (outputs.keypoints == nullptr) {
                outputs.keypoints = &tensor;
            }
            continue;
        }
        if (shape.size() == 3 && shape[0] == 1 && shape[1] == 2) {
            if (holdsInteger(tensor)) {
                if (outputs.matches == nullptr) {
                    outputs.matches = &tensor;
                } else if (outputs.mask == nullptr) {
                    outputs.mask = &tensor;
                }
            } else if (outputs.scores == nullptr) {
                outputs.scores = &tensor;
            }
        }
    }
    return outputs;
}

void requireShape(const Tensor& tensor, std::size_t expected_elements, const char* name) {
    if (elementCount(tensor) != tensor.data.size() || tensor.data.size() != expected_elements) {
        throw std::invalid_argument(std::string("SuperGlue '") + name + "' output expects " +
                                    std::to_string(expected_elements) + " elements but holds " +
                                    std::to_string(tensor.data.size()));
    }
}

} // namespace

SuperGluePostprocessor::SuperGluePostprocessor(float score_threshold) : score_threshold_(score_threshold) {}

KeypointMatching SuperGluePostprocessor::postprocess(const std::vector<Tensor>& tensors, const Size& source_size,
                                                     const Size& target_size) {
    if (tensors.empty()) {
        return {};
    }

    const SuperGlueOutputs outputs = classifyOutputs(tensors);
    if (outputs.keypoints == nullptr) {
        throw std::invalid_argument("SuperGlue: no [1, 2, N, 2] keypoints output found");
    }
    if (outputs.matches == nullptr) {
        throw std::invalid_argument("SuperGlue: no [1, 2, N] integer matches output found");
    }
    if (outputs.scores == nullptr) {
        throw std::invalid_argument("SuperGlue: no [1, 2, N] float matching_scores output found");
    }

    const std::size_t count = static_cast<std::size_t>(outputs.keypoints->shape[2]);
    requireShape(*outputs.keypoints, count * 4, "keypoints");
    requireShape(*outputs.matches, count * 2, "matches");
    requireShape(*outputs.scores, count * 2, "matching_scores");
    if (outputs.mask != nullptr) {
        requireShape(*outputs.mask, count * 2, "mask");
    }

    const std::vector<TensorElement>& keypoint_data = outputs.keypoints->data;
    const std::vector<TensorElement>& match_data = outputs.matches->data;
    const std::vector<TensorElement>& score_data = outputs.scores->data;
    const std::vector<TensorElement>* mask_data = outputs.mask == nullptr ? nullptr : &outputs.mask->data;

    // Matches index the padded keypoint arrays; padding is masked out, so the
    // emitted vectors hold the valid keypoints only and matches are remapped.
    std::vector<int> source_index(count, -1);
    std::vector<int> target_index(count, -1);

    KeypointMatching result;
    result.source_keypoints.reserve(count);
    result.target_keypoints.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        if (mask_data != nullptr && tensorElementToInt((*mask_data)[i]) <= 0) {
            continue;
        }
        Keypoint keypoint;
        keypoint.x = tensorElementToFloat(keypoint_data[i * 2]) * static_cast<float>(source_size.width);
        keypoint.y = tensorElementToFloat(keypoint_data[i * 2 + 1]) * static_cast<float>(source_size.height);
        source_index[i] = static_cast<int>(result.source_keypoints.size());
        result.source_keypoints.push_back(keypoint);
    }

    for (std::size_t i = 0; i < count; ++i) {
        if (mask_data != nullptr && tensorElementToInt((*mask_data)[count + i]) <= 0) {
            continue;
        }
        Keypoint keypoint;
        keypoint.x = tensorElementToFloat(keypoint_data[(count + i) * 2]) * static_cast<float>(target_size.width);
        keypoint.y = tensorElementToFloat(keypoint_data[(count + i) * 2 + 1]) * static_cast<float>(target_size.height);
        target_index[i] = static_cast<int>(result.target_keypoints.size());
        result.target_keypoints.push_back(keypoint);
    }

    for (std::size_t i = 0; i < count; ++i) {
        if (source_index[i] < 0) {
            continue;
        }
        const int matched = tensorElementToInt(match_data[i]);
        if (matched < 0 || static_cast<std::size_t>(matched) >= count ||
            target_index[static_cast<std::size_t>(matched)] < 0) {
            continue;
        }
        const float score = tensorElementToFloat(score_data[i]);
        if (!(score > score_threshold_)) {
            continue;
        }
        result.matches.emplace_back(source_index[i], target_index[static_cast<std::size_t>(matched)], score);
    }

    return result;
}

} // namespace neuriplo_tasks
