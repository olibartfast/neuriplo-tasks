#include "neuriplo/tasks/keypoint_matching/flexglue_postprocessor.hpp"

#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace neuriplo_tasks {

namespace {

struct FlexGlueOutputs {
    const Tensor* keypoints0{nullptr};
    const Tensor* keypoints1{nullptr};
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
 * @brief Group the matcher's tensors by shape and element type
 *
 * kpts0/1 [1, N, 2] float (positional: first forwarded is kpts0),
 * matches0 [1, N0] int, mscores0 [1, N0] float. The optional
 * covisibility_score [1] float and unrelated tensors are ignored.
 */
FlexGlueOutputs classifyOutputs(const std::vector<Tensor>& tensors) {
    FlexGlueOutputs outputs;
    for (const Tensor& tensor : tensors) {
        const std::vector<int64_t>& shape = tensor.shape;
        if (shape.size() == 3 && shape[0] == 1 && shape[2] == 2 && !holdsInteger(tensor)) {
            if (outputs.keypoints0 == nullptr) {
                outputs.keypoints0 = &tensor;
            } else if (outputs.keypoints1 == nullptr) {
                outputs.keypoints1 = &tensor;
            }
            continue;
        }
        if (shape.size() == 2 && shape[0] == 1) {
            if (holdsInteger(tensor)) {
                if (outputs.matches == nullptr) {
                    outputs.matches = &tensor;
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
        throw std::invalid_argument(std::string("FlexGlue '") + name + "' output expects " +
                                    std::to_string(expected_elements) + " elements but holds " +
                                    std::to_string(tensor.data.size()));
    }
}

} // namespace

FlexGluePostprocessor::FlexGluePostprocessor(float score_threshold) : score_threshold_(score_threshold) {}

KeypointMatching FlexGluePostprocessor::postprocess(const std::vector<Tensor>& tensors, const Size& source_size,
                                                    const Size& target_size) {
    (void)source_size;
    (void)target_size;
    if (tensors.empty()) {
        return {};
    }

    const FlexGlueOutputs outputs = classifyOutputs(tensors);
    if (outputs.keypoints0 == nullptr || outputs.keypoints1 == nullptr) {
        throw std::invalid_argument("FlexGlue: two forwarded [1, N, 2] float keypoint tensors (kpts0, kpts1) required");
    }
    if (outputs.matches == nullptr) {
        throw std::invalid_argument("FlexGlue: no [1, N0] integer matches0 output found");
    }
    if (outputs.scores == nullptr) {
        throw std::invalid_argument("FlexGlue: no [1, N0] float mscores0 output found");
    }

    const std::size_t source_count = static_cast<std::size_t>(outputs.keypoints0->shape[1]);
    const std::size_t target_count = static_cast<std::size_t>(outputs.keypoints1->shape[1]);
    const std::size_t match_count = static_cast<std::size_t>(outputs.matches->shape[1]);
    requireShape(*outputs.keypoints0, source_count * 2, "kpts0");
    requireShape(*outputs.keypoints1, target_count * 2, "kpts1");
    requireShape(*outputs.matches, match_count, "matches0");
    requireShape(*outputs.scores, match_count, "mscores0");
    if (match_count != source_count) {
        throw std::invalid_argument("FlexGlue: matches0 length (" + std::to_string(match_count) +
                                    ") must equal the kpts0 count (" + std::to_string(source_count) + ")");
    }

    const std::vector<TensorElement>& source_data = outputs.keypoints0->data;
    const std::vector<TensorElement>& target_data = outputs.keypoints1->data;
    const std::vector<TensorElement>& match_data = outputs.matches->data;
    const std::vector<TensorElement>& score_data = outputs.scores->data;

    // Keypoints arrive in original-image corner pixels, so they pass through
    // unscaled; only finiteness is validated here.
    KeypointMatching result;
    result.source_keypoints.reserve(source_count);
    result.target_keypoints.reserve(target_count);

    for (std::size_t i = 0; i < source_count; ++i) {
        Keypoint keypoint;
        keypoint.x = tensorElementToFloat(source_data[i * 2]);
        keypoint.y = tensorElementToFloat(source_data[i * 2 + 1]);
        if (!std::isfinite(keypoint.x) || !std::isfinite(keypoint.y)) {
            throw std::invalid_argument("FlexGlue: non-finite coordinate in kpts0");
        }
        result.source_keypoints.push_back(keypoint);
    }
    for (std::size_t i = 0; i < target_count; ++i) {
        Keypoint keypoint;
        keypoint.x = tensorElementToFloat(target_data[i * 2]);
        keypoint.y = tensorElementToFloat(target_data[i * 2 + 1]);
        if (!std::isfinite(keypoint.x) || !std::isfinite(keypoint.y)) {
            throw std::invalid_argument("FlexGlue: non-finite coordinate in kpts1");
        }
        result.target_keypoints.push_back(keypoint);
    }

    for (std::size_t i = 0; i < match_count; ++i) {
        const int matched = tensorElementToInt(match_data[i]);
        if (matched < 0 || static_cast<std::size_t>(matched) >= target_count) {
            continue;
        }
        const float score = tensorElementToFloat(score_data[i]);
        if (!(score > score_threshold_)) {
            continue;
        }
        result.matches.emplace_back(static_cast<int>(i), matched, score);
    }

    return result;
}

} // namespace neuriplo_tasks
