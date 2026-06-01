#include "vision-core/classification/torchvision_postprocessor.hpp"

#include "vision-core/core/tensor_utils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vision_core {

TorchvisionPostprocessor::TorchvisionPostprocessor(int top_k, bool apply_softmax)
    : top_k_(top_k), apply_softmax_(apply_softmax) {}

std::vector<Classification> TorchvisionPostprocessor::postprocess(const std::vector<TensorElement>& output,
                                                                  const std::vector<int64_t>& shape) {

    if (output.empty() || shape.empty()) {
        return {};
    }

    // Torchvision models typically output [1, 1000] (ImageNet classes)
    // Logic is similar to default but we can add specific checks or handling here

    std::vector<float> scores;
    scores.reserve(output.size());
    for (const auto& element : output) {
        scores.push_back(tensorElementToFloat(element));
    }

    if (apply_softmax_) {
        applySoftmax(scores);
    }

    std::vector<std::pair<int, float>> indexed_scores;
    for (size_t i = 0; i < scores.size(); ++i) {
        indexed_scores.emplace_back(static_cast<int>(i), scores[i]);
    }

    std::partial_sort(indexed_scores.begin(),
                      indexed_scores.begin() + std::min(top_k_, static_cast<int>(indexed_scores.size())),
                      indexed_scores.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<Classification> classifications;
    int limit = std::min(top_k_, static_cast<int>(indexed_scores.size()));
    for (size_t i = 0; i < static_cast<size_t>(limit); ++i) {
        Classification cls;
        cls.class_id = static_cast<float>(indexed_scores[i].first);
        cls.class_confidence = indexed_scores[i].second;
        classifications.push_back(cls);
    }

    return classifications;
}

void TorchvisionPostprocessor::applySoftmax(std::vector<float>& logits) {
    if (logits.empty()) {
        return;
    }
    float max_val = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (auto& logit : logits) {
        logit = std::exp(logit - max_val);
        sum += logit;
    }
    if (sum > 0.0f) {
        for (auto& logit : logits) {
            logit /= sum;
        }
    }
}

} // namespace vision_core
