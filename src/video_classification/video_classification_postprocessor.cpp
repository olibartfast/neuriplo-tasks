#include "vision-core/video_classification/video_classification_postprocessor.hpp"

#include <algorithm>
#include <cmath>

namespace vision_core {

VideoClassificationPostprocessor::VideoClassificationPostprocessor(int top_k, bool apply_softmax)
    : top_k_(top_k), apply_softmax_(apply_softmax) {}

std::vector<VideoClassification> VideoClassificationPostprocessor::postprocess(const std::vector<TensorElement>& output,
                                                                               const std::vector<int64_t>& shape) {

    if (output.empty() || shape.empty()) {
        return {};
    }

    // Extract logits/probabilities
    std::vector<float> scores;
    scores.reserve(output.size());
    for (const auto& element : output) {
        scores.push_back(getTensorFloat(element));
    }

    // Apply softmax if needed
    if (apply_softmax_) {
        applySoftmax(scores);
    }

    // Get top-k classifications
    std::vector<std::pair<int, float>> indexed_scores;
    indexed_scores.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        indexed_scores.emplace_back(static_cast<int>(i), scores[i]);
    }

    if (indexed_scores.empty()) {
        return {};
    }

    // Sort by score (highest first)
    int k = std::min(top_k_, static_cast<int>(indexed_scores.size()));
    std::partial_sort(indexed_scores.begin(), indexed_scores.begin() + k, indexed_scores.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

    // Create VideoClassification results
    std::vector<VideoClassification> classifications;
    classifications.reserve(static_cast<size_t>(k));
    for (size_t i = 0; i < static_cast<size_t>(k); ++i) {
        VideoClassification cls;
        cls.class_id = static_cast<float>(indexed_scores[i].first);
        cls.class_confidence = indexed_scores[i].second;
        classifications.push_back(cls);
    }

    return classifications;
}

float VideoClassificationPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float { return static_cast<float>(value); }, element);
}

void VideoClassificationPostprocessor::applySoftmax(std::vector<float>& logits) {
    if (logits.empty())
        return;

    // Find max value for numerical stability
    float max_val = *std::max_element(logits.begin(), logits.end());

    // Compute softmax
    float sum = 0.0f;
    for (auto& logit : logits) {
        logit = std::exp(logit - max_val);
        sum += logit;
    }

    // Normalize
    if (sum > 0.0f) {
        for (auto& logit : logits) {
            logit /= sum;
        }
    }
}

} // namespace vision_core
