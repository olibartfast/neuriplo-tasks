#include "vision-core/classification/classifier_postprocessor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace vision_core {

namespace {

float get_float(const TensorElement& elem) {
    return std::visit([](auto&& arg) -> float {
        return static_cast<float>(arg);
    }, elem);
}

} // anonymous namespace

std::vector<float> ClassifierPostprocessor::apply_softmax(
    const std::vector<float>& logits) noexcept
{
    std::vector<float> probabilities(logits.size());
    
    // Find max for numerical stability
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    
    // Compute exp(logit - max) and sum
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum += probabilities[i];
    }
    
    // Normalize
    for (float& prob : probabilities) {
        prob /= sum;
    }
    
    return probabilities;
}

std::vector<ClassificationResult> ClassifierPostprocessor::postprocess(
    const TensorElement* output,
    const std::vector<int64_t>& shape,
    int top_k,
    bool apply_softmax_flag)
{
    if (shape.empty()) {
        throw std::invalid_argument("Output shape cannot be empty");
    }

    const int num_classes = shape.back();
    
    // Convert to float vector
    std::vector<float> scores(num_classes);
    for (int i = 0; i < num_classes; ++i) {
        scores[i] = get_float(output[i]);
    }

    // Apply softmax if requested
    if (apply_softmax_flag) {
        scores = apply_softmax(scores);
    }

    // Create pairs of (class_id, score)
    std::vector<std::pair<int, float>> class_scores;
    class_scores.reserve(num_classes);
    
    for (int i = 0; i < num_classes; ++i) {
        class_scores.emplace_back(i, scores[i]);
    }

    // Sort by score descending
    std::partial_sort(
        class_scores.begin(),
        class_scores.begin() + std::min(top_k, num_classes),
        class_scores.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; }
    );

    // Build results
    std::vector<ClassificationResult> results;
    results.reserve(std::min(top_k, num_classes));

    for (int i = 0; i < std::min(top_k, num_classes); ++i) {
        results.emplace_back(
            class_scores[i].first,   // class_id
            class_scores[i].second   // confidence
        );
    }

    return results;
}

} // namespace vision_core
