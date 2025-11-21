#include "vision-core/video_classification/timesformer_postprocessor.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace vision_core {

namespace {

float get_float(const TensorElement& elem) {
    return std::visit([](auto&& arg) -> float { 
        return static_cast<float>(arg); 
    }, elem);
}

} // anonymous namespace

std::vector<float> TimesformerPostprocessor::softmax(const std::vector<float>& logits) {
    std::vector<float> probabilities(logits.size());
    
    // Find max for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum += probabilities[i];
    }
    
    // Normalize
    for (auto& prob : probabilities) {
        prob /= sum;
    }
    
    return probabilities;
}

std::vector<VideoClassificationResult> TimesformerPostprocessor::postprocess(
    const TensorElement* output,
    const std::vector<int64_t>& shape,
    int top_k,
    bool apply_softmax)
{
    if (shape.empty()) {
        throw std::invalid_argument("Output shape is empty");
    }

    const int64_t num_classes = shape.back();
    
    if (num_classes <= 0) {
        throw std::invalid_argument("Invalid number of classes");
    }

    // Convert to float vector
    std::vector<float> scores;
    scores.reserve(num_classes);
    
    for (int64_t i = 0; i < num_classes; ++i) {
        scores.push_back(get_float(output[i]));
    }

    // Apply softmax if requested
    if (apply_softmax) {
        scores = softmax(scores);
    }

    // Create indexed pairs for sorting
    std::vector<std::pair<int, float>> indexed_scores;
    indexed_scores.reserve(num_classes);
    
    for (int i = 0; i < static_cast<int>(num_classes); ++i) {
        indexed_scores.emplace_back(i, scores[i]);
    }

    // Sort by score descending
    std::partial_sort(
        indexed_scores.begin(),
        indexed_scores.begin() + std::min(top_k, static_cast<int>(num_classes)),
        indexed_scores.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; }
    );

    // Build results
    std::vector<VideoClassificationResult> results;
    results.reserve(std::min(top_k, static_cast<int>(num_classes)));
    
    for (int i = 0; i < std::min(top_k, static_cast<int>(num_classes)); ++i) {
        results.emplace_back(indexed_scores[i].first, indexed_scores[i].second);
    }

    return results;
}

} // namespace vision_core
