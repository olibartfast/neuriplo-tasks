#include "vision-core/classification/vit_postprocessor.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vision_core {

ViTPostprocessor::ViTPostprocessor(int top_k, bool apply_softmax)
    : top_k_(top_k)
    , apply_softmax_(apply_softmax) {}

std::vector<Classification> ViTPostprocessor::postprocess(
    const std::vector<TensorElement>& output,
    const std::vector<int64_t>& shape) {
    
    if (output.empty() || shape.empty()) {
        return {};
    }
    
    // ViT models might have different output shapes, e.g., [1, cls_token + patches, dim]
    // But for classification head, it usually boils down to [1, num_classes]
    // We can add specific checks here if needed
    
    std::vector<float> scores;
    scores.reserve(output.size());
    for (const auto& element : output) {
        scores.push_back(getTensorFloat(element));
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
                     indexed_scores.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::vector<Classification> classifications;
    int limit = std::min(top_k_, static_cast<int>(indexed_scores.size()));
    for (int i = 0; i < limit; ++i) {
        Classification cls;
        cls.class_id = indexed_scores[i].first;
        cls.class_confidence = indexed_scores[i].second;
        classifications.push_back(cls);
    }
    
    return classifications;
}

float ViTPostprocessor::getTensorFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float {
        return static_cast<float>(value);
    }, element);
}

void ViTPostprocessor::applySoftmax(std::vector<float>& logits) {
    if (logits.empty()) return;
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
