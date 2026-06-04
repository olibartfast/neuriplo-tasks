#include "vision-core/classification/classification_postprocessor.hpp"

#include "vision-core/classification/classification_logits_layout.hpp"
#include "vision-core/core/tensor_utils.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

namespace vision_core {

namespace {

void applySoftmaxInPlace(std::vector<float>& logits) {
    if (logits.empty()) {
        return;
    }

    const float max_val = *std::max_element(logits.begin(), logits.end());
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

std::vector<Classification> topKFromScores(const std::vector<float>& scores, int top_k) {
    std::vector<std::pair<int, float>> indexed_scores;
    indexed_scores.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        indexed_scores.emplace_back(static_cast<int>(i), scores[i]);
    }

    if (indexed_scores.empty()) {
        return {};
    }

    const int limit = std::min(top_k, static_cast<int>(indexed_scores.size()));
    std::partial_sort(indexed_scores.begin(), indexed_scores.begin() + limit, indexed_scores.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<Classification> classifications;
    classifications.reserve(static_cast<size_t>(limit));
    for (int i = 0; i < limit; ++i) {
        classifications.push_back(Classification{static_cast<float>(indexed_scores[static_cast<size_t>(i)].first),
                                                 indexed_scores[static_cast<size_t>(i)].second});
    }
    return classifications;
}

std::vector<float> sliceScores(const std::vector<TensorElement>& output, size_t offset, int num_classes) {
    std::vector<float> scores;
    scores.reserve(static_cast<size_t>(num_classes));
    for (int i = 0; i < num_classes; ++i) {
        scores.push_back(tensorElementToFloat(output[offset + static_cast<size_t>(i)]));
    }
    return scores;
}

} // namespace

std::vector<Classification> postprocessClassificationLogits(const std::vector<TensorElement>& output,
                                                            const std::vector<int64_t>& shape, int top_k,
                                                            bool apply_softmax) {
    if (output.empty() || shape.empty() || top_k <= 0) {
        return {};
    }

    const classification_detail::LogitBatchLayout layout =
        classification_detail::parseLogitBatchLayout(shape, output.size());
    if (layout.batch_size <= 0 || layout.num_classes <= 0) {
        return {};
    }

    const int per_batch_top_k = layout.batch_size > 1 ? 1 : top_k;
    std::vector<Classification> classifications;
    classifications.reserve(static_cast<size_t>(layout.batch_size * per_batch_top_k));

    for (int batch_index = 0; batch_index < layout.batch_size; ++batch_index) {
        const size_t offset = classification_detail::batchSliceOffset(batch_index, layout);
        std::vector<float> scores = sliceScores(output, offset, layout.num_classes);

        if (apply_softmax) {
            applySoftmaxInPlace(scores);
        }

        auto batch_top = topKFromScores(scores, per_batch_top_k);
        classifications.insert(classifications.end(), batch_top.begin(), batch_top.end());
    }

    return classifications;
}

DefaultClassificationPostprocessor::DefaultClassificationPostprocessor(int top_k, bool apply_softmax)
    : top_k_(top_k), apply_softmax_(apply_softmax) {}

std::vector<Classification> DefaultClassificationPostprocessor::postprocess(const std::vector<TensorElement>& output,
                                                                            const std::vector<int64_t>& shape) {
    return postprocessClassificationLogits(output, shape, top_k_, apply_softmax_);
}

void DefaultClassificationPostprocessor::applySoftmax(std::vector<float>& logits) { applySoftmaxInPlace(logits); }

} // namespace vision_core
