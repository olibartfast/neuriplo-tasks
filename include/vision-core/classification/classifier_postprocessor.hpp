#pragma once

#include "vision-core/core/result_types.hpp"
#include <cstdint>
#include <variant>
#include <vector>

namespace vision_core {

/// Generic tensor element type
using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

/**
 * @brief Generic classification postprocessor
 * 
 * Processes classification model outputs (logits or probabilities)
 * and returns top-k predictions with optional softmax application.
 */
class ClassifierPostprocessor {
public:
    /**
     * @brief Process classification output
     * 
     * @param output Raw output logits or probabilities
     * @param shape Output tensor shape [batch, num_classes]
     * @param top_k Number of top predictions to return
     * @param apply_softmax Whether to apply softmax to logits
     * @return Vector of classification results sorted by confidence
     */
    [[nodiscard]] static std::vector<ClassificationResult> postprocess(
        const TensorElement* output,
        const std::vector<int64_t>& shape,
        int top_k = 5,
        bool apply_softmax = true
    );

private:
    /**
     * @brief Apply softmax activation to logits
     */
    [[nodiscard]] static std::vector<float> apply_softmax(
        const std::vector<float>& logits
    ) noexcept;
};

} // namespace vision_core
