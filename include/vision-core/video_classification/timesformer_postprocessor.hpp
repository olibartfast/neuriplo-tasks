#pragma once

#include <cstdint>
#include <variant>
#include <vector>

namespace vision_core {

using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

/**
 * @brief Video classification result
 */
struct VideoClassificationResult {
    int class_id;
    float confidence;
    
    VideoClassificationResult(int cid, float conf)
        : class_id(cid), confidence(conf) {}
};

/**
 * @brief TimeSformer video classification postprocessing
 * 
 * Handles temporal transformer models for video classification.
 * Applies softmax to logits and returns top-k predictions.
 */
class TimesformerPostprocessor {
public:
    /**
     * @brief Process TimeSformer output tensor
     * 
     * @param output Raw output logits from inference engine
     * @param shape Output tensor shape [batch, num_classes]
     * @param top_k Number of top predictions to return
     * @param apply_softmax Whether to apply softmax to logits
     * @return Vector of top-k classification results
     */
    [[nodiscard]] static std::vector<VideoClassificationResult> postprocess(
        const TensorElement* output,
        const std::vector<int64_t>& shape,
        int top_k = 5,
        bool apply_softmax = true
    );

private:
    /**
     * @brief Apply softmax activation to logits
     * 
     * @param logits Input logits
     * @return Probabilities after softmax
     */
    static std::vector<float> softmax(const std::vector<float>& logits);
};

} // namespace vision_core
