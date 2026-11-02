#pragma once

#include "neuriplo/tasks/core/preprocessor.hpp"

#include <vector>

namespace neuriplo_tasks {

/**
 * @brief Pair preprocessor for keypoint matching models
 *
 * Resizes each image of the pair to the model input size, applies the shared
 * configuration (BGR to RGB, [0, 1] scaling) and emits one NCHW float32 buffer
 * per model input. Families whose graph takes two image inputs get one buffer
 * per image (the LightGlue-ONNX pipeline, MatchAnything exports); families whose
 * graph takes the pair as one tensor override preprocess_pair() and emit a
 * single packed buffer (SuperGlue, see SuperGluePreprocessor).
 */
class KeypointMatchingPreprocessor : public Preprocessor {
  public:
    explicit KeypointMatchingPreprocessor(const Size& input_size = Size(512, 512));

    /**
     * @brief Preprocess an image pair
     *
     * @param image1 First image of the pair (source)
     * @param image2 Second image of the pair (target)
     * @return One buffer per model input, in the order the model expects them
     */
    [[nodiscard]] virtual std::vector<std::vector<uint8_t>> preprocess_pair(const ImageView& image1,
                                                                            const ImageView& image2) const;
};

} // namespace neuriplo_tasks
