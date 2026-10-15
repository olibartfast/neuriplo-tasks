#pragma once

#include "neuriplo/tasks/core/preprocessor.hpp"

#include <vector>

namespace neuriplo_tasks {

/**
 * @brief Pair preprocessor for keypoint matching models
 *
 * Resizes each image of the pair to the model input size, applies the shared
 * configuration (BGR to RGB, [0, 1] scaling) and emits one NCHW float32 buffer
 * per image. It is shared by the SuperGlue, LightGlue, and MatchAnything
 * families until a real export shows their normalization or channel handling
 * differs; the class is split per family at that point.
 */
class KeypointMatchingPreprocessor : public Preprocessor {
  public:
    explicit KeypointMatchingPreprocessor(const Size& input_size = Size(512, 512));

    /**
     * @brief Preprocess an image pair
     *
     * @param image1 First image of the pair (source)
     * @param image2 Second image of the pair (target)
     * @return Two preprocessed buffers in call order
     */
    [[nodiscard]] std::vector<std::vector<uint8_t>> preprocess_pair(const ImageView& image1,
                                                                    const ImageView& image2) const;
};

} // namespace neuriplo_tasks
