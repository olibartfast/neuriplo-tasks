#pragma once

#include "neuriplo/tasks/keypoint_matching/keypoint_matching_preprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief Pair preprocessor for SuperGlue end-to-end exports
 *
 * SuperGlue exports from transformers (SuperGlueForKeypointMatching, e.g.
 * magic-leap-community/superglue_outdoor exported with Optimum) take the image
 * pair as one rank-5 input, pixel_values [1, 2, 3, H, W], and expect grayscale
 * pixels replicated across the three channels in [0, 1]. preprocess_pair()
 * therefore returns a single packed buffer holding both images, in call order:
 * source at pair index 0, target at pair index 1.
 *
 * The transformers image processor resizes to 640x480 by default and applies no
 * mean/std normalization, which is what this preprocessor emits.
 */
class SuperGluePreprocessor : public KeypointMatchingPreprocessor {
  public:
    explicit SuperGluePreprocessor(const Size& input_size = Size(640, 480));

    /**
     * @brief Pack the pair into one [2, 3, H, W] float32 buffer
     *
     * @param source First image of the pair (source)
     * @param target Second image of the pair (target)
     * @return A single buffer, because the export has a single input tensor
     */
    [[nodiscard]] std::vector<std::vector<uint8_t>> preprocess_pair(const ImageView& source,
                                                                    const ImageView& target) const override;

    /// preprocess_pair() applies its own grayscale conversion and [0, 1] scaling.
    [[nodiscard]] bool supportsRawPixelOutput() const override { return false; }
};

} // namespace neuriplo_tasks
