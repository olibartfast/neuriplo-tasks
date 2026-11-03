#pragma once

#include "neuriplo/tasks/keypoint_matching/keypoint_matching_preprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief Pair preprocessor for end-to-end LightGlue pipeline exports
 *
 * These graphs take the pair interleaved on the batch axis as a single tensor
 * [2, C, H, W]. The verified SuperPoint-based export
 * (thomasonzhou/superpoint-lightglue) takes grayscale, C = 1; ALIKED/DISK
 * based pipelines take 3 channels, so the channel count follows the model's
 * input shape. Values are [0, 1] with no mean/std, matching the LightGlue-ONNX
 * pipeline preprocessing.
 */
class LightGluePreprocessor : public KeypointMatchingPreprocessor {
  public:
    explicit LightGluePreprocessor(const Size& input_size = Size(512, 512), int channels = 1);

    /**
     * @brief Pack the pair into one [2, C, H, W] float32 buffer
     *
     * @param source First image of the pair (batch index 0)
     * @param target Second image of the pair (batch index 1)
     * @return A single buffer, because the export has a single input tensor
     */
    [[nodiscard]] std::vector<std::vector<uint8_t>> preprocess_pair(const ImageView& source,
                                                                    const ImageView& target) const override;

    /// preprocess_pair() applies its own [0, 1] scaling.
    [[nodiscard]] bool supportsRawPixelOutput() const override { return false; }

  private:
    int channels_{1};
};

} // namespace neuriplo_tasks
