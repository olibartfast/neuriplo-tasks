#pragma once

#include "neuriplo/tasks/keypoint_matching/keypoint_matching_preprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief Pair preprocessor for FlexGlue descriptor exports
 *
 * The FlexGlue ONNX flow is two models: `flexglue_descriptor.onnx` runs once
 * per image (`image [1, 3, S, S]`, RGB in [0, 1], squashed with a bilinear
 * resize and no mean/std), and the sparse matcher consumes the two feature
 * maps plus externally detected keypoints. preprocess_pair() therefore
 * returns two buffers — one NCHW float32 RGB buffer per image, source first —
 * each mapping 1:1 onto a descriptor run, not a packed pair.
 */
class FlexGluePreprocessor : public KeypointMatchingPreprocessor {
  public:
    explicit FlexGluePreprocessor(const Size& input_size = Size(512, 512));

    /**
     * @brief Emit one RGB [0, 1] buffer per image for the descriptor
     *
     * @param source First image of the pair (source)
     * @param target Second image of the pair (target)
     * @return Two buffers, source first, each [1, 3, H, W] float32 planar
     */
    [[nodiscard]] std::vector<std::vector<uint8_t>> preprocess_pair(const ImageView& source,
                                                                    const ImageView& target) const override;

    /// preprocess_pair() applies its own [0, 1] scaling.
    [[nodiscard]] bool supportsRawPixelOutput() const override { return false; }
};

} // namespace neuriplo_tasks
