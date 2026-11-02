#pragma once

#include "neuriplo/tasks/keypoint_matching/keypoint_matching_postprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief SuperGlue postprocessor
 *
 * Decodes the transformers `SuperGlueForKeypointMatching` ONNX export (for
 * example `magic-leap-community/superglue_outdoor` exported with Optimum). The
 * graph takes the pair as one tensor and returns four outputs, `N` being
 * dynamic and equal for every output:
 *
 * - `keypoints` [1, 2, N, 2] float32 — normalized coordinates: x / width,
 *   y / height, for the source (index 0) and target (index 1) images.
 * - `matches` [1, 2, N] int64 — for index 0, the target keypoint index this
 *   source keypoint matches, or -1 when unmatched. Index 1 is the inverse map.
 * - `matching_scores` [1, 2, N] float32 — mutual-match probability.
 * - `mask` [1, 2, N] int64 — 1 for a detected keypoint, 0 for padding.
 *
 * Tensors are identified by shape, not by position, so an engine that reports
 * them in a different order still decodes. Keypoints are returned in the
 * original image space of each image of the pair, and match indices point into
 * `KeypointMatching::source_keypoints` / `target_keypoints` (masked-out padding
 * keypoints are dropped). Matches are kept only when the score is above
 * `score_threshold`, which the task fills from
 * `TaskConfig::confidence_threshold`.
 *
 * The export does not carry the SuperPoint detector score, so
 * `Keypoint::confidence` stays at its default value for these keypoints; use
 * `KeypointMatch::score` for match quality.
 *
 * Matcher-only exports (keypoints, scores, and descriptors in — the original
 * SuperGlue matcher) cannot be driven from images and are not supported here.
 */
class SuperGluePostprocessor : public KeypointMatchingPostprocessor {
  public:
    explicit SuperGluePostprocessor(float score_threshold = 0.0f);

    KeypointMatching postprocess(const std::vector<Tensor>& tensors, const Size& source_size,
                                 const Size& target_size) override;

  private:
    float score_threshold_{0.0f};
};

} // namespace neuriplo_tasks
