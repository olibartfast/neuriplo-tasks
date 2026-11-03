#pragma once

#include "neuriplo/tasks/keypoint_matching/keypoint_matching_postprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief LightGlue postprocessor
 *
 * Decodes end-to-end LightGlue pipeline exports (extractor and matcher in one
 * graph), verified against `thomasonzhou/superpoint-lightglue`:
 *
 * - `images [2, 1, H, W]` in (grayscale pair on the batch axis)
 * - `keypoints [2, K, 2]` int64 — pixel coordinates in the model input space,
 *   source at index 0 and target at index 1
 * - `matches [M, 3]` int64 — rows of `[pair_index, source_index, target_index]`
 * - `mscores [M]` float32 — match score
 *
 * Tensors are identified by shape. Only rows whose pair index is 0 are decoded
 * (the task processes one pair), and matches at or below `score_threshold` are
 * dropped. Keypoints are returned in each image's original size, so the
 * postprocessor must know the model input size the graph ran at.
 */
class LightGluePostprocessor : public KeypointMatchingPostprocessor {
  public:
    explicit LightGluePostprocessor(float score_threshold = 0.0f, Size input_size = Size(512, 512));

    KeypointMatching postprocess(const std::vector<Tensor>& tensors, const Size& source_size,
                                 const Size& target_size) override;

  private:
    float score_threshold_{0.0f};
    Size input_size_{512, 512};
};

} // namespace neuriplo_tasks
