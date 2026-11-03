#pragma once

#include "neuriplo/tasks/keypoint_matching/keypoint_matching_postprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief LightGlue postprocessor
 *
 * Targets the end-to-end LightGlue pipeline exports (extractor and matcher in
 * one graph), whose interleaved image-pair input is (2B, 1, H, W).
 *
 * Expected outputs, in ModelInfo::output_names order:
 * - keypoints [2, K, 2]: the pair's keypoints, source then target
 * - matches [M, 3]: rows of [batch_index, source_index, target_index]
 * - match_scores [M]: score per row of matches
 *
 * The layout is a placeholder: postprocess() throws std::logic_error until it
 * has been verified against a real ONNX export.
 */
class LightGluePostprocessor : public KeypointMatchingPostprocessor {
  public:
    KeypointMatching postprocess(const std::vector<Tensor>& tensors, const Size& source_size,
                                 const Size& target_size) override;
};

} // namespace neuriplo_tasks
