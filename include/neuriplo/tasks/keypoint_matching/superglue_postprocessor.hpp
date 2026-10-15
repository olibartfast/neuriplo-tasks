#pragma once

#include "neuriplo/tasks/keypoint_matching/keypoint_matching_postprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief SuperGlue-style keypoint matching postprocessor
 *
 * Only end-to-end exports (extractor and matcher in one graph) are usable
 * through this task. The upstream SuperGlue matcher alone takes keypoints,
 * scores, and descriptors as inputs and cannot be driven from images.
 *
 * Expected outputs, in ModelInfo::output_names order:
 * - keypoints0 [1, N, 2] and keypoints1 [1, M, 2] in the preprocessed input space
 * - matches0 [1, N]: matching keypoint index into keypoints1, or -1
 * - matching_scores0 [1, N]
 *
 * The layout is a placeholder: postprocess() throws std::logic_error until it
 * has been verified against a real ONNX export.
 */
class SuperGluePostprocessor : public KeypointMatchingPostprocessor {
  public:
    KeypointMatching postprocess(const std::vector<Tensor>& tensors, const Size& source_size,
                                 const Size& target_size) override;
};

} // namespace neuriplo_tasks
