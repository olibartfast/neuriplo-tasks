#pragma once

#include "neuriplo/tasks/keypoint_matching/keypoint_matching_postprocessor.hpp"

namespace neuriplo_tasks {

/**
 * @brief MatchAnything postprocessor
 *
 * Targets detector-free matcher exports trained with the MatchAnything
 * framework (EfficientLoFTR / RoMa backbones). The ONNX output contract is not
 * yet pinned; postprocess() throws std::logic_error until a real export from
 * the chosen export path (Hugging Face Optimum or custom) has been probed.
 */
class MatchAnythingPostprocessor : public KeypointMatchingPostprocessor {
  public:
    KeypointMatching postprocess(const std::vector<Tensor>& tensors, const Size& source_size,
                                 const Size& target_size) override;
};

} // namespace neuriplo_tasks
