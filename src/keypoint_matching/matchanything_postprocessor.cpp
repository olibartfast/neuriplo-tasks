#include "neuriplo/tasks/keypoint_matching/matchanything_postprocessor.hpp"

#include <stdexcept>

namespace neuriplo_tasks {

KeypointMatching MatchAnythingPostprocessor::postprocess(const std::vector<Tensor>& /*tensors*/,
                                                         const Size& /*source_size*/, const Size& /*target_size*/) {
    throw std::logic_error("MatchAnything decoding is not implemented yet: the ONNX output contract has not been "
                           "pinned down (see specs/2026-10-15-keypoint-matching)");
}

} // namespace neuriplo_tasks
