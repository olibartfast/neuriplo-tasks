#include "neuriplo/tasks/keypoint_matching/superglue_postprocessor.hpp"

#include <stdexcept>

namespace neuriplo_tasks {

KeypointMatching SuperGluePostprocessor::postprocess(const std::vector<Tensor>& /*tensors*/,
                                                     const Size& /*source_size*/, const Size& /*target_size*/) {
    throw std::logic_error("SuperGlue decoding is not implemented yet: the output tensor layout has not been "
                           "verified against a real ONNX export (see specs/2026-10-15-keypoint-matching)");
}

} // namespace neuriplo_tasks
