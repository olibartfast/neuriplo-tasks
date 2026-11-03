#include "neuriplo/tasks/keypoint_matching/lightglue_postprocessor.hpp"

#include <stdexcept>

namespace neuriplo_tasks {

KeypointMatching LightGluePostprocessor::postprocess(const std::vector<Tensor>& /*tensors*/,
                                                     const Size& /*source_size*/, const Size& /*target_size*/) {
    throw std::logic_error("LightGlue decoding is not implemented yet: the output tensor layout has not been "
                           "verified against a real ONNX export (see specs/2026-10-15-keypoint-matching)");
}

} // namespace neuriplo_tasks
