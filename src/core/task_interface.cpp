#include "vision-core/core/task_interface.hpp"

namespace vision_core {

TaskInterface::TaskInterface(const ModelInfo& model_info) : model_info_(model_info) {
    std::tie(input_width_, input_height_, input_channels_) = initializeInputDimensions(model_info_);
    // Zero dimensions are valid for non-image tasks (e.g. text generation).
    // Tasks that require spatial dimensions perform their own validation.
}

} // namespace vision_core
