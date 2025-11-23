#include "vision-core/core/task_interface.hpp"

namespace vision_core {

TaskInterface::TaskInterface(const ModelInfo& model_info)
    : model_info_(model_info) {
    std::tie(input_width_, input_height_, input_channels_) = initializeInputDimensions(model_info_);
    
    if (input_width_ <= 0 || input_height_ <= 0 || input_channels_ <= 0) {
        throw InputDimensionError("Invalid input dimensions");
    }
}

} // namespace vision_core
