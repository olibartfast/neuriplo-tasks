#include "vision-core/core/task_interface.hpp"
#include <iostream>

namespace vision_core {

TaskInterface::TaskInterface(const ModelInfo& model_info)
    : model_info_(model_info) {
    std::cout << "[TaskInterface DEBUG] Constructor called" << std::endl << std::flush;
    std::tie(input_width_, input_height_, input_channels_) = initializeInputDimensions(model_info_);
    std::cout << "[TaskInterface DEBUG] Dimensions: " << input_width_ << "x" << input_height_ << "x" << input_channels_ << std::endl << std::flush;
    
    if (input_width_ <= 0 || input_height_ <= 0 || input_channels_ <= 0) {
        throw InputDimensionError("Invalid input dimensions");
    }
}

} // namespace vision_core
