#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cstdint>

namespace vision_core {

/**
 * @brief Model information structure
 * 
 * Contains metadata about a model's inputs and outputs.
 * Compatible with Triton model info but framework-agnostic.
 */
struct ModelInfo {
    std::vector<std::vector<int64_t>> input_shapes;   ///< Input tensor shapes
    std::vector<std::string> input_formats;           ///< Input formats (FORMAT_NCHW, FORMAT_NHWC)
    std::vector<std::string> input_names;             ///< Input tensor names
    std::vector<std::string> output_names;            ///< Output tensor names
    std::vector<int> input_types;                     ///< OpenCV data types
    int max_batch_size_{1};                           ///< Maximum batch size
    int batch_size_{1};                               ///< Current batch size
    
    ModelInfo() = default;
};

} // namespace vision_core
