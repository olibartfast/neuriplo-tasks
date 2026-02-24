#include "vision-core/depth_estimation/depth_estimation_preprocessor.hpp"

namespace vision_core {

DepthAnythingV2Preprocessor::DepthAnythingV2Preprocessor(const cv::Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          true, // normalize to [0,1]
          true, // ImageNet normalization
          true  // BGR to RGB
      }) {}

} // namespace vision_core
