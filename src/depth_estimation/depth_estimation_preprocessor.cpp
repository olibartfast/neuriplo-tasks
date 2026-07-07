#include "neuriplo/tasks/depth_estimation/depth_estimation_preprocessor.hpp"

namespace neuriplo_tasks {

DepthAnythingV2Preprocessor::DepthAnythingV2Preprocessor(const Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          true, // normalize to [0,1]
          true, // ImageNet normalization
          true  // BGR to RGB
      }) {}

} // namespace neuriplo_tasks
