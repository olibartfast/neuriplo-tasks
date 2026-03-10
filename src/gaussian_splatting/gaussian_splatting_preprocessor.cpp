#include "vision-core/gaussian_splatting/gaussian_splatting_preprocessor.hpp"

namespace vision_core {

GaussianSplattingPreprocessor::GaussianSplattingPreprocessor(const cv::Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size,
          ImageFormat::NCHW,
          DataType::FLOAT32,
          true, // normalise to [0, 1]
          true, // ImageNet mean/std normalisation
          true  // BGR → RGB
      }) {}

} // namespace vision_core
