#include "vision-core/classification/classification_preprocessor.hpp"

namespace vision_core {

ClassifierPreprocessor::ClassifierPreprocessor(const cv::Size& input_size, bool use_imagenet_norm)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          true,              // normalize to [0,1]
          use_imagenet_norm, // optional ImageNet normalization
          true               // BGR to RGB
      }) {}

TorchvisionPreprocessor::TorchvisionPreprocessor(const cv::Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          true, // normalize to [0,1]
          true, // ImageNet normalization
          true  // BGR to RGB
      }) {}

TensorflowPreprocessor::TensorflowPreprocessor(const cv::Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size,
          ImageFormat::NHWC, // TensorFlow uses channel-last
          DataType::UINT8,   // TensorFlow often uses uint8
          false,             // no normalization (keeps 0-255)
          false,             // no ImageNet normalization
          false              // TensorFlow models expect BGR
      }) {}

ViTPreprocessor::ViTPreprocessor(const cv::Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          true, // normalize to [0,1]
          true, // ImageNet normalization
          true  // BGR to RGB
      }) {}

} // namespace vision_core
