#include "neuriplo/tasks/keypoint_matching/keypoint_matching_preprocessor.hpp"

namespace neuriplo_tasks {

KeypointMatchingPreprocessor::KeypointMatchingPreprocessor(const Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          true,  // scale to [0, 1]
          false, // no ImageNet normalization
          true   // BGR to RGB
      }) {}

std::vector<std::vector<uint8_t>> KeypointMatchingPreprocessor::preprocess_pair(const ImageView& image1,
                                                                                const ImageView& image2) const {
    std::vector<std::vector<uint8_t>> buffers;
    buffers.reserve(2);
    buffers.push_back(preprocess(image1));
    buffers.push_back(preprocess(image2));
    return buffers;
}

} // namespace neuriplo_tasks
