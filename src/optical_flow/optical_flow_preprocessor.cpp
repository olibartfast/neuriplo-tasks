#include "neuriplo/tasks/optical_flow/optical_flow_preprocessor.hpp"

#include "image_ops.hpp"

namespace neuriplo_tasks {

RaftPreprocessor::RaftPreprocessor(const vision::Size& input_size)
    : Preprocessor(PreprocessConfig{
          input_size, ImageFormat::NCHW, DataType::FLOAT32,
          false, // NO base normalization (RAFT does custom [-1,1] normalization)
          false, // no ImageNet norm
          true   // BGR to RGB
      }) {}

std::vector<std::vector<uint8_t>> RaftPreprocessor::preprocess_pair(const vision::ImageView& frame1,
                                                                    const vision::ImageView& frame2) const {
    std::vector<vision::ImageView> frames = {frame1, frame2};

    std::vector<std::vector<uint8_t>> preprocessed_frames;
    preprocessed_frames.reserve(2);

    for (const auto& frame : frames) {
        vision::Image processed =
            vision::Image::uninit(frame.width(), frame.height(), frame.channels(), frame.pixelType());
        std::memcpy(processed.raw(), frame.raw(), frame.sizeBytes());

        if (config_.bgr_to_rgb && processed.channels() == 3) {
            image_ops::swapBgrRgb(processed);
        }

        image_ops::Interpolation interp = image_ops::Interpolation::Linear;
        if (frame.width() > config_.input_size.width || frame.height() > config_.input_size.height) {
            interp = image_ops::Interpolation::Area;
        }
        processed = image_ops::resize(processed, config_.input_size, interp);

        processed.convertTo(vision::PixelType::Float32, 2.0 / 255.0, -1.0);

        std::vector<uint8_t> output;
        if (config_.format == ImageFormat::NCHW) {
            std::vector<vision::Image> channels = image_ops::splitChannels(processed.view());

            for (const auto& channel : channels) {
                const float* data = channel.data<float>();
                const size_t channel_size = channel.totalPixels() * sizeof(float);
                const uint8_t* byte_data = reinterpret_cast<const uint8_t*>(data);
                output.insert(output.end(), byte_data, byte_data + channel_size);
            }
        } else {
            const float* data = processed.data<float>();
            const size_t total_size = processed.sizeBytes();
            const uint8_t* byte_data = reinterpret_cast<const uint8_t*>(data);
            output.insert(output.end(), byte_data, byte_data + total_size);
        }

        preprocessed_frames.push_back(std::move(output));
    }

    return preprocessed_frames;
}

} // namespace neuriplo_tasks
