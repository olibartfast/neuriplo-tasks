#include "neuriplo/tasks/keypoint_matching/flexglue_preprocessor.hpp"

#include "image_ops.hpp"

#include <cstring>
#include <stdexcept>

namespace neuriplo_tasks {

namespace {

/**
 * @brief Write one image as a planar RGB [0, 1] NCHW buffer
 *
 * FlexGlue's descriptor takes RGB (not grayscale) squashed to S x S with a
 * plain bilinear resize, matching upstream `describe()` (cv2 INTER_LINEAR,
 * no antialiasing).
 */
void packImage(const ImageView& image, const Size& input_size, float* slot) {
    if (image.empty()) {
        throw std::invalid_argument("Empty image provided in pair");
    }

    Image processed = Image::uninit(image.width(), image.height(), image.channels(), image.pixelType());
    std::memcpy(processed.raw(), image.raw(), image.sizeBytes());

    if (processed.channels() == 3) {
        image_ops::swapBgrRgb(processed);
    }
    if (processed.size() != input_size) {
        processed = image_ops::resize(processed, input_size, image_ops::Interpolation::Linear);
    }
    processed.convertTo(PixelType::Float32, 1.0 / 255.0);

    const std::size_t plane = static_cast<std::size_t>(input_size.width) * static_cast<std::size_t>(input_size.height);
    const int source_channels = processed.channels();
    const float* pixels = processed.data<float>();

    for (int channel = 0; channel < 3; ++channel) {
        float* out = slot + static_cast<std::size_t>(channel) * plane;
        for (std::size_t p = 0; p < plane; ++p) {
            const std::size_t offset = p * static_cast<std::size_t>(source_channels);
            out[p] = source_channels >= 3 ? pixels[offset + static_cast<std::size_t>(channel)] : pixels[p];
        }
    }
}

} // namespace

FlexGluePreprocessor::FlexGluePreprocessor(const Size& input_size) : KeypointMatchingPreprocessor(input_size) {}

std::vector<std::vector<uint8_t>> FlexGluePreprocessor::preprocess_pair(const ImageView& source,
                                                                        const ImageView& target) const {
    const Size input_size = config_.input_size;
    const std::size_t plane = static_cast<std::size_t>(input_size.width) * static_cast<std::size_t>(input_size.height);
    const std::size_t slot_values = 3 * plane;
    const std::size_t slot_bytes = slot_values * sizeof(float);

    std::vector<uint8_t> first(slot_bytes);
    packImage(source, input_size, reinterpret_cast<float*>(first.data()));
    std::vector<uint8_t> second(slot_bytes);
    packImage(target, input_size, reinterpret_cast<float*>(second.data()));

    std::vector<std::vector<uint8_t>> buffers;
    buffers.push_back(std::move(first));
    buffers.push_back(std::move(second));
    return buffers;
}

} // namespace neuriplo_tasks
