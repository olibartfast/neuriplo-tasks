#include "neuriplo/tasks/keypoint_matching/lightglue_preprocessor.hpp"

#include "image_ops.hpp"

#include <cstring>
#include <stdexcept>

namespace neuriplo_tasks {

namespace {

// torchvision's rgb_to_grayscale weights, as used by the LightGlue-ONNX
// SuperPoint pipeline.
constexpr float kRedWeight = 0.2989f;
constexpr float kGreenWeight = 0.5870f;
constexpr float kBlueWeight = 0.1140f;

/**
 * @brief Write one image into its slot of the packed [2, C, H, W] buffer
 *
 * The slot is planar: `channels` planes of `plane` floats.
 */
void packImage(const ImageView& image, const Size& input_size, int channels, float* slot) {
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

    if (channels == 1) {
        for (std::size_t p = 0; p < plane; ++p) {
            float gray = pixels[p];
            if (source_channels >= 3) {
                const std::size_t offset = p * static_cast<std::size_t>(source_channels);
                gray =
                    kRedWeight * pixels[offset] + kGreenWeight * pixels[offset + 1] + kBlueWeight * pixels[offset + 2];
            }
            slot[p] = gray;
        }
        return;
    }

    for (int channel = 0; channel < channels; ++channel) {
        float* out = slot + static_cast<std::size_t>(channel) * plane;
        for (std::size_t p = 0; p < plane; ++p) {
            out[p] = pixels[p * static_cast<std::size_t>(source_channels) + static_cast<std::size_t>(channel)];
        }
    }
}

} // namespace

LightGluePreprocessor::LightGluePreprocessor(const Size& input_size, int channels)
    : KeypointMatchingPreprocessor(input_size), channels_(channels == 3 ? 3 : 1) {}

std::vector<std::vector<uint8_t>> LightGluePreprocessor::preprocess_pair(const ImageView& source,
                                                                         const ImageView& target) const {
    const Size input_size = config_.input_size;
    const std::size_t plane = static_cast<std::size_t>(input_size.width) * static_cast<std::size_t>(input_size.height);
    const std::size_t slot_values = static_cast<std::size_t>(channels_) * plane;
    const std::size_t slot_bytes = slot_values * sizeof(float);

    std::vector<uint8_t> packed(2 * slot_bytes);
    auto* base = reinterpret_cast<float*>(packed.data());
    packImage(source, input_size, channels_, base);
    packImage(target, input_size, channels_, base + slot_values);

    std::vector<std::vector<uint8_t>> buffers;
    buffers.push_back(std::move(packed));
    return buffers;
}

} // namespace neuriplo_tasks
