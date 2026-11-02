#include "neuriplo/tasks/keypoint_matching/superglue_preprocessor.hpp"

#include "image_ops.hpp"

#include <cstring>
#include <stdexcept>

namespace neuriplo_tasks {

namespace {

// Weights used by torchvision's rgb_to_grayscale (num_output_channels=3), which
// the transformers SuperGlue image processor applies after resizing.
constexpr float kRedWeight = 0.2989f;
constexpr float kGreenWeight = 0.5870f;
constexpr float kBlueWeight = 0.1140f;

/**
 * @brief Write one image into its slot of the packed [2, 3, H, W] buffer
 *
 * The slot is planar: three planes of `plane` floats, all holding the same
 * grayscale value.
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
    const int channels = processed.channels();
    const float* pixels = processed.data<float>();

    for (std::size_t p = 0; p < plane; ++p) {
        float gray = pixels[p];
        if (channels >= 3) {
            const std::size_t offset = p * static_cast<std::size_t>(channels);
            gray = kRedWeight * pixels[offset] + kGreenWeight * pixels[offset + 1] + kBlueWeight * pixels[offset + 2];
        }
        slot[p] = gray;
        slot[plane + p] = gray;
        slot[2 * plane + p] = gray;
    }
}

} // namespace

SuperGluePreprocessor::SuperGluePreprocessor(const Size& input_size) : KeypointMatchingPreprocessor(input_size) {}

std::vector<std::vector<uint8_t>> SuperGluePreprocessor::preprocess_pair(const ImageView& source,
                                                                         const ImageView& target) const {
    const Size input_size = config_.input_size;
    const std::size_t plane = static_cast<std::size_t>(input_size.width) * static_cast<std::size_t>(input_size.height);
    const std::size_t slot_values = 3 * plane;
    const std::size_t slot_bytes = slot_values * sizeof(float);

    std::vector<uint8_t> packed(2 * slot_bytes);
    auto* base = reinterpret_cast<float*>(packed.data());
    packImage(source, input_size, base);
    packImage(target, input_size, base + slot_values);

    std::vector<std::vector<uint8_t>> buffers;
    buffers.push_back(std::move(packed));
    return buffers;
}

} // namespace neuriplo_tasks
