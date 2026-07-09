#pragma once

#include "neuriplo/tasks/core/image.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace neuriplo_tasks {

/**
 * @brief Image format specification
 */
enum class ImageFormat : uint8_t {
    NCHW, // Channel-first: [N, C, H, W]
    NHWC  // Channel-last: [N, H, W, C]
};

/**
 * @brief Data type specification
 */
enum class DataType : uint8_t { FLOAT32, UINT8, INT32, INT64 };

/**
 * @brief Preprocessing configuration
 */
struct PreprocessConfig {
    Size input_size; // Target input size
    ImageFormat format = ImageFormat::NCHW;
    DataType data_type = DataType::FLOAT32;
    bool normalize = true;            // Normalize to [0, 1]
    bool apply_imagenet_norm = false; // Apply ImageNet mean/std
    bool bgr_to_rgb = true;           // Convert BGR to RGB
};

/**
 * @brief Base preprocessor class
 *
 * Provides common preprocessing operations for computer vision tasks.
 */
class Preprocessor {
  public:
    explicit Preprocessor(const PreprocessConfig& config);
    virtual ~Preprocessor() = default;

    /**
     * @brief Preprocess a single image
     *
     * @param image Input image (BGR format, interleaved HxWxC)
     * @return Preprocessed data as byte vector
     */
    [[nodiscard]] virtual std::vector<uint8_t> preprocess(const ImageView& image) const;

    /**
     * @brief Preprocess multiple images
     *
     * @param images Input images
     * @return Vector of preprocessed data for each image
     */
    [[nodiscard]] virtual std::vector<std::vector<uint8_t>> preprocess(const std::vector<Image>& images) const;

  protected:
    /**
     * @brief Core preprocessing implementation
     *
     * @param image Input image
     * @param target_size Target size for resizing
     * @param format Output format (NCHW or NHWC)
     * @param data_type Output data type
     * @return Preprocessed data
     */
    [[nodiscard]] std::vector<uint8_t> preprocess_image(const ImageView& image, const Size& target_size,
                                                        ImageFormat format, DataType data_type) const;

    /**
     * @brief Apply ImageNet normalization (in-place on a Float32 3-channel image)
     */
    void apply_imagenet_normalization(Image& image) const;

    PreprocessConfig config_;

  private:
    static constexpr std::array<float, 3> kImageNetMean = {0.485f, 0.456f, 0.406f};
    static constexpr std::array<float, 3> kImageNetStd = {0.229f, 0.224f, 0.225f};
};

} // namespace neuriplo_tasks
