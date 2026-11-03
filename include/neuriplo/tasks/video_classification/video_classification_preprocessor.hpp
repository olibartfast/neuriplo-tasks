#pragma once

#include "neuriplo/tasks/core/preprocessor.hpp"

#include <array>

namespace neuriplo_tasks {

/**
 * @brief VideoMAE preprocessor
 *
 * Direct resize to target size (224x224), rescale /255, ImageNet normalization.
 * mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
 */
class VideoMAEPreprocessor : public Preprocessor {
  public:
    explicit VideoMAEPreprocessor(const Size& input_size);

    using Preprocessor::preprocess;
    [[nodiscard]] std::vector<uint8_t> preprocess(const ImageView& image) const override;
    [[nodiscard]] bool supportsRawPixelOutput() const override { return false; }

  private:
    static constexpr std::array<float, 3> kMean = {0.485f, 0.456f, 0.406f};
    static constexpr std::array<float, 3> kStd = {0.229f, 0.224f, 0.225f};
};

/**
 * @brief ViViT preprocessor
 *
 * Aspect-preserving resize (shortest edge 256), center crop (224x224),
 * rescale *(1/127.5) with offset (-1), ImageNet normalization.
 * mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
 */
class VivitPreprocessor : public Preprocessor {
  public:
    explicit VivitPreprocessor(const Size& input_size);

    using Preprocessor::preprocess;
    [[nodiscard]] std::vector<uint8_t> preprocess(const ImageView& image) const override;
    [[nodiscard]] bool supportsRawPixelOutput() const override { return false; }

  private:
    static constexpr int kShortestEdge = 256;
    static constexpr std::array<float, 3> kMean = {0.485f, 0.456f, 0.406f};
    static constexpr std::array<float, 3> kStd = {0.229f, 0.224f, 0.225f};
};

/**
 * @brief TimeSformer preprocessor
 *
 * Aspect-preserving resize (shortest edge 224), center crop (224x224),
 * rescale /255, custom normalization.
 * mean=[0.45,0.45,0.45], std=[0.225,0.225,0.225]
 */
class TimeSformerPreprocessor : public Preprocessor {
  public:
    explicit TimeSformerPreprocessor(const Size& input_size);

    using Preprocessor::preprocess;
    [[nodiscard]] std::vector<uint8_t> preprocess(const ImageView& image) const override;
    [[nodiscard]] bool supportsRawPixelOutput() const override { return false; }

  private:
    static constexpr int kShortestEdge = 224;
    static constexpr std::array<float, 3> kMean = {0.45f, 0.45f, 0.45f};
    static constexpr std::array<float, 3> kStd = {0.225f, 0.225f, 0.225f};
};

/**
 * @brief PE Video preprocessor
 *
 * Direct resize to 336x336, rescale /255, normalize with
 * mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] (facebook/pe-av-large video processor).
 */
class PeVideoPreprocessor : public Preprocessor {
  public:
    explicit PeVideoPreprocessor(const Size& input_size);

    using Preprocessor::preprocess;
    [[nodiscard]] std::vector<uint8_t> preprocess(const ImageView& image) const override;
    [[nodiscard]] bool supportsRawPixelOutput() const override { return false; }

  private:
    static constexpr std::array<float, 3> kMean = {0.5f, 0.5f, 0.5f};
    static constexpr std::array<float, 3> kStd = {0.5f, 0.5f, 0.5f};
};

/**
 * @brief V-JEPA 2 preprocessor
 *
 * Aspect-preserving resize (shortest edge = crop x 256/224, 292 for a 256 crop),
 * center crop, rescale /255, ImageNet normalization
 * (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
 * Matches transformers' VJEPA2VideoProcessor defaults.
 */
class Vjepa2Preprocessor : public Preprocessor {
  public:
    explicit Vjepa2Preprocessor(const Size& input_size);

    using Preprocessor::preprocess;
    [[nodiscard]] std::vector<uint8_t> preprocess(const ImageView& image) const override;
    [[nodiscard]] bool supportsRawPixelOutput() const override { return false; }

  private:
    static constexpr std::array<float, 3> kMean = {0.485f, 0.456f, 0.406f};
    static constexpr std::array<float, 3> kStd = {0.229f, 0.224f, 0.225f};
};

/**
 * @brief VideoPrism preprocessor
 *
 * Direct resize to 288x288 (bicubic), rescale /255, **no** normalization.
 * Matches transformers' VideoPrism processor defaults (do_normalize=False).
 */
class VideoPrismPreprocessor : public Preprocessor {
  public:
    explicit VideoPrismPreprocessor(const Size& input_size);

    using Preprocessor::preprocess;
    [[nodiscard]] std::vector<uint8_t> preprocess(const ImageView& image) const override;
    [[nodiscard]] bool supportsRawPixelOutput() const override { return false; }
};

} // namespace neuriplo_tasks
