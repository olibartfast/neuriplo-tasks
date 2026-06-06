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
    explicit VideoMAEPreprocessor(const cv::Size& input_size);

    using Preprocessor::preprocess;
    [[nodiscard]] std::vector<uint8_t> preprocess(const cv::Mat& image) const override;

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
    explicit VivitPreprocessor(const cv::Size& input_size);

    using Preprocessor::preprocess;
    [[nodiscard]] std::vector<uint8_t> preprocess(const cv::Mat& image) const override;

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
    explicit TimeSformerPreprocessor(const cv::Size& input_size);

    using Preprocessor::preprocess;
    [[nodiscard]] std::vector<uint8_t> preprocess(const cv::Mat& image) const override;

  private:
    static constexpr int kShortestEdge = 224;
    static constexpr std::array<float, 3> kMean = {0.45f, 0.45f, 0.45f};
    static constexpr std::array<float, 3> kStd = {0.225f, 0.225f, 0.225f};
};

} // namespace neuriplo_tasks
