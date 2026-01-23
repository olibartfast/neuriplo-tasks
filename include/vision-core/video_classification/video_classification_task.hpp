#pragma once

#include "vision-core/core/task_interface.hpp"
#include "vision-core/core/preprocessor.hpp"
#include <memory>
#include <string>

namespace vision_core {

class VideoClassificationPostprocessor;

/**
 * @brief Video classification task for temporal video models
 *
 * Handles video classification models that process multiple frames:
 * - VideoMAE
 * - ViViT
 * - TimeSformer
 *
 * Input shape is 5D: [batch, num_frames, channels, height, width]
 * Each frame is preprocessed independently, then concatenated into a single buffer.
 */
class VideoClassificationTask : public TaskInterface {
public:
    enum class ModelType {
        VIDEOMAE,
        VIVIT,
        TIMESFORMER
    };

    explicit VideoClassificationTask(const ModelInfo& model_info,
                                     const std::string& model_name,
                                     int top_k = 5,
                                     bool apply_softmax = true);

    ~VideoClassificationTask() override;

    TaskType getTaskType() override { return TaskType::VideoClassification; }

    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override;

    std::vector<Result> postprocess(
        const cv::Size& frame_size,
        const std::vector<Tensor>& tensors) override;

private:
    ModelType model_type_;
    std::string model_name_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<VideoClassificationPostprocessor> postprocessor_;
    int top_k_;
    bool apply_softmax_;
    int num_frames_;
    int input_width_;
    int input_height_;

    static ModelType detectModelType(const std::string& model_name);

    std::unique_ptr<Preprocessor> createPreprocessor(ModelType type, const cv::Size& input_size);

    std::unique_ptr<VideoClassificationPostprocessor> createPostprocessor();

    /**
     * @brief Extract video input dimensions from 5D shape [batch, frames, C, H, W]
     */
    void extractVideoInputSize(const ModelInfo& model_info);
};

} // namespace vision_core
