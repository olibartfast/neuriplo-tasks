#pragma once

#include "vision-core/core/preprocessor.hpp"
#include "vision-core/core/task_config.hpp"
#include "vision-core/core/task_interface.hpp"
#include "vision-core/open_vocab_detection/clip_tokenizer.hpp"
#include "vision-core/open_vocab_detection/postprocessor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace vision_core {

class OpenVocabDetectionTask : public TaskInterface {
  public:
    enum class ModelType : uint8_t { Unknown, OWLV2, OWLVIT };

    OpenVocabDetectionTask(const ModelInfo& model_info, const std::string& model_name, const TaskConfig& config);

    TaskType getTaskType() override { return TaskType::OpenVocabDetection; }

    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) override;

    std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<Tensor>& tensors) override;

  private:
    ModelType model_type_;
    TaskConfig config_;
    std::unique_ptr<Preprocessor> image_preprocessor_;
    std::unique_ptr<OpenVocabPostprocessor> postprocessor_;
    std::unique_ptr<ClipTokenizer> tokenizer_;

    static ModelType detectModelType(const std::string& model_name);
    static std::vector<std::string> extractPrompts(const TaskConfig& config);
    static cv::Size extractInputSize(const ModelInfo& model_info);

    std::pair<std::vector<int32_t>, std::vector<int32_t>> encodePrompts(int context_length) const;
};

} // namespace vision_core
