#pragma once

#include "neuriplo/tasks/core/preprocessor.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

#include <string>
#include <vector>

namespace neuriplo_tasks {

class BaseTask : public TaskInterface {
  public:
    explicit BaseTask(const ModelInfo& model_info, std::string empty_input_message = "Empty input image provided");

    std::vector<std::vector<uint8_t>> preprocess(const std::vector<cv::Mat>& imgs) final;
    std::vector<Result> postprocess(const cv::Size& frame_size, const std::vector<Tensor>& tensors) final;

  protected:
    [[nodiscard]] virtual const Preprocessor& getPreprocessor() const = 0;
    [[nodiscard]] virtual bool validateOutputs(const std::vector<Tensor>& tensors) const;
    virtual std::vector<Result> decode(const cv::Size& frame_size, const std::vector<Tensor>& tensors) = 0;

    template <typename Value> [[nodiscard]] static std::vector<Result> toResults(const std::vector<Value>& values) {
        std::vector<Result> results;
        results.reserve(values.size());
        for (const auto& value : values) {
            results.emplace_back(value);
        }
        return results;
    }

  private:
    std::string empty_input_message_;
};

} // namespace neuriplo_tasks
