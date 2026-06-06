#pragma once

#include "neuriplo/tasks/core/result_types.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace neuriplo_tasks {

using TaskPipelineStage = std::function<std::vector<Result>(const std::vector<Result>&)>;

class TaskPipeline {
  public:
    virtual ~TaskPipeline() = default;

    [[nodiscard]] virtual std::vector<Result> run(const std::vector<Result>& inputs) const = 0;
};

class SequentialTaskPipeline final : public TaskPipeline {
  public:
    SequentialTaskPipeline() = default;
    explicit SequentialTaskPipeline(std::vector<TaskPipelineStage> stages);

    void addStage(TaskPipelineStage stage);

    [[nodiscard]] size_t stageCount() const noexcept;
    [[nodiscard]] std::vector<Result> run(const std::vector<Result>& inputs) const override;

  private:
    std::vector<TaskPipelineStage> stages_;
};

} // namespace neuriplo_tasks
