#include "vision-core/core/task_pipeline.hpp"

#include <stdexcept>
#include <utility>

namespace vision_core {

SequentialTaskPipeline::SequentialTaskPipeline(std::vector<TaskPipelineStage> stages) {
    for (auto& stage : stages) {
        addStage(std::move(stage));
    }
}

void SequentialTaskPipeline::addStage(TaskPipelineStage stage) {
    if (!stage) {
        throw std::invalid_argument("TaskPipeline stage must be callable");
    }

    stages_.push_back(std::move(stage));
}

size_t SequentialTaskPipeline::stageCount() const noexcept { return stages_.size(); }

std::vector<Result> SequentialTaskPipeline::run(const std::vector<Result>& inputs) const {
    std::vector<Result> current = inputs;

    for (const auto& stage : stages_) {
        current = stage(current);
    }

    return current;
}

} // namespace vision_core
