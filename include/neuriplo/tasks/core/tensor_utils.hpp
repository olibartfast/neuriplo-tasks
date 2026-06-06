#pragma once

#include "neuriplo/tasks/core/task_interface.hpp"

namespace neuriplo_tasks {

[[nodiscard]] inline float tensorElementToFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float { return static_cast<float>(value); }, element);
}

[[nodiscard]] inline int tensorElementToInt(const TensorElement& element) {
    return std::visit([](auto&& value) -> int { return static_cast<int>(value); }, element);
}

} // namespace neuriplo_tasks
