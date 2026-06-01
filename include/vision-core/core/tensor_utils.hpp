#pragma once

#include "vision-core/core/task_interface.hpp"

namespace vision_core {

[[nodiscard]] inline float tensorElementToFloat(const TensorElement& element) {
    return std::visit([](auto&& value) -> float { return static_cast<float>(value); }, element);
}

[[nodiscard]] inline int tensorElementToInt(const TensorElement& element) {
    return std::visit([](auto&& value) -> int { return static_cast<int>(value); }, element);
}

} // namespace vision_core
