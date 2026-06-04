#pragma once

#include "vision-core/core/batch_types.hpp"
#include "vision-core/core/task_interface.hpp"

namespace vision_core {

/**
 * @brief Run existing per-image preprocess and attach batch metadata.
 *
 * Calls `task.preprocess(request.images)` and sets `batch_size` to
 * `request.images.size()`.
 *
 * @throws std::invalid_argument if `request.images` is empty or exceeds
 *         `task.getModelInfo().max_batch_size_` when that value is positive.
 */
[[nodiscard]] BatchPreprocessOutput batchPreprocess(TaskInterface& task, const BatchRequest& request);

} // namespace vision_core
