#pragma once

#include "neuriplo/tasks/core/batch_types.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace neuriplo_tasks {

/**
 * @brief Run existing postprocess and align results with the requested batch size.
 *
 * Calls `task.postprocess(frame_size, tensors)` and sets `batch_size` on the
 * returned `BatchPostprocessOutput`.
 *
 * **Per-index split:** when the task already returns one `Result` per batch
 * index (e.g. depth, ViTPose), those results are returned unchanged when
 * `results.size() == batch_size`.
 *
 * **Aggregate result:** when the task returns a single combined result
 * (e.g. Gaussian splatting multi-view), `results.size()` may be 1 while
 * `batch_size > 1`. No duplication is performed.
 *
 * **Variable detection count:** detection and instance segmentation may return
 * more than `batch_size` results (one `Detection` per box); `batch_size` stays
 * the consumer image count.
 *
 * **Tensor-led batch count:** for depth and pose tasks, if the postprocessor
 * emits more than one result whose count differs from `batch_size`, the helper
 * trusts the postprocessor count and updates `output.batch_size` accordingly.
 *
 * @throws std::invalid_argument if `batch_size <= 0` or exceeds
 *         `task.getModelInfo().max_batch_size_` when that value is positive.
 */
[[nodiscard]] BatchPostprocessOutput batchPostprocess(TaskInterface& task, const cv::Size& frame_size,
                                                      const std::vector<Tensor>& tensors, int batch_size);

} // namespace neuriplo_tasks
