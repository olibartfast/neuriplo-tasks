#pragma once

#include "neuriplo/tasks/core/image.hpp"
#include "neuriplo/tasks/core/result_types.hpp"

#include <cstddef>
#include <vector>

namespace neuriplo_tasks {

/**
 * @brief Input to batch preprocess helpers.
 *
 * For standard image-batch tasks, `images.size()` is the batch dimension `N`
 * passed to inference after the consumer stacks per-image buffers.
 *
 * Not all tasks treat `images` as independent batch items — see
 * docs/batch_support_matrix.md (video frames, flow pairs, multi-view splatting).
 */
struct BatchRequest {
    std::vector<Image> images;
};

/**
 * @brief Output of batch preprocess helpers.
 *
 * Wraps existing `TaskInterface::preprocess` output plus explicit batch metadata.
 *
 * Invariants (image-batch tasks):
 * - `batch_size ==` number of images in the originating `BatchRequest`.
 * - `buffers` is whatever `TaskInterface::preprocess` returned (often one buffer
 *   per image; multi-input models may return `buffers.size() != batch_size`).
 *
 * When the consumer sets `ModelInfo.batch_size_`, helpers should keep
 * `batch_size` aligned with that value for the inference request.
 */
struct BatchPreprocessOutput {
    std::vector<std::vector<uint8_t>> buffers;
    int batch_size{0};
};

/**
 * @brief Output of batch postprocess helpers.
 *
 * Invariants:
 * - `batch_size` is the number of batch indices the consumer requested.
 * - For tasks that split batched tensors, `results.size() == batch_size`.
 * - When a task returns a single aggregate result (e.g. Gaussian splatting),
 *   `results.size()` may be 1 while `batch_size > 1`; helpers document this
 *   per domain in batch_support_matrix.md.
 *
 * Empty tensors: match existing task behaviour — typically an empty `results`
 * vector (some tasks throw on empty images at preprocess time instead).
 */
struct BatchPostprocessOutput {
    std::vector<Result> results;
    int batch_size{0};
};

/**
 * @brief True when batch metadata matches the image list length.
 */
[[nodiscard]] inline bool imageBatchSizeMatches(const BatchRequest& request, int batch_size) noexcept {
    return batch_size >= 0 && static_cast<size_t>(batch_size) == request.images.size();
}

/**
 * @brief True when postprocess output carries one result per batch index.
 */
[[nodiscard]] inline bool postprocessResultsMatchBatchSize(const BatchPostprocessOutput& output) noexcept {
    return output.batch_size >= 0 && static_cast<size_t>(output.batch_size) == output.results.size();
}

} // namespace neuriplo_tasks
