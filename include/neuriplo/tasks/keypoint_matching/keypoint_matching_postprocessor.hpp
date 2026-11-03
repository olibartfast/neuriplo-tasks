#pragma once

#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

#include <vector>

namespace neuriplo_tasks {

/**
 * @brief Interface for keypoint matching postprocessing
 *
 * Decoders consume the ordered output tensors of one image pair and return
 * keypoints rescaled to the original image sizes. Each family's expected
 * tensor layout is documented on its concrete postprocessor and must be
 * verified against a real ONNX export before the decoder is implemented.
 */
class KeypointMatchingPostprocessor {
  public:
    virtual ~KeypointMatchingPostprocessor() = default;

    /**
     * @brief Decode one pair's matches
     *
     * @param tensors Output tensors, ordered as ModelInfo::output_names
     * @param source_size Original size of the first image in the pair
     * @param target_size Original size of the second image in the pair
     * @return One KeypointMatching result
     */
    virtual KeypointMatching postprocess(const std::vector<Tensor>& tensors, const Size& source_size,
                                         const Size& target_size) = 0;
};

} // namespace neuriplo_tasks
