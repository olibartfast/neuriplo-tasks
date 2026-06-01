#pragma once

#include "vision-core/core/result_types.hpp"
#include "vision-core/core/task_interface.hpp"

#include <vector>

namespace vision_core {

/**
 * @brief Postprocessor for video classification models
 *
 * Takes logits tensor [1, num_classes], applies softmax,
 * returns top-K VideoClassification results.
 */
class VideoClassificationPostprocessor {
  public:
    VideoClassificationPostprocessor(int top_k, bool apply_softmax);

    std::vector<VideoClassification> postprocess(const std::vector<TensorElement>& output,
                                                 const std::vector<int64_t>& shape);

  private:
    int top_k_;
    bool apply_softmax_;

    static void applySoftmax(std::vector<float>& logits);
};

} // namespace vision_core
