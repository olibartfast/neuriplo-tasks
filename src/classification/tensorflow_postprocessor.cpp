#include "vision-core/classification/tensorflow_postprocessor.hpp"

#include "vision-core/classification/classification_postprocessor.hpp"

namespace vision_core {

TensorflowPostprocessor::TensorflowPostprocessor(int top_k, bool apply_softmax)
    : top_k_(top_k), apply_softmax_(apply_softmax) {}

std::vector<Classification> TensorflowPostprocessor::postprocess(const std::vector<TensorElement>& output,
                                                                 const std::vector<int64_t>& shape) {
    return postprocessClassificationLogits(output, shape, top_k_, apply_softmax_);
}

void TensorflowPostprocessor::applySoftmax(std::vector<float>& logits) { (void)logits; }

} // namespace vision_core
