#pragma once

#include "vision-core/open_vocab_detection/postprocessor.hpp"

#include <string>
#include <vector>

namespace vision_core {

class OWLv2Postprocessor : public OpenVocabPostprocessor {
  public:
    OWLv2Postprocessor(const cv::Size& input_size, float confidence_threshold, float text_threshold,
                       std::vector<std::string> prompt_labels, std::vector<std::string> output_names);

    std::vector<OpenVocabDetection> postprocess(const std::vector<Tensor>& tensors,
                                                const cv::Size& frame_size) override;

  private:
    cv::Size input_size_;
    float confidence_threshold_;
    float text_threshold_;
    std::vector<std::string> prompt_labels_;
    std::vector<std::string> output_names_;
};

} // namespace vision_core
