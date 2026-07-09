#pragma once

#include "neuriplo/tasks/optical_flow/optical_flow_postprocessor.hpp"

namespace neuriplo_tasks {

class RaftPostprocessor : public OpticalFlowPostprocessor {
  public:
    RaftPostprocessor();

    std::vector<OpticalFlow> postprocess(const std::vector<TensorElement>& flow_output,
                                         const std::vector<int64_t>& shape, const Size& frame_size) override;

  private:
    Image visualizeFlow(const ImageView& flow_x, const ImageView& flow_y);
    Image makeColorwheel();
};

} // namespace neuriplo_tasks
