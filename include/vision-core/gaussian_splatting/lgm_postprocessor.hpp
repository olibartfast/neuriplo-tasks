#pragma once

#include "vision-core/gaussian_splatting/gaussian_splatting_postprocessor.hpp"

namespace vision_core {

/**
 * @brief LGM / GRM Gaussian Splatting postprocessor
 *
 * Handles the standard feed-forward output layout used by LGM and GRM:
 *
 *   [N, G, 14]  or  [G, 14]
 *
 * where G is the number of Gaussians and each element contains:
 *   [0..2]  xyz position
 *   [3..5]  log-scale (sx, sy, sz)
 *   [6..9]  rotation quaternion (w, x, y, z)
 *   [10]    opacity logit
 *   [11..13] SH DC colour (r, g, b)
 */
class LgmPostprocessor : public GaussianSplattingPostprocessor {
  public:
    GaussianSplatting postprocess(const std::vector<TensorElement>& tensor_data,
                                  const std::vector<int64_t>& shape) override;

  private:
};

} // namespace vision_core
