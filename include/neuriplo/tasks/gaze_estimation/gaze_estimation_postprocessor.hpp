#pragma once

#include "neuriplo/tasks/core/task_interface.hpp"

#include <array>

namespace neuriplo_tasks {

/**
 * @brief ETH-XGaze-style gaze postprocessor.
 *
 * Decodes the first output tensor per batch row:
 * - [B,2] (or [2]): pitch, yaw in radians (ETH-XGaze normalized-space order).
 * - [B,3] (or [3]): xyz gaze vector, normalised with pitch/yaw derived back.
 *
 * Angle convention follows the ETH-XGaze / OpenGaze normalized camera space
 * (as in ptgaze): forward is -Z and
 *   direction = [-cos(pitch)*sin(yaw), -sin(pitch), -cos(pitch)*cos(yaw)]
 *   pitch = asin(-y), yaw = atan2(-x, -z)
 * so a horizontal flip of the input negates yaw while pitch is unchanged.
 *
 * An optional second tensor ([B,1], [B], or scalar) carries confidence in
 * [0,1]; when absent confidence is 0.
 */
class GazePostprocessor {
  public:
    explicit GazePostprocessor(float confidence_default = 0.0f);

    std::vector<GazeEstimation> postprocess(const std::vector<Tensor>& tensors) const;

    [[nodiscard]] static std::array<float, 3> directionFromPitchYaw(float pitch, float yaw);
    static void pitchYawFromDirection(const std::array<float, 3>& direction, float& pitch, float& yaw);

  private:
    float confidence_default_;
};

} // namespace neuriplo_tasks
