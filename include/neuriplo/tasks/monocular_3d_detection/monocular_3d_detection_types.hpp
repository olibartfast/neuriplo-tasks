#pragma once

#include "neuriplo/tasks/core/bounding_box.hpp"

#include <array>

namespace neuriplo_tasks {

/**
 * @brief A projected image-space corner of a monocular 3D cuboid.
 *
 * UrbanOmniDetect predicts eight 2D image coordinates representing the
 * projection of a 3D object cuboid. Confidence describes confidence in the
 * cuboid geometry rather than human-joint visibility.
 */
struct CuboidKeypoint {
    float x{0.0f};
    float y{0.0f};
    float confidence{0.0f};
};

/**
 * @brief Result of monocular 3D object detection represented by a projected cuboid.
 *
 * This is intentionally separate from PoseEstimation: the eight keypoints are
 * rigid-object cuboid corners, not 2D/3D human skeleton joints and not a
 * canonical 6-DoF (R,t) object pose.
 *
 * Corner convention used by UrbanOmniDetect v2:
 *   - corners 0..3: ground-contact corners
 *   - corners 4..7: roof corners
 *   - corner i+4 lies above ground corner i
 *   - corners 0 and 3 belong to the front face
 */
struct Monocular3DDetection {
    BoundingBox bbox; ///< 2D image-space detection box
    int class_id{-1};
    float class_confidence{0.0f};

    std::array<CuboidKeypoint, 8> projected_corners{};
    float geometry_confidence{0.0f};

    Monocular3DDetection() = default;
    Monocular3DDetection(const BoundingBox& box, float class_conf, int cls)
        : bbox(box), class_id(cls), class_confidence(class_conf) {}
};

} // namespace neuriplo_tasks
