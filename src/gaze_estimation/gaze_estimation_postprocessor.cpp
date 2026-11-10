#include "neuriplo/tasks/gaze_estimation/gaze_estimation_postprocessor.hpp"

#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace neuriplo_tasks {

GazePostprocessor::GazePostprocessor(float confidence_default) : confidence_default_(confidence_default) {}

std::array<float, 3> GazePostprocessor::directionFromPitchYaw(float pitch, float yaw) {
    // ETH-XGaze / OpenGaze normalized-camera convention (cf. ptgaze).
    const float cos_pitch = std::cos(pitch);
    std::array<float, 3> direction = {-std::sin(yaw) * cos_pitch, -std::sin(pitch), -std::cos(yaw) * cos_pitch};
    const float norm =
        std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
    if (norm > 1e-12f) {
        direction[0] /= norm;
        direction[1] /= norm;
        direction[2] /= norm;
        return direction;
    }
    return {0.0f, 0.0f, -1.0f};
}

void GazePostprocessor::pitchYawFromDirection(const std::array<float, 3>& direction, float& pitch, float& yaw) {
    float norm = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
    float x = direction[0];
    float y = direction[1];
    float z = direction[2];
    if (norm > 1e-12f) {
        x /= norm;
        y /= norm;
        z /= norm;
    } else {
        pitch = 0.0f;
        yaw = 0.0f;
        return;
    }
    const float clamped_y = std::max(-1.0f, std::min(1.0f, -y));
    pitch = std::asin(clamped_y);
    yaw = std::atan2(-x, -z);
}

std::vector<GazeEstimation> GazePostprocessor::postprocess(const std::vector<Tensor>& tensors) const {
    if (tensors.empty()) {
        return {};
    }
    const Tensor& primary = tensors[0];
    if (primary.shape.empty()) {
        return {};
    }

    // Accept [2]/[3] or [B,2]/[B,3]. Anything else is malformed.
    int batch = 1;
    int width = 0;
    if (primary.shape.size() == 1) {
        width = static_cast<int>(primary.shape[0]);
    } else if (primary.shape.size() == 2) {
        batch = static_cast<int>(primary.shape[0]);
        width = static_cast<int>(primary.shape[1]);
    } else {
        return {};
    }
    if ((width != 2 && width != 3) || batch <= 0) {
        return {};
    }
    const size_t expected = static_cast<size_t>(batch) * static_cast<size_t>(width);
    if (primary.data.size() < expected) {
        return {};
    }

    // Optional confidence tensor: [B,1], [B], or scalar broadcast.
    const Tensor* confidence_tensor = nullptr;
    int confidence_batch = 0;
    if (tensors.size() > 1) {
        confidence_tensor = &tensors[1];
        if (confidence_tensor->shape.size() == 1 || confidence_tensor->shape.size() == 2) {
            confidence_batch = static_cast<int>(confidence_tensor->shape[0]);
        } else if (confidence_tensor->shape.empty() && !confidence_tensor->data.empty()) {
            confidence_batch = 1;
        }
    }

    std::vector<GazeEstimation> results;
    results.reserve(static_cast<size_t>(batch));
    for (int b = 0; b < batch; ++b) {
        const size_t base = static_cast<size_t>(b) * static_cast<size_t>(width);
        GazeEstimation gaze;
        if (width == 2) {
            gaze.pitch = tensorElementToFloat(primary.data[base + 0]);
            gaze.yaw = tensorElementToFloat(primary.data[base + 1]);
            gaze.direction = directionFromPitchYaw(gaze.pitch, gaze.yaw);
        } else {
            std::array<float, 3> raw = {tensorElementToFloat(primary.data[base + 0]),
                                        tensorElementToFloat(primary.data[base + 1]),
                                        tensorElementToFloat(primary.data[base + 2])};
            const float norm = std::sqrt(raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]);
            if (norm > 1e-12f) {
                raw[0] /= norm;
                raw[1] /= norm;
                raw[2] /= norm;
            } else {
                raw = {0.0f, 0.0f, -1.0f};
            }
            gaze.direction = raw;
            pitchYawFromDirection(raw, gaze.pitch, gaze.yaw);
        }
        gaze.confidence = confidence_default_;
        if (confidence_tensor != nullptr && !confidence_tensor->data.empty()) {
            if (confidence_batch == batch) {
                gaze.confidence = tensorElementToFloat(confidence_tensor->data[static_cast<size_t>(b)]);
            } else if (confidence_batch == 1) {
                gaze.confidence = tensorElementToFloat(confidence_tensor->data[0]);
            }
            gaze.confidence = std::max(0.0f, std::min(1.0f, gaze.confidence));
        }
        results.push_back(gaze);
    }
    return results;
}

} // namespace neuriplo_tasks
