#pragma once

#include <cstdint>
#include <variant>
#include <vector>
#include <opencv2/opencv.hpp>

namespace vision_core {

using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

/**
 * @brief Optical flow result from RAFT
 */
struct OpticalFlowResult {
    cv::Mat raw_flow;         // Raw flow field (CV_32FC2)
    cv::Mat flow_visualization; // Colored visualization (CV_8UC3)
    double max_displacement;   // Maximum displacement magnitude
    
    OpticalFlowResult(const cv::Mat& raw, const cv::Mat& vis, double max_disp)
        : raw_flow(raw), flow_visualization(vis), max_displacement(max_disp) {}
};

/**
 * @brief RAFT optical flow postprocessing
 * 
 * Processes RAFT model output to generate optical flow fields.
 * Expected input: Two consecutive frames
 * Expected output: Flow field [batch, 2, H, W] where channels are (u, v)
 */
class RaftPostprocessor {
public:
    /**
     * @brief Process RAFT output tensor
     * 
     * @param output Raw output data [batch, 2, H, W] - flow in (u, v) format
     * @param shape Output tensor shape
     * @param frame_size Original/target image size for resizing
     * @return Optical flow result with raw flow and visualization
     */
    [[nodiscard]] static OpticalFlowResult postprocess(
        const TensorElement* output,
        const std::vector<int64_t>& shape,
        const cv::Size& frame_size
    );

private:
    /**
     * @brief Create color wheel for flow visualization
     */
    static cv::Mat make_color_wheel();

    /**
     * @brief Convert flow field to color visualization
     * 
     * @param flow_mat Flow field (CV_32FC2)
     * @return Color-coded visualization (CV_8UC3)
     */
    static cv::Mat flow_to_color(const cv::Mat& flow_mat);
};

} // namespace vision_core
