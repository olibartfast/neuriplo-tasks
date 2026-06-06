#include "vision-core/depth_estimation/depth_anything_v2_postprocessor.hpp"

#include "vision-core/core/opencv_interop.hpp"
#include "vision-core/core/tensor_utils.hpp"

#include <algorithm>
#include <stdexcept>

namespace vision_core {

namespace {

struct DepthLayout {
    int batch{0};
    int height{0};
    int width{0};
};

DepthLayout parseDepthLayout(const std::vector<int64_t>& shape) {
    if (shape.size() == 2) {
        return {1, static_cast<int>(shape[0]), static_cast<int>(shape[1])};
    }

    if (shape.size() == 3) {
        return {static_cast<int>(shape[0]), static_cast<int>(shape[1]), static_cast<int>(shape[2])};
    }

    if (shape.size() == 4) {
        const int dim1 = static_cast<int>(shape[1]);
        if (dim1 == 1) {
            return {static_cast<int>(shape[0]), static_cast<int>(shape[2]), static_cast<int>(shape[3])};
        }

        const int dim4 = static_cast<int>(shape[3]);
        if (dim4 == 1) {
            return {static_cast<int>(shape[0]), static_cast<int>(shape[1]), static_cast<int>(shape[2])};
        }
    }

    return {};
}

} // namespace

std::vector<DepthEstimation> DepthAnythingV2Postprocessor::postprocess(const std::vector<TensorElement>& depth_output,
                                                                       const std::vector<int64_t>& shape,
                                                                       const cv::Size& frame_size) {
    if (depth_output.empty() || shape.empty()) {
        return {};
    }

    const DepthLayout layout = parseDepthLayout(shape);
    if (layout.batch <= 0 || layout.height <= 0 || layout.width <= 0) {
        return {};
    }

    const int64_t expected =
        static_cast<int64_t>(layout.batch) * static_cast<int64_t>(layout.height) * static_cast<int64_t>(layout.width);
    if (static_cast<int64_t>(depth_output.size()) < expected) {
        return {};
    }

    const size_t map_size = static_cast<size_t>(layout.height) * static_cast<size_t>(layout.width);
    std::vector<DepthEstimation> results;
    results.reserve(static_cast<size_t>(layout.batch));

    for (int batch_index = 0; batch_index < layout.batch; ++batch_index) {
        const size_t start_offset = static_cast<size_t>(batch_index) * map_size;

        cv::Mat depth(layout.height, layout.width, CV_32FC1);
        float* depth_ptr = depth.ptr<float>(0);

        for (size_t idx = 0; idx < map_size; ++idx) {
            depth_ptr[idx] = tensorElementToFloat(depth_output[start_offset + idx]);
        }

        if (frame_size.width > 0 && frame_size.height > 0 &&
            (depth.cols != frame_size.width || depth.rows != frame_size.height)) {
            cv::Mat resized;
            cv::resize(depth, resized, frame_size, 0.0, 0.0, cv::INTER_CUBIC);
            depth = resized;
        }

        double min_value = 0.0;
        double max_value = 0.0;
        cv::minMaxLoc(depth, &min_value, &max_value);

        cv::Mat normalized_depth = cv::Mat::zeros(depth.size(), CV_32FC1);
        if (max_value > min_value) {
            const double scale = 1.0 / (max_value - min_value);
            depth.convertTo(normalized_depth, CV_32FC1, scale, -min_value * scale);
        }

        DepthEstimation estimation;
        estimation.depth = fromCvMat(depth);
        estimation.normalized_depth = fromCvMat(normalized_depth);
        estimation.min_depth = static_cast<float>(min_value);
        estimation.max_depth = static_cast<float>(max_value);
        results.push_back(std::move(estimation));
    }

    return results;
}

} // namespace vision_core
