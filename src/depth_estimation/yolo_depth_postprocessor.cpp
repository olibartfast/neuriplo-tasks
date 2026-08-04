#include "neuriplo/tasks/depth_estimation/yolo_depth_postprocessor.hpp"

#include "image_ops.hpp"
#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <algorithm>
#include <cmath>

namespace neuriplo_tasks {

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
        if (shape[1] == 1) {
            return {static_cast<int>(shape[0]), static_cast<int>(shape[2]), static_cast<int>(shape[3])};
        }
        if (shape[3] == 1) {
            return {static_cast<int>(shape[0]), static_cast<int>(shape[1]), static_cast<int>(shape[2])};
        }
    }

    return {};
}

vision::Image restoreFrameGeometry(vision::Image depth, const vision::Size& frame_size) {
    if (frame_size.width <= 0 || frame_size.height <= 0 ||
        (depth.width() == frame_size.width && depth.height() == frame_size.height)) {
        return depth;
    }

    const double gain = std::min(static_cast<double>(depth.height()) / static_cast<double>(frame_size.height),
                                 static_cast<double>(depth.width()) / static_cast<double>(frame_size.width));
    const double pad_width =
        (static_cast<double>(depth.width()) - std::round(static_cast<double>(frame_size.width) * gain)) / 2.0;
    const double pad_height =
        (static_cast<double>(depth.height()) - std::round(static_cast<double>(frame_size.height) * gain)) / 2.0;
    const int left = static_cast<int>(std::round(pad_width - 0.1));
    const int top = static_cast<int>(std::round(pad_height - 0.1));
    const int right = depth.width() - static_cast<int>(std::round(pad_width + 0.1));
    const int bottom = depth.height() - static_cast<int>(std::round(pad_height + 0.1));

    if (left < 0 || top < 0 || right <= left || bottom <= top) {
        return {};
    }

    const vision::Rect source_roi(left, top, right - left, bottom - top);
    vision::Image cropped = vision::Image::uninit(source_roi.width, source_roi.height, 1, vision::PixelType::Float32);
    image_ops::copyRegion(depth.view(), source_roi, cropped, vision::Rect(0, 0, source_roi.width, source_roi.height));
    return image_ops::resize(cropped, frame_size.width, frame_size.height, image_ops::Interpolation::Linear);
}

} // namespace

std::vector<DepthEstimation> YoloDepthPostprocessor::postprocess(const std::vector<TensorElement>& depth_output,
                                                                 const std::vector<int64_t>& shape,
                                                                 const vision::Size& frame_size) {
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
        vision::Image depth = vision::Image::uninit(layout.width, layout.height, 1, vision::PixelType::Float32);
        float* depth_ptr = depth.data<float>();

        for (size_t index = 0; index < map_size; ++index) {
            depth_ptr[index] = tensorElementToFloat(depth_output[start_offset + index]);
        }

        depth = restoreFrameGeometry(std::move(depth), frame_size);
        if (depth.empty()) {
            return {};
        }

        double min_value = 0.0;
        double max_value = 0.0;
        image_ops::minMax(depth.view(), min_value, max_value);

        vision::Image normalized_depth;
        if (max_value > min_value) {
            const double scale = 1.0 / (max_value - min_value);
            normalized_depth = depth.convertedTo(vision::PixelType::Float32, scale, -min_value * scale);
        } else {
            normalized_depth = vision::Image::zeros(depth.width(), depth.height(), 1, vision::PixelType::Float32);
        }

        DepthEstimation estimation;
        estimation.min_depth = static_cast<float>(min_value);
        estimation.max_depth = static_cast<float>(max_value);
        estimation.depth = fromImage(std::move(depth));
        estimation.normalized_depth = fromImage(std::move(normalized_depth));
        results.push_back(std::move(estimation));
    }

    return results;
}

} // namespace neuriplo_tasks
