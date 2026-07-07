#include "neuriplo/tasks/depth_estimation/depth_anything_v2_postprocessor.hpp"

#include "image_ops.hpp"
#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <algorithm>
#include <stdexcept>

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
                                                                       const Size& frame_size) {
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

        Image depth = Image::uninit(layout.width, layout.height, 1, PixelType::Float32);
        float* depth_ptr = depth.data<float>();

        for (size_t idx = 0; idx < map_size; ++idx) {
            depth_ptr[idx] = tensorElementToFloat(depth_output[start_offset + idx]);
        }

        if (frame_size.width > 0 && frame_size.height > 0 &&
            (depth.width() != frame_size.width || depth.height() != frame_size.height)) {
            depth = image_ops::resize(depth, frame_size.width, frame_size.height, image_ops::Interpolation::Cubic);
        }

        double min_value = 0.0;
        double max_value = 0.0;
        image_ops::minMax(depth.view(), min_value, max_value);

        Image normalized_depth;
        if (max_value > min_value) {
            const double scale = 1.0 / (max_value - min_value);
            normalized_depth = depth.convertedTo(PixelType::Float32, scale, -min_value * scale);
        } else {
            normalized_depth = Image::zeros(depth.width(), depth.height(), 1, PixelType::Float32);
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
