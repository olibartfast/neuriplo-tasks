#include "vision-core/gaussian_splatting/lgm_postprocessor.hpp"

#include "vision-core/core/tensor_utils.hpp"

#include <stdexcept>
#include <variant>

namespace vision_core {

namespace {

// Indices within a single Gaussian feature vector (length 14)
constexpr int IDX_X = 0;
constexpr int IDX_Y = 1;
constexpr int IDX_Z = 2;
constexpr int IDX_SCALE_X = 3;
constexpr int IDX_SCALE_Y = 4;
constexpr int IDX_SCALE_Z = 5;
constexpr int IDX_ROT_W = 6;
constexpr int IDX_ROT_X = 7;
constexpr int IDX_ROT_Y = 8;
constexpr int IDX_ROT_Z = 9;
constexpr int IDX_OPACITY = 10;
constexpr int IDX_SH_R = 11;
constexpr int IDX_SH_G = 12;
constexpr int IDX_SH_B = 13;

constexpr int GAUSSIAN_FEATURES = 14;

} // namespace

GaussianSplatting LgmPostprocessor::postprocess(const std::vector<TensorElement>& tensor_data,
                                                const std::vector<int64_t>& shape) {
    GaussianSplatting result;

    if (tensor_data.empty() || shape.empty()) {
        return result;
    }

    // Accept layouts: [G, 14] or [N, G, 14]
    int batch_size = 1;
    int num_gaussians_per_batch = 0;
    int feature_dim = 0;

    if (shape.size() == 2) {
        num_gaussians_per_batch = static_cast<int>(shape[0]);
        feature_dim = static_cast<int>(shape[1]);
    } else if (shape.size() == 3) {
        batch_size = static_cast<int>(shape[0]);
        num_gaussians_per_batch = static_cast<int>(shape[1]);
        feature_dim = static_cast<int>(shape[2]);
    } else {
        return result;
    }

    if (feature_dim < GAUSSIAN_FEATURES) {
        return result;
    }

    const int total_gaussians = batch_size * num_gaussians_per_batch;
    const auto expected = static_cast<int64_t>(total_gaussians) * static_cast<int64_t>(feature_dim);
    if (static_cast<int64_t>(tensor_data.size()) < expected) {
        return result;
    }

    result.gaussians.resize(static_cast<size_t>(total_gaussians));
    result.num_gaussians = total_gaussians;

    for (int i = 0; i < total_gaussians; ++i) {
        const int base = i * feature_dim;
        Gaussian3D& g = result.gaussians[static_cast<size_t>(i)];

        g.x = tensorElementToFloat(tensor_data[static_cast<size_t>(base + IDX_X)]);
        g.y = tensorElementToFloat(tensor_data[static_cast<size_t>(base + IDX_Y)]);
        g.z = tensorElementToFloat(tensor_data[static_cast<size_t>(base + IDX_Z)]);
        g.scale_x = tensorElementToFloat(tensor_data[static_cast<size_t>(base + IDX_SCALE_X)]);
        g.scale_y = tensorElementToFloat(tensor_data[static_cast<size_t>(base + IDX_SCALE_Y)]);
        g.scale_z = tensorElementToFloat(tensor_data[static_cast<size_t>(base + IDX_SCALE_Z)]);
        g.rot_w = tensorElementToFloat(tensor_data[static_cast<size_t>(base + IDX_ROT_W)]);
        g.rot_x = tensorElementToFloat(tensor_data[static_cast<size_t>(base + IDX_ROT_X)]);
        g.rot_y = tensorElementToFloat(tensor_data[static_cast<size_t>(base + IDX_ROT_Y)]);
        g.rot_z = tensorElementToFloat(tensor_data[static_cast<size_t>(base + IDX_ROT_Z)]);
        g.opacity = tensorElementToFloat(tensor_data[static_cast<size_t>(base + IDX_OPACITY)]);
        g.sh_r = tensorElementToFloat(tensor_data[static_cast<size_t>(base + IDX_SH_R)]);
        g.sh_g = tensorElementToFloat(tensor_data[static_cast<size_t>(base + IDX_SH_G)]);
        g.sh_b = tensorElementToFloat(tensor_data[static_cast<size_t>(base + IDX_SH_B)]);
    }

    return result;
}

} // namespace vision_core
