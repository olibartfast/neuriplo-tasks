#pragma once

#include "neuriplo/tasks/core/image.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace neuriplo_tasks {

/**
 * @brief Whether an input with this shape carries an image
 *
 * Image tensors have rank 3 or more ([C,H,W], [N,C,H,W], [N,T,C,H,W]); size
 * tensors and token ids do not.
 */
[[nodiscard]] inline bool isImageInputShape(const std::vector<int64_t>& shape) noexcept { return shape.size() >= 3; }

/**
 * @brief Model information structure
 *
 * Contains metadata about a model's inputs and outputs.
 * Compatible with Triton model info but framework-agnostic.
 * Enhanced with neuriplo compatibility for unified tensor processing.
 */
struct ModelInfo {
    // ===== EXISTING FIELDS (backward compatible) =====
    std::vector<std::vector<int64_t>> input_shapes; ///< Input tensor shapes
    std::vector<std::string> input_formats;         ///< Input formats (FORMAT_NCHW, FORMAT_NHWC)
    std::vector<std::string> input_names;           ///< Input tensor names
    std::vector<std::string> output_names;          ///< Output tensor names
    std::vector<PixelType> input_types;             ///< Per-input data types
    int max_batch_size_{1};                         ///< Maximum batch size
    int batch_size_{1};                             ///< Current batch size

    // ===== NEW FIELDS (neuriplo compatibility) =====
    std::vector<std::vector<int64_t>> output_shapes; ///< Output tensor shapes (from neuriplo)
    std::vector<size_t> input_batch_sizes;           ///< Per-input batch sizes (from neuriplo)
    std::vector<size_t> output_batch_sizes;          ///< Per-output batch sizes (from neuriplo)

    // Lossless runtime scalar tags (FP32, INT32, INT64, ...), per input.
    // PixelType describes image storage and cannot represent all tensor types.
    // When populated, these tags are authoritative; legacy callers may keep
    // using input_types without this field.
    std::vector<std::string> input_datatypes;

    ModelInfo() = default;

    /**
     * @brief Add input layer information (neuriplo compatibility)
     * @param name Input tensor name
     * @param shape Input tensor shape
     * @param batch_size Batch size for this input
     */
    void addInput(const std::string& name, const std::vector<int64_t>& shape, size_t batch_size = 1) {
        input_names.push_back(name);
        input_shapes.push_back(shape);
        input_batch_sizes.push_back(batch_size);

        // Update global batch size info
        max_batch_size_ = std::max(max_batch_size_, static_cast<int>(batch_size));
        batch_size_ = std::max(batch_size_, static_cast<int>(batch_size));

        // Add default values for format and type if not specified
        if (input_formats.size() < input_names.size()) {
            input_formats.emplace_back("FORMAT_NCHW"); // Default format
        }
        if (input_types.size() < input_names.size()) {
            input_types.push_back(PixelType::Float32); // Default type
        }
    }

    /**
     * @brief Add output layer information (neuriplo compatibility)
     * @param name Output tensor name
     * @param shape Output tensor shape
     * @param batch_size Batch size for this output
     */
    void addOutput(const std::string& name, const std::vector<int64_t>& shape, size_t batch_size = 1) {
        output_names.push_back(name);
        output_shapes.push_back(shape);
        output_batch_sizes.push_back(batch_size);
    }
};

/** Resolve one input's scalar type without treating tensor types as pixel types. */
[[nodiscard]] inline std::string modelInputDatatype(const ModelInfo& info, size_t index) {
    if (!info.input_datatypes.empty()) {
        if (info.input_datatypes.size() != info.input_shapes.size() || index >= info.input_datatypes.size() ||
            info.input_datatypes[index].empty()) {
            throw std::invalid_argument("Missing runtime datatype for input #" + std::to_string(index));
        }
        return info.input_datatypes[index];
    }
    // Legacy source compatibility: respect the type the caller supplied.
    if (index < info.input_types.size()) {
        switch (info.input_types[index]) {
        case PixelType::Float32:
            return "FP32";
        case PixelType::UInt8:
            return "UINT8";
        case PixelType::Int32:
            return "INT32";
        }
    }
    throw std::invalid_argument("Missing datatype for input #" + std::to_string(index));
}

} // namespace neuriplo_tasks
