#pragma once

#include "neuriplo/tasks/core/model_info.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace neuriplo_tasks {
namespace detail {
template <typename T> std::vector<uint8_t> packIntegralValues(const std::vector<int64_t>& values) {
    std::vector<T> typed;
    typed.reserve(values.size());
    for (const int64_t value : values) {
        const auto wide = static_cast<long double>(value);
        if (wide < static_cast<long double>(std::numeric_limits<T>::lowest()) ||
            wide > static_cast<long double>(std::numeric_limits<T>::max())) {
            throw std::invalid_argument("Input values do not fit the runtime tensor datatype");
        }
        const auto converted = static_cast<T>(value);
        if (static_cast<long double>(converted) != wide) {
            throw std::invalid_argument("Input values lose precision in the runtime tensor datatype");
        }
        typed.push_back(converted);
    }
    std::vector<uint8_t> bytes(typed.size() * sizeof(T));
    if (!bytes.empty()) {
        std::memcpy(bytes.data(), typed.data(), bytes.size());
    }
    return bytes;
}
} // namespace detail

/** Pack integral auxiliary values using this input's actual runtime metadata. */
[[nodiscard]] inline std::vector<uint8_t> packTensorInputValues(const ModelInfo& info, size_t index,
                                                                const std::vector<int64_t>& values) {
    const std::string datatype = modelInputDatatype(info, index);
    if (datatype == "INT64") {
        return detail::packIntegralValues<int64_t>(values);
    }
    if (datatype == "INT32") {
        return detail::packIntegralValues<int32_t>(values);
    }
    if (datatype == "FP32") {
        return detail::packIntegralValues<float>(values);
    }
    if (datatype == "UINT8") {
        return detail::packIntegralValues<uint8_t>(values);
    }
    if (datatype == "INT8") {
        return detail::packIntegralValues<int8_t>(values);
    }
    if (datatype == "BOOL") {
        for (const int64_t value : values) {
            if (value != 0 && value != 1) {
                throw std::invalid_argument("A runtime BOOL tensor requires boolean input values");
            }
        }
        return detail::packIntegralValues<uint8_t>(values);
    }
    const std::string name = index < info.input_names.size() ? info.input_names[index] : "#" + std::to_string(index);
    throw std::invalid_argument("Unsupported runtime datatype '" + datatype + "' for input '" + name + "'");
}
} // namespace neuriplo_tasks
