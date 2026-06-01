#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace vision_core {

[[nodiscard]] inline int findOutputIndexByName(const std::vector<std::string>& output_names,
                                               const std::string& target_name, int default_index) {
    for (size_t i = 0; i < output_names.size(); ++i) {
        if (output_names[i] == target_name) {
            return static_cast<int>(i);
        }
    }

    return default_index;
}

} // namespace vision_core
