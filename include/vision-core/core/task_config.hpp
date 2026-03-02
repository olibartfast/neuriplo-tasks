#pragma once
#include <map>
#include <string>

namespace vision_core {

struct TaskConfig {
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.45f;
    float mask_threshold = 0.50f;
    int top_k = 5;
    bool apply_softmax = true;

    // Extensible key/value bag for task-specific params.
    // e.g. extra_params["text_prompt"] = "detect all cars";
    std::map<std::string, std::string> extra_params;
};

} // namespace vision_core
