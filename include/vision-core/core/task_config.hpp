#pragma once
#include <map>
#include <string>
#include <vector>

namespace vision_core {

struct TaskConfig {
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.45f;
    float mask_threshold = 0.50f;
    float text_threshold = 0.10f;
    int top_k = 5;
    int max_text_queries = 16;
    bool apply_softmax = true;
    bool cache_text_features = true;
    std::string tokenizer_vocab_path;
    std::string tokenizer_merges_path;
    std::string tokenizer_vocab_json;
    std::string tokenizer_merges_text;
    // BERT tokenizer assets (used by Grounding DINO)
    std::string bert_tokenizer_vocab_path;
    std::string bert_tokenizer_vocab_text;
    std::vector<std::string> text_prompts;

    // Extensible key/value bag for task-specific params.
    // e.g. extra_params["text_prompt"] = "detect all cars";
    std::map<std::string, std::string> extra_params;
};

} // namespace vision_core
