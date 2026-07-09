#include "neuriplo/tasks/open_vocab_detection/grounding_dino_postprocessor.hpp"

#include "neuriplo/tasks/core/tensor_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>

namespace neuriplo_tasks {

namespace {

float sigmoid(float x) {
    if (x >= 0.0F) {
        const float exp_neg = std::exp(-x);
        return 1.0F / (1.0F + exp_neg);
    }
    const float exp_pos = std::exp(x);
    return exp_pos / (1.0F + exp_pos);
}

std::string normalizeName(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (char c : value) {
        if (c == '-' || c == '_' || std::isspace(static_cast<unsigned char>(c)) != 0) {
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

const Tensor* findTensorByNames(const std::vector<Tensor>& tensors, const std::vector<std::string>& output_names,
                                const std::vector<std::string>& candidates) {
    for (size_t i = 0; i < output_names.size() && i < tensors.size(); ++i) {
        const std::string name = normalizeName(output_names[i]);
        for (const auto& c : candidates) {
            if (name == normalizeName(c)) {
                return &tensors[i];
            }
        }
    }
    return nullptr;
}

vision::Rect makeRectFromCenterBox(float cx, float cy, float w, float h, const vision::Size& frame_size) {
    const float x1 = std::max(0.0F, cx - w * 0.5F);
    const float y1 = std::max(0.0F, cy - h * 0.5F);
    const float x2 = std::min(static_cast<float>(frame_size.width), cx + w * 0.5F);
    const float y2 = std::min(static_cast<float>(frame_size.height), cy + h * 0.5F);
    return vision::Rect(static_cast<int>(std::round(x1)), static_cast<int>(std::round(y1)),
                        static_cast<int>(std::round(x2 - x1)), static_cast<int>(std::round(y2 - y1)));
}

} // namespace

// ─── Construction ────────────────────────────────────────────────────────────

GroundingDinoPostprocessor::GroundingDinoPostprocessor(const vision::Size& input_size, float confidence_threshold,
                                                       float text_threshold, std::vector<std::string> prompt_labels,
                                                       std::vector<std::string> output_names,
                                                       std::vector<std::pair<int, int>> phrase_token_ranges)
    : input_size_(input_size), confidence_threshold_(confidence_threshold), text_threshold_(text_threshold),
      prompt_labels_(std::move(prompt_labels)), output_names_(std::move(output_names)),
      phrase_token_ranges_(std::move(phrase_token_ranges)) {}

// ─── Postprocessing ──────────────────────────────────────────────────────────

std::vector<OpenVocabDetection> GroundingDinoPostprocessor::postprocess(const std::vector<Tensor>& tensors,
                                                                        const vision::Size& frame_size) {
    std::vector<OpenVocabDetection> results;
    if (tensors.size() < 2) {
        return results;
    }

    const Tensor* boxes_ptr = findTensorByNames(tensors, output_names_, {"pred_boxes", "boxes"});
    const Tensor* logits_ptr = findTensorByNames(tensors, output_names_, {"pred_logits", "logits"});

    const Tensor& boxes = boxes_ptr != nullptr ? *boxes_ptr : tensors[0];
    const Tensor& logits = logits_ptr != nullptr ? *logits_ptr : tensors[1];

    if (boxes.shape.empty() || logits.shape.empty()) {
        return results;
    }

    // Expected shapes: boxes [batch, num_queries, 4], logits [batch, num_queries, seq_len]
    const int64_t batch = boxes.shape.size() >= 3 ? boxes.shape[0] : 1;
    const int64_t num_queries = boxes.shape[boxes.shape.size() - 2];
    const int64_t box_dims = boxes.shape.back();
    const int64_t seq_len = logits.shape.back();

    if (num_queries <= 0 || box_dims != 4 || seq_len <= 0) {
        return results;
    }

    const size_t per_query_boxes = static_cast<size_t>(box_dims);
    const size_t per_query_logits = static_cast<size_t>(seq_len);
    const size_t batch_boxes_stride = static_cast<size_t>(num_queries) * per_query_boxes;
    const size_t batch_logits_stride = static_cast<size_t>(num_queries) * per_query_logits;

    const size_t boxes_needed = static_cast<size_t>(batch) * batch_boxes_stride;
    const size_t logits_needed = static_cast<size_t>(batch) * batch_logits_stride;
    if (boxes.data.size() < boxes_needed || logits.data.size() < logits_needed) {
        return results;
    }

    for (int64_t b = 0; b < batch; ++b) {
        const size_t boxes_batch_offset = static_cast<size_t>(b) * batch_boxes_stride;
        const size_t logits_batch_offset = static_cast<size_t>(b) * batch_logits_stride;

        for (int64_t query_idx = 0; query_idx < num_queries; ++query_idx) {
            float best_score = -std::numeric_limits<float>::infinity();
            int best_phrase_idx = -1;

            if (!phrase_token_ranges_.empty()) {
                // Score each phrase by the maximum sigmoid(logit) over its token range
                for (size_t phrase_idx = 0; phrase_idx < phrase_token_ranges_.size(); ++phrase_idx) {
                    const int tok_start = phrase_token_ranges_[phrase_idx].first;
                    const int tok_end = phrase_token_ranges_[phrase_idx].second;

                    float phrase_score = -std::numeric_limits<float>::infinity();
                    for (int tok = tok_start; tok < tok_end && tok < static_cast<int>(seq_len); ++tok) {
                        const size_t offset =
                            logits_batch_offset + static_cast<size_t>(query_idx * seq_len + static_cast<int64_t>(tok));
                        const float s = sigmoid(tensorElementToFloat(logits.data[offset]));
                        if (s > phrase_score) {
                            phrase_score = s;
                        }
                    }

                    if (phrase_score > best_score) {
                        best_score = phrase_score;
                        best_phrase_idx = static_cast<int>(phrase_idx);
                    }
                }
            } else {
                // No phrase ranges: take the maximum over all tokens
                for (int64_t tok = 0; tok < seq_len; ++tok) {
                    const size_t offset = logits_batch_offset + static_cast<size_t>(query_idx * seq_len + tok);
                    const float s = sigmoid(tensorElementToFloat(logits.data[offset]));
                    if (s > best_score) {
                        best_score = s;
                    }
                }
            }

            if (best_score < confidence_threshold_ || best_score < text_threshold_) {
                continue;
            }

            const size_t box_offset = boxes_batch_offset + static_cast<size_t>(query_idx * 4);
            float cx = tensorElementToFloat(boxes.data[box_offset]);
            float cy = tensorElementToFloat(boxes.data[box_offset + 1]);
            float bw = tensorElementToFloat(boxes.data[box_offset + 2]);
            float bh = tensorElementToFloat(boxes.data[box_offset + 3]);

            // Grounding DINO always outputs normalised [cx, cy, w, h]
            if (std::max({std::fabs(cx), std::fabs(cy), std::fabs(bw), std::fabs(bh)}) <= 1.5F) {
                cx *= static_cast<float>(frame_size.width);
                cy *= static_cast<float>(frame_size.height);
                bw *= static_cast<float>(frame_size.width);
                bh *= static_cast<float>(frame_size.height);
            } else if (input_size_.width > 0 && input_size_.height > 0 &&
                       (frame_size.width != input_size_.width || frame_size.height != input_size_.height)) {
                const float sx = static_cast<float>(frame_size.width) / static_cast<float>(input_size_.width);
                const float sy = static_cast<float>(frame_size.height) / static_cast<float>(input_size_.height);
                cx *= sx;
                cy *= sy;
                bw *= sx;
                bh *= sy;
            }

            std::string label;
            if (best_phrase_idx >= 0 && static_cast<size_t>(best_phrase_idx) < prompt_labels_.size()) {
                label = prompt_labels_[static_cast<size_t>(best_phrase_idx)];
            }

            results.emplace_back(makeRectFromCenterBox(cx, cy, bw, bh, frame_size), best_score, best_phrase_idx,
                                 std::move(label));
        }
    }

    return results;
}

} // namespace neuriplo_tasks
