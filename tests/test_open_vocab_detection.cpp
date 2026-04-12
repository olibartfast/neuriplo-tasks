#include "vision-core/core/task_factory.hpp"
#include "vision-core/open_vocab_detection/bert_tokenizer.hpp"
#include "vision-core/open_vocab_detection/grounding_dino_postprocessor.hpp"

#include <gtest/gtest.h>

using namespace vision_core;

namespace {

ModelInfo makeOwlv2Info() {
    ModelInfo info;
    info.input_shapes = {{1, 3, 960, 960}, {2, 8}, {2, 8}};
    info.input_formats = {"FORMAT_NCHW", "FORMAT_NCHW", "FORMAT_NCHW"};
    info.input_names = {"pixel_values", "input_ids", "attention_mask"};
    info.output_names = {"pred_boxes", "logits"};
    info.input_types = {CV_32F, CV_32S, CV_32S};
    return info;
}

// Minimal BERT vocab (line number = token ID)
// IDs mirror bert-base-uncased special tokens: [PAD]=0, [UNK]=100, [CLS]=101, [SEP]=102
std::string makeMinimalBertVocab() {
    // Build a string where each line is a token and the 0-based line index is the ID.
    // We need IDs 0, 100, 101, 102 for special tokens, plus content tokens.
    std::string vocab;
    vocab += "[PAD]\n"; // 0
    for (int i = 1; i < 100; ++i) {
        vocab += "[unused" + std::to_string(i) + "]\n";
    }
    vocab += "[UNK]\n"; // 100
    vocab += "[CLS]\n"; // 101
    vocab += "[SEP]\n"; // 102
    vocab += "[MASK]\n"; // 103
    // add common tokens we'll use in tests
    for (int i = 104; i < 200; ++i) {
        vocab += "[unused" + std::to_string(i) + "]\n";
    }
    vocab += ".\n";    // 200  – dot separator
    vocab += "cat\n";  // 201
    vocab += "dog\n";  // 202
    vocab += "car\n";  // 203
    return vocab;
}

ModelInfo makeGroundingDinoInfo() {
    ModelInfo info;
    info.input_shapes = {{1, 3, 800, 800}, {1, 16}, {1, 16}};
    info.input_formats = {"FORMAT_NCHW", "FORMAT_NCHW", "FORMAT_NCHW"};
    info.input_names = {"pixel_values", "input_ids", "attention_mask"};
    info.output_names = {"pred_boxes", "pred_logits"};
    info.input_types = {CV_32F, CV_32S, CV_32S};
    return info;
}

} // namespace

// ─── OWLv2 tests (unchanged) ─────────────────────────────────────────────────

TEST(OpenVocabDetectionTest, PreprocessProducesImageAndTextInputs) {
    TaskConfig cfg;
    cfg.text_prompts = {"cat", "dog"};
    cfg.tokenizer_vocab_json = R"({"<|endoftext|>":49407,"cat":10,"dog":11,"cat</w>":10,"dog</w>":11})";
    cfg.tokenizer_merges_text = "#version: 0.2\nc a\n";

    auto task = TaskFactory::createTaskInstance("owlv2", makeOwlv2Info(), cfg);
    cv::Mat image = cv::Mat::zeros(32, 32, CV_8UC3);
    const auto buffers = task->preprocess({image});

    ASSERT_EQ(buffers.size(), 3U);
    EXPECT_FALSE(buffers[0].empty());
    EXPECT_EQ(buffers[1].size(), static_cast<size_t>(2 * 8 * sizeof(int64_t)));
    EXPECT_EQ(buffers[2].size(), static_cast<size_t>(2 * 8 * sizeof(int64_t)));
}

TEST(OpenVocabDetectionTest, PostprocessReturnsOpenVocabDetectionResults) {
    TaskConfig cfg;
    cfg.text_prompts = {"cat", "dog"};

    auto task = TaskFactory::createTaskInstance("owlv2", makeOwlv2Info(), cfg);

    Tensor boxes(std::vector<TensorElement>{0.5f, 0.5f, 0.25f, 0.25f}, {1, 1, 4});
    Tensor logits(std::vector<TensorElement>{-3.0f, 4.0f}, {1, 1, 2});

    const auto results = task->postprocess(cv::Size(640, 480), {boxes, logits});
    ASSERT_EQ(results.size(), 1U);
    ASSERT_TRUE(std::holds_alternative<OpenVocabDetection>(results[0]));
    const auto& det = std::get<OpenVocabDetection>(results[0]);
    EXPECT_EQ(det.prompt_index, 1);
    EXPECT_EQ(det.label, "dog");
    EXPECT_GT(det.score, 0.9f);
}

TEST(OpenVocabDetectionTest, PostprocessUsesOutputNamesAndObjectness) {
    ModelInfo info = makeOwlv2Info();
    info.output_names = {"objectness_logits", "pred_boxes", "logits"};

    TaskConfig cfg;
    cfg.text_prompts = {"cat", "dog"};

    auto task = TaskFactory::createTaskInstance("owlv2", info, cfg);

    Tensor objectness(std::vector<TensorElement>{-10.0f}, {1, 1});
    Tensor boxes(std::vector<TensorElement>{0.5f, 0.5f, 0.25f, 0.25f}, {1, 1, 4});
    Tensor logits(std::vector<TensorElement>{-3.0f, 4.0f}, {1, 1, 2});

    const auto filtered = task->postprocess(cv::Size(640, 480), {objectness, boxes, logits});
    EXPECT_TRUE(filtered.empty());
}

// ─── BertTokenizer unit tests ─────────────────────────────────────────────────

TEST(BertTokenizerTest, EncodeSpecialTokensOnly) {
    BertTokenizer tok(makeMinimalBertVocab(), true);
    const auto [ids, mask] = tok.encode("", 8);
    ASSERT_EQ(static_cast<int>(ids.size()), 8);
    EXPECT_EQ(ids[0], BertTokenizer::CLS_TOKEN);
    EXPECT_EQ(ids[1], BertTokenizer::SEP_TOKEN);
    EXPECT_EQ(ids[2], BertTokenizer::PAD_TOKEN);
    EXPECT_EQ(mask[0], 1);
    EXPECT_EQ(mask[1], 1);
    EXPECT_EQ(mask[2], 0);
}

TEST(BertTokenizerTest, EncodeKnownTokens) {
    BertTokenizer tok(makeMinimalBertVocab(), true);
    const auto [ids, mask] = tok.encode("cat dog", 8);
    // Expect: [CLS]=101, cat=201, dog=202, [SEP]=102, [PAD]=0 ...
    ASSERT_GE(static_cast<int>(ids.size()), 4);
    EXPECT_EQ(ids[0], BertTokenizer::CLS_TOKEN);
    EXPECT_EQ(ids[1], 201); // cat
    EXPECT_EQ(ids[2], 202); // dog
    EXPECT_EQ(ids[3], BertTokenizer::SEP_TOKEN);
    EXPECT_EQ(mask[3], 1);
    EXPECT_EQ(mask[4], 0);
}

TEST(BertTokenizerTest, EncodePhrasesReturnsCorrectRanges) {
    BertTokenizer tok(makeMinimalBertVocab(), true);
    // Phrases: "cat" (1 token), "dog" (1 token)
    // Full sequence: [CLS] cat . dog . [SEP] [PAD]...
    // cat range: [1,2), dog range: [3,4)
    const auto enc = tok.encodePhrases({"cat", "dog"}, 16);
    ASSERT_EQ(enc.phrase_token_ranges.size(), 2U);
    EXPECT_EQ(enc.phrase_token_ranges[0].first, 1);
    EXPECT_EQ(enc.phrase_token_ranges[0].second, 2);
    EXPECT_EQ(enc.phrase_token_ranges[1].first, 3);
    EXPECT_EQ(enc.phrase_token_ranges[1].second, 4);
    // Token IDs in content area
    EXPECT_EQ(enc.input_ids[0], BertTokenizer::CLS_TOKEN);
    EXPECT_EQ(enc.input_ids[1], 201); // cat
    EXPECT_EQ(enc.input_ids[2], 200); // dot separator
    EXPECT_EQ(enc.input_ids[3], 202); // dog
    EXPECT_EQ(enc.input_ids[4], 200); // dot separator
    EXPECT_EQ(enc.input_ids[5], BertTokenizer::SEP_TOKEN);
}

TEST(BertTokenizerTest, AttentionMaskPaddingIsZero) {
    BertTokenizer tok(makeMinimalBertVocab(), true);
    const auto enc = tok.encodePhrases({"cat"}, 16);
    // Count real tokens: [CLS] cat . [SEP] → 4 tokens with mask=1, rest 0
    int real = 0;
    for (int m : enc.attention_mask) {
        real += m;
    }
    EXPECT_EQ(real, 4); // CLS, cat, dot, SEP
}

// ─── Grounding DINO task-level tests ─────────────────────────────────────────

TEST(GroundingDinoTest, TaskCreationSucceeds) {
    ModelInfo info = makeGroundingDinoInfo();
    TaskConfig cfg;
    cfg.text_prompts = {"cat", "dog"};
    cfg.bert_tokenizer_vocab_text = makeMinimalBertVocab();

    auto task = TaskFactory::createTaskInstance("groundingdino", info, cfg);
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(task->getTaskType(), TaskType::OpenVocabDetection);
}

TEST(GroundingDinoTest, PreprocessProducesCorrectBufferSizes) {
    ModelInfo info = makeGroundingDinoInfo();
    TaskConfig cfg;
    cfg.text_prompts = {"cat", "dog"};
    cfg.bert_tokenizer_vocab_text = makeMinimalBertVocab();

    auto task = TaskFactory::createTaskInstance("groundingdino", info, cfg);
    cv::Mat image = cv::Mat::zeros(32, 32, CV_8UC3);
    const auto buffers = task->preprocess({image});

    // 3 inputs: pixel_values, input_ids, attention_mask
    ASSERT_EQ(buffers.size(), 3U);
    EXPECT_FALSE(buffers[0].empty()); // image
    // input_ids and attention_mask: sequence length = 16, each element int64_t
    EXPECT_EQ(buffers[1].size(), static_cast<size_t>(16 * sizeof(int64_t)));
    EXPECT_EQ(buffers[2].size(), static_cast<size_t>(16 * sizeof(int64_t)));
}

TEST(GroundingDinoTest, PreprocessWithTokenTypeIds) {
    ModelInfo info = makeGroundingDinoInfo();
    // Add a token_type_ids input
    info.input_shapes.push_back({1, 16});
    info.input_formats.push_back("FORMAT_NCHW");
    info.input_names.push_back("token_type_ids");
    info.input_types.push_back(CV_32S);

    TaskConfig cfg;
    cfg.text_prompts = {"cat"};
    cfg.bert_tokenizer_vocab_text = makeMinimalBertVocab();

    auto task = TaskFactory::createTaskInstance("groundingdino", info, cfg);
    cv::Mat image = cv::Mat::zeros(32, 32, CV_8UC3);
    const auto buffers = task->preprocess({image});

    ASSERT_EQ(buffers.size(), 4U);
    // token_type_ids buffer must be all zeros
    const auto& ttids_buf = buffers[3];
    ASSERT_EQ(ttids_buf.size(), static_cast<size_t>(16 * sizeof(int64_t)));
    std::vector<int64_t> ttids(16);
    std::memcpy(ttids.data(), ttids_buf.data(), ttids_buf.size());
    for (int64_t v : ttids) {
        EXPECT_EQ(v, 0);
    }
}

TEST(GroundingDinoTest, PostprocessReturnsDetectionForHighScoreToken) {
    // Two phrases: cat (token range [1,2)), dog (token range [3,4))
    // seq_len = 8. For query 0: logit[3] is high → dog wins.
    std::vector<std::pair<int, int>> phrase_ranges = {{1, 2}, {3, 4}};
    GroundingDinoPostprocessor pp(cv::Size(800, 800), 0.1f, 0.1f, {"cat", "dog"},
                                  {"pred_boxes", "pred_logits"}, phrase_ranges);

    // pred_boxes: [1, 1, 4] → one query, normalised cx cy w h
    Tensor boxes(std::vector<TensorElement>{0.5f, 0.5f, 0.25f, 0.25f}, {1, 1, 4});

    // pred_logits: [1, 1, 8] → one query, 8 text tokens
    // Position 3 (dog range) has high logit, position 1 (cat range) has low logit
    std::vector<TensorElement> logit_data(8, TensorElement{-5.0f});
    logit_data[3] = TensorElement{5.0f}; // dog token gets high score
    Tensor logits(logit_data, {1, 1, 8});

    const auto results = pp.postprocess({boxes, logits}, cv::Size(640, 480));
    ASSERT_EQ(results.size(), 1U);
    EXPECT_EQ(results[0].prompt_index, 1); // dog
    EXPECT_EQ(results[0].label, "dog");
    EXPECT_GT(results[0].score, 0.9f);
}

TEST(GroundingDinoTest, PostprocessFiltersLowScoreQueries) {
    std::vector<std::pair<int, int>> phrase_ranges = {{1, 2}};
    GroundingDinoPostprocessor pp(cv::Size(800, 800), 0.9f, 0.9f, {"cat"},
                                  {"pred_boxes", "pred_logits"}, phrase_ranges);

    Tensor boxes(std::vector<TensorElement>{0.5f, 0.5f, 0.25f, 0.25f}, {1, 1, 4});
    // All logits negative → sigmoid < 0.5 → below threshold
    Tensor logits(std::vector<TensorElement>(4, TensorElement{-5.0f}), {1, 1, 4});

    const auto results = pp.postprocess({boxes, logits}, cv::Size(640, 480));
    EXPECT_TRUE(results.empty());
}

TEST(GroundingDinoTest, PostprocessNoPhraseRangesFallsBackToMaxToken) {
    // When phrase_token_ranges is empty, the postprocessor takes the max over all tokens
    GroundingDinoPostprocessor pp(cv::Size(800, 800), 0.5f, 0.5f, {},
                                  {"pred_boxes", "pred_logits"}, {});

    Tensor boxes(std::vector<TensorElement>{0.5f, 0.5f, 0.25f, 0.25f}, {1, 1, 4});
    std::vector<TensorElement> logit_data(4, TensorElement{-5.0f});
    logit_data[2] = TensorElement{5.0f};
    Tensor logits(logit_data, {1, 1, 4});

    const auto results = pp.postprocess({boxes, logits}, cv::Size(640, 480));
    ASSERT_EQ(results.size(), 1U);
    EXPECT_GT(results[0].score, 0.9f);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
