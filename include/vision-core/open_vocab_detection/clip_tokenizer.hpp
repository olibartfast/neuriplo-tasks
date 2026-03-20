#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vision_core {

/**
 * @brief CLIP byte-level BPE tokenizer
 *
 * Implements the same tokenization used by OpenAI CLIP and HuggingFace
 * OwlViTTokenizer / Owlv2Tokenizer.  Requires the two vocabulary files
 * that ship with every HuggingFace CLIP-based model checkpoint:
 *   - vocab.json   (token string → token id)
 *   - merges.txt   (ordered BPE merge rules, one pair per line)
 *
 * Usage:
 * @code
 *   ClipTokenizer tok("/path/to/vocab.json", "/path/to/merges.txt");
 *   auto [ids, mask] = tok.batchEncode({"a photo of a cat", "a dog"}, 16);
 * @endcode
 */
class ClipTokenizer {
  public:
    static constexpr int32_t SOT_TOKEN = 49406; ///< <|startoftext|>
    static constexpr int32_t EOT_TOKEN = 49407; ///< <|endoftext|>
    static constexpr int32_t PAD_TOKEN = 0;     ///< padding token id

    /**
     * @brief Construct the tokenizer from vocabulary and merge-rule files
     * @param vocab_file  Path to vocab.json
     * @param merges_file Path to merges.txt
     * @throws std::runtime_error if either file cannot be opened or parsed
     */
    ClipTokenizer(const std::string& vocab_file, const std::string& merges_file);

    /**
     * @brief Tokenize a single text string
     * @param text           Input text (UTF-8)
     * @param context_length Maximum number of tokens (padded / truncated)
     * @return Token IDs with SOT prepended and EOT appended, padded to context_length
     */
    [[nodiscard]] std::vector<int32_t> encode(const std::string& text, int context_length = 16) const;

    /**
     * @brief Tokenize a batch of text strings
     * @param texts          Input texts
     * @param context_length Sequence length for each query
     * @return Pair of flat arrays:
     *           first  – input_ids    [num_texts × context_length], row-major
     *           second – attention_mask [num_texts × context_length], 1 where real, 0 for pad
     */
    [[nodiscard]] std::pair<std::vector<int32_t>, std::vector<int32_t>>
    batchEncode(const std::vector<std::string>& texts, int context_length = 16) const;

  private:
    std::unordered_map<std::string, int32_t> vocab_;
    // Ordered merge rules: each entry is (left_piece, right_piece)
    std::vector<std::pair<std::string, std::string>> merges_;
    // byte value (0-255) → printable unicode string used by BPE
    std::unordered_map<uint8_t, std::string> byte_encoder_;
    // Merge rank lookup for O(1) pair lookup during BPE
    std::unordered_map<std::string, int> merge_rank_;

    /**
     * @brief Build the fixed byte-to-unicode mapping defined by CLIP
     */
    static std::unordered_map<uint8_t, std::string> buildByteEncoder();

    /**
     * @brief Apply BPE to a single pre-split word token
     * @param token  Space-separated unicode characters representing the word
     * @return Space-separated BPE pieces
     */
    [[nodiscard]] std::string bpe(const std::string& token) const;

    /**
     * @brief Split text into words and encode each through BPE
     */
    [[nodiscard]] std::vector<int32_t> tokenize(const std::string& text) const;
};

} // namespace vision_core
